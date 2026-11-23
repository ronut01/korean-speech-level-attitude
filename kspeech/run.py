"""Inference runner (candidate-logit scoring and short generation) with resume.

    python -m kspeech.run --phase pilot            # 1,320 planned pilot evaluations
    python -m kspeech.run --phase main             # main score + neutral baseline + controls + generation audit
    python -m kspeech.run --phase main --part score --tasks ynat
"""
import argparse, json, platform, time
from pathlib import Path
from .common import CFG, LABELS, LETTERS, ROOT, load_stimuli
from .data import load_items
from .parse import parse_generation
from .prompts import conditions, options, user_message


def make_jobs(task, phase, part, stim, limit=None):
    items = load_items(task, phase)
    if limit:
        items = items[:limit]
    qids = [q["id"] for q in stim["quartets"]]
    if phase == "pilot":
        qids = qids[:2]
    sub = [it for it in items if it["gen"]]
    jobs = []
    if part == "score":
        jobs += [(it, c) for it in items for c in conditions(stim, qids)]
        jobs += [(it, {"ending": h, "att": "N", "q": None}) for it in items for h in "YB"]
        if phase == "main":
            jobs += [(it, {"ending": h, "att": "X", "q": None}) for it in sub for h in "YB"]
    else:
        jobs += [(it, c) for it in sub for c in conditions(stim, qids)]
    return jobs


def job_key(part, it, c):
    return f"{part}|{it['task']}|{it['guid']}|{c['ending']}{c['att']}|{c['q'] or '-'}"


def load_model(name, revision):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(name, revision=revision, padding_side="left")
    try:                                   # transformers >= 4.56 names the argument `dtype`
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=dtype)
    model = model.to(dev).eval()
    return tok, model, dev


def check_single_token(tok, chat_text, letters):
    """Each option letter must be exactly one token when it follows the assistant header."""
    base = tok(chat_text, add_special_tokens=False)["input_ids"]
    ids = []
    for l in letters:
        full = tok(chat_text + l, add_special_tokens=False)["input_ids"]
        assert full[:len(base)] == base and len(full) == len(base) + 1, f"letter {l} is not a single token"
        ids.append(full[-1])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "main"], required=True)
    ap.add_argument("--part", choices=["score", "gen", "all"], default="all")
    ap.add_argument("--tasks", nargs="+", default=list(CFG["tasks"]))
    ap.add_argument("--model", default=CFG["model"])
    ap.add_argument("--stimuli", default=None)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N items per task")
    ap.add_argument("--batch-size", type=int, default=CFG["batch_size"])
    ap.add_argument("--allow-provisional", action="store_true")
    a = ap.parse_args()

    import torch, transformers
    stim = load_stimuli(a.stimuli)
    if stim.get("provisional") and a.phase == "main" and not a.allow_provisional:
        raise SystemExit("stimuli/selected.json is provisional (no human validation). Run kspeech.ratings select first.")
    torch.manual_seed(CFG["seed"])
    tok, model, dev = load_model(a.model, CFG["model_revision"] if a.model == CFG["model"] else None)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = a.model.split("/")[-1]
    rev = CFG["model_revision"] if a.model == CFG["model"] else None
    meta = {"model": a.model, "model_revision": getattr(model.config, "_commit_hash", None) or rev, "device": dev,
            "dtype": str(model.dtype), "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "seed": CFG["seed"], "phase": a.phase,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
            "stimuli_provisional": bool(stim.get("provisional")), "started": time.strftime("%F %T")}

    for task in a.tasks:
        letters = LETTERS[:len(LABELS[task])]
        shots = load_items(task, "fewshot") * CFG["fewshot_per_class"][task]
        for part in (["score", "gen"] if a.part == "all" else [a.part]):
            path = out / f"{a.phase}_{part}_{task}_{tag}.jsonl"
            done = {json.loads(l)["key"] for l in open(path, encoding="utf-8")} if path.exists() else set()
            jobs = []
            for it, c in make_jobs(task, a.phase, part, stim, a.limit):
                key = job_key(part, it, c)
                if key in done:
                    continue
                msg = user_message(stim, c, it, shots)
                text = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                               add_generation_prompt=True)
                jobs.append((key, it, c, text))
            if not jobs:
                print(f"[{task}/{part}] nothing to do"); continue
            cand_ids = check_single_token(tok, jobs[0][3], letters)
            jobs.sort(key=lambda j: len(j[3]))
            print(f"[{task}/{part}] {len(jobs)} evaluations ({len(done)} already done)")
            t_start, n_done = time.time(), 0
            with open(path, "a", encoding="utf-8") as f:
                for b in range(0, len(jobs), a.batch_size):
                    batch = jobs[b:b + a.batch_size]
                    enc = tok([j[3] for j in batch], return_tensors="pt", padding=True,
                              add_special_tokens=False).to(dev)
                    n_tok = enc["attention_mask"].sum(1).tolist()
                    t0 = time.time()
                    with torch.no_grad():
                        if part == "score":
                            h = model.base_model(**enc).last_hidden_state[:, -1, :]   # left padding: last = final token
                            logits = model.get_output_embeddings()(h).float()
                            p = torch.softmax(logits, -1)[:, cand_ids].cpu()
                        else:
                            gen = model.generate(**enc, max_new_tokens=CFG["max_new_tokens"], do_sample=False,
                                                 pad_token_id=tok.pad_token_id)[:, enc["input_ids"].shape[1]:]
                    dt = time.time() - t0
                    for n, (key, it, c, text) in enumerate(batch):
                        opt = dict(options(it))
                        row = {"key": key, "task": task, "guid": it["guid"], "ending": c["ending"], "att": c["att"],
                               "q": c["q"], "gold": it["label"], "n_tokens": n_tok[n],
                               "over_limit": n_tok[n] > CFG["max_input_tokens"], "batch_sec": round(dt, 4)}
                        if part == "score":
                            probs = p[n].tolist()
                            letter = letters[max(range(len(probs)), key=probs.__getitem__)]
                            row.update(cand_mass=round(sum(probs), 6), probs=[round(x, 6) for x in probs],
                                       pred_letter=letter, pred=opt[letter])
                        else:
                            new = gen[n].tolist()
                            hit_cap = len(new) == CFG["max_new_tokens"] and tok.eos_token_id not in new \
                                and tok.pad_token_id not in new
                            raw = tok.decode(new, skip_special_tokens=True)
                            ps = parse_generation(raw, letters, {l: LABELS[task][ci] for l, ci in opt.items()})
                            row.update(raw=raw, hit_cap=hit_cap, **ps,
                                       pred=opt[ps["extracted"]] if ps["extracted"] else None)
                        row["correct"] = int(row["pred"] == it["label"])
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush(); n_done += len(batch)
                    if (b // a.batch_size) % 20 == 0:
                        el = time.time() - t_start
                        print(f"  {n_done}/{len(jobs)}  {n_done / el:.1f} eval/s  eta {(len(jobs) - n_done) / (n_done / el) / 60:.1f} min")
            meta[f"{part}_{task}"] = {"n": len(jobs), "seconds": round(time.time() - t_start, 1),
                                      "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else None}
    meta["finished"] = time.strftime("%F %T")
    with open(out / f"meta_{a.phase}_{tag}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
