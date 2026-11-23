"""Fixed item samples from KLUE (YNAT, NLI). Run once; the saved lists are the source of truth.

    python -m kspeech.data
"""
import random
from collections import defaultdict
from .common import CFG, LABELS, ROOT, norm_text, read_jsonl, text_hash, write_jsonl


def _raw(task, split):
    from datasets import load_dataset
    ds = load_dataset(CFG["dataset"], task, split=split, revision=CFG["dataset_revision"])
    names = ds.features["label"].names
    if task == "ynat":
        assert names == LABELS["ynat"], names
        return [{"guid": r["guid"], "text": {"title": r["title"]}, "label": r["label"],
                 "norm": norm_text(r["title"])} for r in ds]
    assert names == ["entailment", "neutral", "contradiction"], names
    return [{"guid": r["guid"], "text": {"premise": r["premise"], "hypothesis": r["hypothesis"]},
             "label": r["label"], "norm": norm_text(r["premise"] + "\t" + r["hypothesis"])}
            for r in ds if r["label"] in (0, 1, 2)]


def _sample(rows, quota, seed, banned=()):
    """Shuffle within gold class, drop duplicates / banned texts, keep quota[c] per class."""
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    seen, picked = set(banned), {}
    for c in sorted(by):
        rs = sorted(by[c], key=lambda r: r["guid"])
        random.Random(f"{seed}-{c}").shuffle(rs)
        out = []
        for r in rs:
            if len(out) == quota[c]:
                break
            if r["norm"] in seen:
                continue
            seen.add(r["norm"])
            out.append(r)
        assert len(out) == quota[c], (c, len(out))
        picked[c] = out
    return picked


def _finish(task, picked, gen_per_class=None, gen_total=None):
    """Interleave classes; attach option rotation, generation-subset flag and control quartet slot."""
    k = len(LABELS[task])
    items, depth = [], max(len(v) for v in picked.values())
    for j in range(depth):
        for c in sorted(picked):
            if j < len(picked[c]):
                r = picked[c][j]
                items.append({"guid": r["guid"], "task": task, "text": r["text"], "label": r["label"],
                              "hash": text_hash(r["norm"]), "j": j,
                              "rot": j % k,          # option letter i shows class (i + rot) % k
                              "cq": j % 4})          # quartet slot used for control comparisons
    for n, it in enumerate(items):
        it["gen"] = (it["j"] < gen_per_class) if gen_per_class is not None else (n < gen_total)
    return items


def build(task):
    t, seed, k = CFG["tasks"][task], CFG["seed"], len(LABELS[task])
    val, train = _raw(task, "validation"), _raw(task, "train")
    ev = _sample(val, {c: t["per_class"] for c in range(k)}, f"{seed}-{task}-eval")
    ev_norm = {r["norm"] for v in ev.values() for r in v}
    ev_guid = {r["guid"] for v in ev.values() for r in v}
    train = [r for r in train if r["guid"] not in ev_guid]
    quota = {c: t["pilot_n"] // k + (1 if c < t["pilot_n"] % k else 0) for c in range(k)}
    pilot = _sample(train, quota, f"{seed}-{task}-pilot", banned=ev_norm)
    used = ev_norm | {r["norm"] for v in pilot.values() for r in v}
    shots = _sample(train, {c: 1 for c in range(k)}, f"{seed}-{task}-shot", banned=used)
    write_jsonl(ROOT / "data" / f"{task}_main.jsonl", _finish(task, ev, gen_per_class=t["gen_per_class"]))
    write_jsonl(ROOT / "data" / f"{task}_pilot.jsonl", _finish(task, pilot, gen_total=t["pilot_gen_n"]))
    write_jsonl(ROOT / "data" / f"{task}_fewshot.jsonl", _finish(task, shots, gen_total=0))


def load_items(task, phase):
    path = ROOT / "data" / f"{task}_{phase}.jsonl"
    if not path.exists():
        build(task)
    return read_jsonl(path)


if __name__ == "__main__":
    for task in CFG["tasks"]:
        build(task)
        for ph in ("main", "pilot", "fewshot"):
            items = load_items(task, ph)
            print(task, ph, len(items), "gen subset:", sum(i["gen"] for i in items))
