"""Supplementary (post hoc, exploratory) analyses from the saved evaluation rows. Standard library only.

    python -m kspeech.extra                     # results/*.jsonl -> results/extra_main.json
    python -m kspeech.extra --out DIR           # same, for another results folder

These analyses were not prespecified in the proposal. They cover: contrasts against the neutral request on all
items, the comparison a single politeness scale would make, per-quartet attitude effects, predicted-class shares and per-class recall by condition, same-item comparisons
of the non-interpersonal negative control (X) with the neutral, affiliative and rude conditions, strict-format
compliance, disagreements between scoring and generation, and one seeded random example per transition category.

The bootstrap follows kspeech.analyze.boot (class-stratified item resampling, percentile 95% CI, centered two-sided
p) but uses Python's `random` instead of NumPy, so interval end points can differ from NumPy-based runs in the last
digit. Point estimates are exact.
"""
import argparse, collections, json, random
from pathlib import Path
from .common import CFG, LABELS, ROOT, load_stimuli, read_jsonl

B, MARGIN = CFG["bootstrap_replicates"], CFG["practical_margin"]


def _pct(sorted_vals, q):                       # linear-interpolation percentile, as numpy.percentile
    k = (len(sorted_vals) - 1) * q / 100
    f = int(k); c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def boot(v, strata, seed=CFG["seed"]):
    """Stratified item bootstrap of mean(v): estimate, 95% percentile CI, centered two-sided p."""
    groups = collections.defaultdict(list)
    for x, s in zip(v, strata):
        groups[s].append(float(x))
    groups = [groups[s] for s in sorted(groups)]
    n = len(v); est = sum(v) / n
    rng = random.Random(seed)
    means = sorted(sum(sum(rng.choices(g, k=len(g))) for g in groups) / n for _ in range(B))
    p = (1 + sum(abs(m - est) >= abs(est) - 1e-12 for m in means)) / (B + 1)
    lo, hi = _pct(means, 2.5), _pct(means, 97.5)
    return {"est": est, "lo": lo, "hi": hi, "p": p, "within_margin": bool(lo > -MARGIN and hi < MARGIN), "n": n}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def analyze(out, task, tag, qids):
    items = read_jsonl(ROOT / "data" / f"{task}_main.jsonl")
    sc = read_jsonl(Path(out) / f"main_score_{task}_{tag}.jsonl")
    ge = read_jsonl(Path(out) / f"main_gen_{task}_{tag}.jsonl")
    names, K = LABELS[task], len(LABELS[task])
    S = {(r["guid"], r["ending"], r["att"], r["q"]): r for r in sc}
    G = {(r["guid"], r["ending"], r["att"], r["q"]): r for r in ge}
    guids = [i["guid"] for i in items]; lab = {i["guid"]: i["label"] for i in items}
    strata = [lab[g] for g in guids]
    sub = [i["guid"] for i in items if i["gen"]]; sub_strata = [lab[g] for g in sub]
    H = "YB"

    def cell(g, h, att, q=None, f=lambda r: r["correct"]):
        """Item-level value: one row for N/X, mean over the given (or all) quartets for P/R."""
        if att in "NX":
            return float(f(S[g, h, att, None]))
        return mean(float(f(S[g, h, att, x])) for x in ([q] if q else qids))

    def item(g, att, hs=H, q=None, f=lambda r: r["correct"]):
        return mean(cell(g, h, att, q, f) for h in hs)

    def contrast(a, b, pop=guids, st=strata, hs=H, q=None, f=lambda r: r["correct"]):
        return boot([item(g, a, hs, q, f) - item(g, b, hs, q, f) for g in pop], st)

    r = {"n_items": len(guids), "n_control_subset": len(sub), "quartets": qids, "bootstrap_replicates": B}

    # 1. contrasts against the neutral request, all items
    r["vs_neutral"] = {f"{a}_minus_N": {**{h: contrast(a, "N", hs=h) for h in H}, "pooled": contrast(a, "N")}
                       for a in "PR"}
    r["neutral_Y_minus_B"] = boot([cell(g, "Y", "N") - cell(g, "B", "N") for g in guids], strata)

    # 1b. what a single politeness scale would compare: one "polite" cell against one "rude" cell
    two = lambda h1, a1, h0, a0: boot([cell(g, h1, a1) - cell(g, h0, a0) for g in guids], strata)
    r["single_scale"] = {"BR_minus_YP": two("B", "R", "Y", "P"),      # hae+rude vs haeyo+affiliative
                         "YR_minus_BP": two("Y", "R", "B", "P"),      # haeyo+rude vs hae+affiliative
                         "ending_within_P_B_minus_Y": two("B", "P", "Y", "P"),
                         "attitude_within_B_R_minus_P": two("B", "R", "B", "P")}

    # 2. attitude effect (rude - affiliative) per quartet
    r["attitude_by_quartet"] = {q: {**{h: contrast("R", "P", hs=h, q=q) for h in H}, "pooled": contrast("R", "P", q=q)}
                                for q in sorted(qids)}

    # 3. predicted-class shares (%) and per-class recall (%) by condition
    def share(pop, h, att, q=None):
        return {names[c]: 100 * mean(cell(g, h, att, q, lambda x: x["pred"] == c) for g in pop) for c in range(K)}

    def recall(pop, h, att, q=None):
        return {names[c]: 100 * mean(cell(g, h, att, q) for g in pop if lab[g] == c) for c in range(K)}

    r["pred_share_all_items"] = {h + a: share(guids, h, a) for h in H for a in "NPR"}
    r["pred_share_control_subset"] = {h + a: share(sub, h, a) for h in H for a in "NPRX"}
    r["pred_share_control_subset"].update(
        {"pooled_" + a: {n: mean(r["pred_share_control_subset"][h + a][n] for h in H) for n in names} for a in "NPRX"})
    r["recall_all_items"] = {h + a: recall(guids, h, a) for h in H for a in "NPR"}
    r["recall_control_subset"] = {h + a: recall(sub, h, a) for h in H for a in "NPRX"}
    r["pred_share_by_quartet"] = {q: {h + a: share(guids, h, a, q) for h in H for a in "PR"} for q in sorted(qids)}
    r["recall_by_quartet"] = {q: {a: {n: mean(recall(guids, h, a, q)[n] for h in H) for n in names} for a in "PR"}
                              for q in sorted(qids)}

    # 4. shifts in predicted-class share: rude - affiliative on all items; X against N/P/R on the same subset
    is_c = lambda c: (lambda x: x["pred"] == c)
    r["share_shift_R_minus_P_all_items"] = {names[c]: contrast("R", "P", f=is_c(c)) for c in range(K)}
    if task == "nli":
        r["share_shift_control_subset"] = {
            names[c]: {f"{a}_minus_{b}": contrast(a, b, sub, sub_strata, f=is_c(c))
                       for a, b in (("X", "N"), ("X", "P"), ("X", "R"), ("R", "N"), ("R", "P"))} for c in range(K)}
    r["accuracy_control_subset_quartet_mean"] = {
        "acc": {h + a: 100 * mean(cell(g, h, a) for g in sub) for h in H for a in "NPRX"},
        **{f"{a}_minus_{b}": contrast(a, b, sub, sub_strata) for a, b in (("X", "N"), ("X", "P"), ("X", "R"))}}

    # 5. generation audit: strict-format compliance and scoring/generation disagreements
    gsub = sorted({k[0] for k in G}, key=guids.index)
    gst = [lab[g] for g in gsub]
    gitem = lambda g, att, col: mean(float(G[g, h, att, q][col]) for h in H for q in qids)
    r["strict_format_R_minus_P"] = boot([gitem(g, "R", "strict") - gitem(g, "P", "strict") for g in gsub], gst)
    non = [v for v in G.values() if not v["strict"]]
    r["non_strict_outputs"] = {"n": len(non), "by_attitude": dict(collections.Counter(v["att"] for v in non)),
                               "by_raw_text": dict(collections.Counter(v["raw"] for v in non).most_common())}
    by_letter = {}                                   # does non-strict output depend on which option letter is chosen?
    for v in G.values():
        d = by_letter.setdefault(v["extracted"], {"n": 0, "non_strict": 0, "n_P": 0, "non_strict_P": 0, "n_R": 0,
                                                  "non_strict_R": 0, "by_class_at_letter": {}})
        d["n"] += 1; d["non_strict"] += not v["strict"]
        d["n_" + v["att"]] += 1; d["non_strict_" + v["att"]] += not v["strict"]
        c = d["by_class_at_letter"].setdefault(names[v["pred"]], {"n": 0, "non_strict": 0})
        c["n"] += 1; c["non_strict"] += not v["strict"]
    r["non_strict_outputs"]["by_extracted_letter"] = {k: by_letter[k] for k in sorted(by_letter, key=str)}
    mism = []
    for k, v in sorted(G.items(), key=lambda kv: kv[1]["key"]):
        s = S[k]
        if v["pred"] != s["pred"]:
            top = sorted(s["probs"], reverse=True)
            mism.append({"key": v["key"], "generated": v["raw"], "scored_letter": s["pred_letter"],
                         "top1_prob": top[0], "top2_prob": top[1], "gap": round(top[0] - top[1], 6)})
    r["score_vs_generation"] = {"n_compared": len(G), "n_disagree": len(mism),
                                "n_exact_tie": sum(m["gap"] == 0 for m in mism), "disagreements": mism}

    # 6. one seeded random example per category (no manual selection): unit = (item, quartet, ending)
    text = {i["guid"]: i["text"] for i in items}
    units = sorted((g, q, h) for g in guids for q in qids for h in H)
    pred = lambda g, q, h, a: S[g, h, a, q if a in "PR" else None]["pred"]
    cats = {"correct_to_incorrect": lambda g, p0, p1: p0 == lab[g] and p1 != lab[g],
            "incorrect_to_correct": lambda g, p0, p1: p0 != lab[g] and p1 == lab[g],
            "incorrect_to_other_incorrect": lambda g, p0, p1: p0 != lab[g] and p1 != lab[g] and p0 != p1}
    ex = {}
    for name, cond in cats.items():
        pool = [(g, q, h) for g, q, h in units if cond(g, pred(g, q, h, "P"), pred(g, q, h, "R"))]
        g, q, h = random.Random(f"{CFG['seed']}-{task}-{name}").choice(pool)
        ex[name] = {"pool_size": len(pool), "guid": g, "quartet": q, "ending": h, "text": text[g],
                    "gold": names[lab[g]], "pred_neutral": names[pred(g, q, h, "N")],
                    "pred_affiliative": names[pred(g, q, h, "P")], "pred_rude": names[pred(g, q, h, "R")]}
    pool = sorted(non, key=lambda v: v["key"])
    if pool:
        v = random.Random(f"{CFG['seed']}-{task}-non_strict").choice(pool)
        ex["non_strict_generation"] = {"pool_size": len(pool), "guid": v["guid"], "quartet": v["q"],
                                       "ending": v["ending"], "attitude": v["att"], "text": text[v["guid"]],
                                       "gold": names[lab[v["guid"]]], "generated": v["raw"],
                                       "extracted": v["extracted"], "pred": names[v["pred"]]}
    r["sampled_examples"] = ex
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CFG["model"])
    ap.add_argument("--stimuli", default=None)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    qids = [q["id"] for q in load_stimuli(a.stimuli)["quartets"]]
    res = {t: analyze(a.out, t, a.model.split("/")[-1], qids) for t in CFG["tasks"]}
    Path(a.out, "extra_main.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for t, r in res.items():
        f = lambda d: f"{100 * d['est']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}] p={d['p']:.4f}"
        print(t, "P-N", f(r["vs_neutral"]["P_minus_N"]["pooled"]), "| R-N", f(r["vs_neutral"]["R_minus_N"]["pooled"]),
              "| strict R-P", f(r["strict_format_R_minus_P"]))
    print("wrote", Path(a.out) / "extra_main.json")
