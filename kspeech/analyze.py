"""Statistics and tables from the saved evaluation rows. No model is loaded here.

    python -m kspeech.analyze --phase pilot     # operational gates + runtime
    python -m kspeech.analyze --phase main      # results/summary_main.json + results/tables/*.tex
    python -m kspeech.tables                    # re-render the tables from summary_main.json (no recomputation)
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from .tables import write_tables
from .common import CFG, ROOT, load_stimuli, read_jsonl
from .data import load_items

B, MARGIN = CFG["bootstrap_replicates"], CFG["practical_margin"]


def boot(v, strata, seed=CFG["seed"]):
    """Stratified item bootstrap of mean(v): estimate, 95% percentile CI, centered two-sided p."""
    v, strata = np.asarray(v, float), np.asarray(strata)
    rng = np.random.default_rng(seed)
    est, n = v.mean(), len(v)
    tot, tot0 = np.zeros(B), np.zeros(B)
    for c in np.unique(strata):
        x = v[strata == c]
        idx = rng.integers(0, len(x), size=(B, len(x)))
        s = x[idx].sum(1)
        tot += s
        tot0 += s - len(x) * est                      # same draws applied to centered values v - est
    means, null = tot / n, tot0 / n
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = (1 + int((np.abs(null) >= abs(est) - 1e-12).sum())) / (B + 1)
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "p": float(p),
            "within_margin": bool(lo > -MARGIN and hi < MARGIN), "n": int(n)}


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def load(out, phase, part, task, tag):
    df = pd.DataFrame(read_jsonl(Path(out) / f"{phase}_{part}_{task}_{tag}.jsonl"))
    if len(df):
        df["q"] = df["q"].fillna("-")
    return df


def cube(df, guids, qids, col="correct"):
    """Array [item, quartet, ending(Y,B), attitude(P,R)] of `col`; NaN where a run is missing."""
    a = np.full((len(guids), len(qids), 2, 2), np.nan)
    gi, qi = {g: i for i, g in enumerate(guids)}, {q: i for i, q in enumerate(qids)}
    d = df[df["att"].isin(["P", "R"]) & df["guid"].isin(gi) & df["q"].isin(qi)]
    a[d["guid"].map(gi), d["q"].map(qi), (d["ending"] == "B").astype(int), (d["att"] == "R").astype(int)] = \
        d[col].astype(float)
    return a


def interaction(c):          # c: [item, q, h, t] -> per-item D contribution, averaged over quartets
    return ((c[:, :, 0, 1] - c[:, :, 0, 0]) - (c[:, :, 1, 1] - c[:, :, 1, 0])).mean(1)


def single(df, guids, att, ending, col="correct"):
    d = df[(df["att"] == att) & (df["ending"] == ending)].set_index("guid")[col].astype(float)
    return d.reindex(guids).to_numpy()


def analyze_task(out, phase, task, tag, stim):
    items = load_items(task, phase)
    sc, ge = load(out, phase, "score", task, tag), load(out, phase, "gen", task, tag)
    qids = [q["id"] for q in stim["quartets"]][: (2 if phase == "pilot" else 4)]
    over = set(sc.loc[sc["over_limit"], "guid"]) if len(sc) else set()
    full = cube(sc, [i["guid"] for i in items], qids)
    keep = [n for n, i in enumerate(items) if i["guid"] not in over and not np.isnan(full[n]).any()]
    items = [items[n] for n in keep]; c = full[keep]
    guids, lab = [i["guid"] for i in items], np.array([i["label"] for i in items])
    r = {"n_items": len(items), "n_excluded_over_token_limit": len(over), "quartets": qids}

    r["acc"] = {h + t: float(c[:, :, hi, ti].mean()) for hi, h in enumerate("YB") for ti, t in enumerate("PR")}
    r["acc_neutral"] = {h: float(np.nanmean(single(sc, guids, "N", h))) for h in "YB"}
    r["cand_mass"] = {h + t: float(cube(sc, guids, qids, "cand_mass")[:, :, hi, ti].mean())
                      for hi, h in enumerate("YB") for ti, t in enumerate("PR")}
    r["D"] = boot(interaction(c), lab)
    r["attitude_effect"] = {h: boot((c[:, :, hi, 1] - c[:, :, hi, 0]).mean(1), lab) for hi, h in enumerate("YB")}
    r["ending_effect"] = boot((c[:, :, 0, :] - c[:, :, 1, :]).mean((1, 2)), lab)       # haeyo - hae
    r["attitude_main_effect"] = boot((c[:, :, :, 1] - c[:, :, :, 0]).mean((1, 2)), lab)  # rude - affiliative
    r["D_by_quartet"] = {q: boot(interaction(c[:, [k]]), lab) for k, q in enumerate(qids)}
    if len(qids) > 2:
        r["D_leave_one_out"] = {q: boot(interaction(np.delete(c, k, 1)), lab) for k, q in enumerate(qids)}
    r["acc_by_quartet"] = {q: {h + t: float(c[:, k, hi, ti].mean()) for hi, h in enumerate("YB")
                               for ti, t in enumerate("PR")} for k, q in enumerate(qids)}

    pr = cube(sc, guids, qids, "pred"); g = lab[:, None]
    r["transitions"] = {}
    for hi, h in enumerate("YB"):                    # affiliative -> rude, over (item, quartet) pairs
        p0, p1 = pr[:, :, hi, 0], pr[:, :, hi, 1]
        r["transitions"][h] = {"correct_to_incorrect": float(((p0 == g) & (p1 != g)).mean()),
                               "incorrect_to_correct": float(((p0 != g) & (p1 == g)).mean()),
                               "incorrect_to_other_incorrect": float(((p0 != g) & (p1 != g) & (p0 != p1)).mean()),
                               "any_prediction_change": float((p0 != p1).mean())}
    for ti, t in enumerate("PR"):                    # haeyo -> hae with the attitude phrase fixed
        p0, p1 = pr[:, :, 0, ti], pr[:, :, 1, ti]
        r["transitions"]["ending_" + t] = {"any_prediction_change": float((p0 != p1).mean()),
                                           "net_accuracy_change": float(((p1 == g).mean() - (p0 == g).mean()))}

    sub = [n for n, i in enumerate(items) if i["gen"]]
    if len(ge) and sub:
        sg, sl = [guids[n] for n in sub], lab[sub]
        gc = cube(ge, sg, qids)
        ok = ~np.isnan(gc).any((1, 2, 3)); sg = [x for x, o in zip(sg, ok) if o]; sl = sl[ok]; gc = gc[ok]
        gs, cs = cube(ge, sg, qids, "strict"), c[np.array(sub)[ok]]
        gen = {"n_items": len(sg), "by_condition": {}}
        for hi, h in enumerate("YB"):
            for ti, t in enumerate("PR"):
                d = ge[(ge["ending"] == h) & (ge["att"] == t) & ge["guid"].isin(sg)]
                gen["by_condition"][h + t] = {
                    "score_acc": float(cs[:, :, hi, ti].mean()), "extracted_acc": float(gc[:, :, hi, ti].mean()),
                    "strict_format": float(gs[:, :, hi, ti].mean()), "unparseable": float(d["unparseable"].mean()),
                    "multi_answer": float(d["multi"].mean()), "hit_cap": float(d["hit_cap"].mean())}
        gen["extraction_rate"] = float(ge["extracted"].notna().mean())
        gen["D_score_subset"] = boot(interaction(cs), sl)
        gen["D_extracted"] = boot(interaction(gc), sl)
        gen["D_strict_format"] = boot(interaction(gs), sl)
        r["generation"] = gen

    if phase == "main" and sub:                      # neutral / attitude / non-interpersonal negative, same items
        sg, sl = [guids[n] for n in sub], lab[sub]
        slot = np.array([items[n]["cq"] for n in sub]); rows = np.arange(len(sub))
        ctl = {}
        for hi, h in enumerate("YB"):
            neu, neg = single(sc, sg, "N", h), single(sc, sg, "X", h)
            aff, rude = c[sub][rows, slot, hi, 0], c[sub][rows, slot, hi, 1]
            m = ~np.isnan(neg)
            ctl[h] = {"n": int(m.sum()), "acc": {"neutral": float(neu[m].mean()), "affiliative": float(aff[m].mean()),
                                                 "rude": float(rude[m].mean()), "nonpersonal_negative": float(neg[m].mean())},
                      "affiliative_minus_neutral": boot((aff - neu)[m], sl[m]),
                      "rude_minus_neutral": boot((rude - neu)[m], sl[m]),
                      "negative_minus_neutral": boot((neg - neu)[m], sl[m]),
                      "rude_minus_negative": boot((rude - neg)[m], sl[m])}
        r["controls"] = ctl
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "main"], required=True)
    ap.add_argument("--model", default=CFG["model"])
    ap.add_argument("--stimuli", default=None)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    tag, stim = a.model.split("/")[-1], load_stimuli(a.stimuli)
    res = {t: analyze_task(a.out, a.phase, t, tag, stim) for t in CFG["tasks"]}
    for t, ph in zip(res, holm([res[t]["D"]["p"] for t in res])):
        res[t]["D"]["p_holm"] = ph
    meta = read_jsonl(Path(a.out) / f"meta_{a.phase}_{tag}.jsonl")
    summary = {"phase": a.phase, "model": a.model, "stimuli_provisional": bool(stim.get("provisional")),
               "bootstrap_replicates": B, "practical_margin": MARGIN, "tasks": res, "run_meta": meta}
    if a.phase == "pilot":
        gates = {}
        for t, r in res.items():
            neu = float(np.mean(list(r["acc_neutral"].values())))
            ext = r.get("generation", {}).get("extraction_rate")
            gates[t] = {"neutral_acc": neu, "neutral_gate": CFG["tasks"][t]["neutral_gate"],
                        "neutral_pass": neu >= CFG["tasks"][t]["neutral_gate"], "extraction_rate": ext,
                        "extraction_pass": ext is not None and ext >= CFG["gen_extraction_gate"]}
        summary["gates"] = gates
        print(json.dumps(gates, indent=1))
    else:
        write_tables(res, a.out)
    Path(a.out, f"summary_{a.phase}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for t, r in res.items():
        d = r["D"]
        print(f"{t}: n={r['n_items']} acc={ {k: round(v, 3) for k, v in r['acc'].items()} } "
              f"neutral={ {k: round(v, 3) for k, v in r['acc_neutral'].items()} } "
              f"D={d['est']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] p={d['p']:.4f} p_holm={d['p_holm']:.4f}")


if __name__ == "__main__":
    main()
