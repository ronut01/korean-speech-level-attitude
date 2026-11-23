"""LaTeX tables for the manuscript, rendered from the analysis summary (standard library only).

    python -m kspeech.tables                    # results/summary_main.json -> results/tables/*.tex
    python -m kspeech.tables --out DIR          # same, for another results folder

kspeech.analyze --phase main calls write_tables() itself; this entry point only re-renders the saved numbers.
"""
import argparse, json
from pathlib import Path


def pct(x, signed=False):
    return f"${100 * x:+.1f}$" if signed else f"{100 * x:.1f}"      # signed values in math mode (true minus sign)


NAME = {"ynat": "YNAT", "nli": "NLI"}
END = (("Y", "\\textit{Haeyo}"), ("B", "\\textit{Hae}"))


def _ci(d):
    return f"{pct(d['est'], 1)} [{pct(d['lo'], 1)}, {pct(d['hi'], 1)}]"


def _group(t, ncol):
    return f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{NAME[t]}}}}} \\\\"


def write_tables(res, out):
    """LaTeX tabulars for the manuscript. All cells are percentages; D and its CI are in percentage points."""
    tdir = Path(out) / "tables"; tdir.mkdir(exist_ok=True)

    def save(name, spec, head, body):
        L = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", *head, "\\midrule", *body, "\\bottomrule", "\\end{tabular}"]
        (tdir / f"{name}.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    two_level = ("\\multicolumn{2}{c}{\\textit{Haeyo}} & \\multicolumn{2}{c}{\\textit{Hae}}",
                 "Affil. & Rude & Affil. & Rude")
    body = []
    for t, r in res.items():
        a, d = r["acc"], r["D"]
        body.append(f"{NAME[t]} & {pct(r['acc_neutral']['Y'])} & {pct(r['acc_neutral']['B'])} & {pct(a['YP'])} & "
                    f"{pct(a['YR'])} & {pct(a['BP'])} & {pct(a['BR'])} & {_ci(d)} & {d['p_holm']:.3f} \\\\")
    save("main", "lrrrrrrcr",
         [f" & \\multicolumn{{2}}{{c}}{{Neutral}} & {two_level[0]} & & \\\\",
          "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
          f"Task & \\textit{{Haeyo}} & \\textit{{Hae}} & {two_level[1]} & $D$ [95\\% CI] & $p_{{\\mathrm{{Holm}}}}$ \\\\"], body)

    body = []
    for t, r in res.items():
        body.append(_group(t, 6))
        for q in sorted(r["D_by_quartet"]):
            a = r["acc_by_quartet"][q]
            body.append(f"{q} & {pct(a['YP'])} & {pct(a['YR'])} & {pct(a['BP'])} & {pct(a['BR'])} & "
                        f"{_ci(r['D_by_quartet'][q])} \\\\")
    save("quartets", "lrrrrc",
         [f" & {two_level[0]} & \\\\", "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
          f"Quartet & {two_level[1]} & $D$ [95\\% CI] \\\\"], body)

    body = []
    for t, r in res.items():
        body.append(_group(t, 5))
        for h, lab in END:
            x = r["transitions"][h]
            body.append(f"{lab} & {pct(x['correct_to_incorrect'])} & {pct(x['incorrect_to_correct'])} & "
                        f"{pct(x['incorrect_to_other_incorrect'])} & {pct(x['any_prediction_change'])} \\\\")
    save("transitions", "lrrrr", ["Ending & C$\\to$I & I$\\to$C & I$\\to$I$'$ & Any change \\\\"], body)

    if all("generation" in r for r in res.values()):
        body = []
        for t, r in res.items():
            body.append(_group(t, 6))
            for h, lab in END:
                for att, alab in (("P", "Affil."), ("R", "Rude")):
                    x = r["generation"]["by_condition"][h + att]
                    body.append(f"{lab} & {alab} & {pct(x['score_acc'])} & {pct(x['extracted_acc'])} & "
                                f"{pct(x['strict_format'])} & {pct(x['unparseable'])} \\\\")
        save("generation", "llrrrr", ["Ending & Attitude & Score & Extracted & Strict & Unpars. \\\\"], body)
    if all("controls" in r for r in res.values()):
        body = []
        for t, r in res.items():
            body.append(_group(t, 5))
            for h, lab in END:
                x = r["controls"][h]["acc"]
                body.append(f"{lab} & {pct(x['neutral'])} & {pct(x['affiliative'])} & {pct(x['rude'])} & "
                            f"{pct(x['nonpersonal_negative'])} \\\\")
        save("controls", "lrrrr", ["Ending & Neutral & Affil. & Rude & Non-pers. neg. \\\\"], body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results"))
    a = ap.parse_args()
    write_tables(json.loads(Path(a.out, "summary_main.json").read_text(encoding="utf-8"))["tasks"], a.out)
    print("wrote", Path(a.out) / "tables")
