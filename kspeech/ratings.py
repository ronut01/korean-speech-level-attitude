"""Human stimulus validation: make rating sheets, then select quartets by the prespecified rules.

    python -m kspeech.ratings make          # ratings/rater{1,2,3}.xlsx  (+ ratings/key.json)
    python -m kspeech.ratings select        # reads the filled sheets -> stimuli/selected.json, ratings/report.json
    python -m kspeech.ratings provisional   # UNVALIDATED placeholder for smoke tests only
"""
import json, random, statistics as st, sys
from .common import CFG, ROOT, load_stimuli

RDIR = ROOT / "ratings"
COLS = ["코드", "문장", "말높임", "무례함(1-5)", "자연스러움(1-5)", "요청 강도(1-5)", "부정 정서(1-5)",
        "비꼼", "위협", "능력 비난·압박", "과제 의미 변화", "문법 어색", "메모"]
GUIDE = [
    "상황: 성인 사용자가 AI 도구에게 작업을 요청하는 말입니다. 각 문장을 따로따로 판단해 주세요.",
    "말높임: 문장 끝 어미만 보고 '해요체'(~요), '해체'(반말), '판단불가' 중 하나를 고릅니다. 내용이 공손한지는 보지 않습니다.",
    "무례함: 1 = 전혀 무례하지 않음, 5 = 매우 무례함 (듣는 상대를 대하는 태도)",
    "자연스러움: 1 = 매우 어색함, 5 = 매우 자연스러움 (한국어 문장으로서)",
    "요청 강도: 1 = 매우 약하게 부탁함, 5 = 매우 강하게 요구함",
    "부정 정서: 1 = 부정적 감정이 전혀 없음, 5 = 매우 강함",
    "비꼼~문법 어색: 해당하면 O를 고르고, 아니면 비워 둡니다.",
    "메모: 이상하거나 애매한 점이 있으면 자유롭게 적어 주세요. 다른 평정자와 상의하지 말아 주세요.",
]


def _sentences(cand):
    req = cand["requests"]["ynat"]
    rows = []
    for q in cand["quartets"]:
        rows += [{"unit": q["id"], "cell": h + a, "text": f"{q[a][h]} {req[h]}"} for h in "YB" for a in "PR"]
    for x in cand["controls"]:
        rows += [{"unit": x["id"], "cell": h + "X", "text": f"{x[h]} {req[h]}"} for h in "YB"]
    return rows


def _order(rows, rng):
    for _ in range(10000):                       # no two consecutive sentences from the same unit
        rng.shuffle(rows)
        if all(rows[i]["unit"] != rows[i + 1]["unit"] for i in range(len(rows) - 1)):
            return rows
    raise RuntimeError("could not find a valid order")


def make():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    cand = load_stimuli(ROOT / "stimuli" / "candidates.json")
    rows = _sentences(cand)
    rng = random.Random(CFG["seed"])
    codes = rng.sample(range(100, 1000), len(rows))
    key = {f"S{c}": {"unit": r["unit"], "cell": r["cell"], "text": r["text"]} for c, r in zip(codes, rows)}
    RDIR.mkdir(exist_ok=True)
    (RDIR / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    for rater in (1, 2, 3):
        order = _order([{"code": c, **v} for c, v in key.items()], random.Random(f"{CFG['seed']}-{rater}"))
        wb = Workbook(); g = wb.active; g.title = "안내"
        for i, line in enumerate(GUIDE, 1):
            g.cell(i, 1, line)
        g.column_dimensions["A"].width = 120
        ws = wb.create_sheet("평정"); ws.append(COLS)
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in order:
            ws.append([r["code"], r["text"]])
        n = len(order) + 1
        for col, formula in [("C", '"해요체,해체,판단불가"')] + [(c, '"1,2,3,4,5"') for c in "DEFG"] + \
                            [(c, '"O"') for c in "HIJKL"]:
            dv = DataValidation(type="list", formula1=formula, allow_blank=True)
            ws.add_data_validation(dv); dv.add(f"{col}2:{col}{n}")
        ws.column_dimensions["B"].width = 62; ws.column_dimensions["M"].width = 40
        for col in "CDEFGHIJKL":
            ws.column_dimensions[col].width = 15
        for row in ws.iter_rows(min_row=2, max_col=2):
            row[1].alignment = Alignment(wrap_text=True)
        ws.freeze_panes = "C2"
        wb.save(RDIR / f"rater{rater}.xlsx")
    print(f"wrote 3 sheets with {len(rows)} sentences each to {RDIR}")


def _read():
    from openpyxl import load_workbook
    key = json.loads((RDIR / "key.json").read_text(encoding="utf-8"))
    data = {c: [] for c in key}
    for rater in (1, 2, 3):
        ws = load_workbook(RDIR / f"rater{rater}.xlsx")["평정"]
        for r in ws.iter_rows(min_row=2, values_only=True):
            if not r[0]:
                continue
            vals = list(r) + [None] * (len(COLS) - len(r))
            if vals[2] is None or any(vals[i] is None for i in (3, 4, 5, 6)):
                raise SystemExit(f"rater{rater}: {vals[0]} is incomplete")
            data[vals[0]].append({"level": str(vals[2]).strip(), "rude": float(vals[3]), "natural": float(vals[4]),
                                  "intensity": float(vals[5]), "negaff": float(vals[6]),
                                  "flags": [COLS[i] for i in range(7, 12) if vals[i]], "note": vals[12]})
    assert all(len(v) == 3 for v in data.values()), "every sentence needs three ratings"
    return key, data


def select():
    key, data = _read()
    cand = load_stimuli(ROOT / "stimuli" / "candidates.json")
    want = {"Y": "해요체", "B": "해체"}
    cell = {}
    for code, k in key.items():
        rs = data[code]
        cell[(k["unit"], k["cell"])] = {
            "level_ok": sum(r["level"] == want[k["cell"][0]] for r in rs),
            **{m: st.mean(r[m] for r in rs) for m in ("rude", "natural", "intensity", "negaff")},
            "flags": {f: sum(f in r["flags"] for r in rs) for f in COLS[7:12]},
            "notes": [r["note"] for r in rs if r["note"]]}
    report = {"quartets": {}, "controls": {}}
    for q in cand["quartets"]:
        c = {x: cell[(q["id"], x)] for x in ("YP", "YR", "BP", "BR")}
        r = {"type": q.get("type"), "level_agree_of_12": sum(v["level_ok"] for v in c.values()),
             "min_natural": min(v["natural"] for v in c.values()),
             "contrast_Y": c["YR"]["rude"] - c["YP"]["rude"], "contrast_B": c["BR"]["rude"] - c["BP"]["rude"],
             "excluded_flag": any(v["flags"][f] >= 2 for v in c.values() for f in ("위협", "과제 의미 변화")),
             "competence_flag_max": max(v["flags"]["능력 비난·압박"] for v in c.values()),   # reported, not excluded
             "cells": c}
        r["min_contrast"] = min(r["contrast_Y"], r["contrast_B"])
        r["pass"] = (r["level_agree_of_12"] >= 11 and r["min_natural"] >= 4 and r["min_contrast"] >= 1.5
                     and not r["excluded_flag"])
        report["quartets"][q["id"]] = r
    for x in cand["controls"]:   # rule fixed before any model output: natural, not rude, clearly negative
        c = {h: cell[(x["id"], h + "X")] for h in "YB"}
        r = {"level_agree_of_6": sum(v["level_ok"] for v in c.values()),
             "min_natural": min(v["natural"] for v in c.values()),
             "max_rude": max(v["rude"] for v in c.values()), "min_negaff": min(v["negaff"] for v in c.values()),
             "cells": c}
        r["pass"] = r["level_agree_of_6"] >= 5 and r["min_natural"] >= 4 and r["max_rude"] <= 2 and r["min_negaff"] >= 3
        report["controls"][x["id"]] = r
    report["agreement"] = _agreement(key, data, want)
    rq = report["quartets"]
    okq = sorted([i for i in rq if rq[i]["pass"]], key=lambda i: (-rq[i]["min_natural"], -rq[i]["min_contrast"], i))
    rc = report["controls"]
    okc = sorted([i for i in rc if rc[i]["pass"]], key=lambda i: (-rc[i]["min_natural"], -rc[i]["min_negaff"], i))
    selq, filled = _pick(okq, {i: rq[i]["type"] for i in rq})
    report["selected_quartets"], report["selected_controls"] = selq, okc[:4]
    report["type_cap_filled"] = filled        # non-empty = the 2-per-type cap could not be met
    (RDIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("passing quartets:", okq, "| passing controls:", okc)
    if len(okq) < 4 or len(okc) < 4:
        raise SystemExit("Fewer than four passed. Revise the failed candidates once and re-rate (proposal App. A).")
    print("selected quartets:", selq, "types:", [rq[i]["type"] for i in selq], "| cap fallback:", filled)
    _write_selected(cand, selq, okc[:4], provisional=False)


def _agreement(key, data, want):
    """Descriptive inter-rater agreement over all rated sentences (three raters)."""
    codes = sorted(key)
    out = {"n_sentences": len(codes),
           "speech_level_correct_of_all_judgments": [sum(r["level"] == want[key[c]["cell"][0]] for c in codes for r in data[c]),
                                                     3 * len(codes)]}
    for m in ("rude", "natural", "intensity", "negaff"):
        v = [[data[c][k][m] for c in codes] for k in range(3)]
        rs = [st.correlation(v[i], v[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        spread = [max(x) - min(x) for x in zip(*v)]
        out[m] = {"mean_pairwise_pearson": st.mean(rs), "min_pairwise_pearson": min(rs),
                  "share_all_within_one_point": sum(d <= 1 for d in spread) / len(spread),
                  "share_identical": sum(d == 0 for d in spread) / len(spread)}
    return out


def _pick(ranked, type_of, n=4, cap=2):
    """Take quartets in ranked order, at most `cap` per rude-phrase type. If that leaves fewer than n,
    fill with the skipped ones in the same order. Rule fixed on 2026-11-21, before any rating was collected."""
    chosen, skipped, count = [], [], {}
    for i in ranked:
        t = type_of.get(i)
        if len(chosen) < n and (t is None or count.get(t, 0) < cap):
            chosen.append(i); count[t] = count.get(t, 0) + 1
        else:
            skipped.append(i)
    filled = skipped[: n - len(chosen)]
    return chosen + filled, filled


def _write_selected(cand, qids, cids, provisional):
    sel = {"provisional": provisional, "requests": cand["requests"],
           "quartets": [q for i in qids for q in cand["quartets"] if q["id"] == i],
           "controls": [x for i in cids for x in cand["controls"] if x["id"] == i]}
    (ROOT / "stimuli" / "selected.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote stimuli/selected.json", "(PROVISIONAL)" if provisional else "")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "make":
        make()
    elif cmd == "select":
        select()
    elif cmd == "provisional":
        cand = load_stimuli(ROOT / "stimuli" / "candidates.json")
        _write_selected(cand, [q["id"] for q in cand["quartets"][:4]], [x["id"] for x in cand["controls"][:4]], True)
    else:
        print(__doc__)
