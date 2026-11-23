"""Prompt construction. Only the prefix phrase and the request ending differ between conditions."""
from .common import CFG, LABELS, LETTERS

FORMAT_LINE = "출력 형식: 보기의 알파벳 한 글자"


def options(item):
    """[(letter, class_index)] for this item; identical in every condition."""
    k = len(LABELS[item["task"]])
    return [(LETTERS[i], (i + item["rot"]) % k) for i in range(k)]


def _body(item, answer=None):
    t = item["text"]
    head = f"제목: {t['title']}" if item["task"] == "ynat" else f"전제: {t['premise']}\n가설: {t['hypothesis']}"
    opts = "\n".join(f"{l}. {LABELS[item['task']][c]}" for l, c in options(item))
    s = f"{head}\n보기:\n{opts}"
    if answer is not None:
        s += f"\n답: {answer}"
    return s


def conditions(stim, quartet_ids):
    """Main 2x2 cells for the given quartets."""
    return [{"ending": h, "att": a, "q": q} for q in quartet_ids for h in "YB" for a in "PR"]


def prefix(stim, cond, item):
    if cond["att"] == "N":                      # neutral request, no attitude phrase
        return ""
    if cond["att"] == "X":                      # non-interpersonal negative affect
        return stim["controls"][item["cq"]][cond["ending"]]
    q = next(x for x in stim["quartets"] if x["id"] == cond["q"])
    return q[cond["att"]][cond["ending"]]


def user_message(stim, cond, item, fewshot=()):
    pre = prefix(stim, cond, item)
    req = stim["requests"][item["task"]][cond["ending"]]
    parts = [(pre + " " if pre else "") + req, ""]
    for ex in fewshot:                          # identical exemplars in every condition (recovery only)
        gold = next(l for l, c in options(ex) if c == ex["label"])
        parts += [_body(ex, answer=gold), ""]
    parts += [_body(item), FORMAT_LINE]
    return "\n".join(parts)
