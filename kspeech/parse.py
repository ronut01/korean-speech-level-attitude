"""Prespecified parser for generated answers (Appendix B of the proposal). Do not tune on test outputs."""


def parse_generation(text, letters, class_names):
    """letters: valid option letters for the task; class_names: {letter: surface text of that option}."""
    s = text.strip()
    out = {"extracted": None, "strict": False, "multi": False, "unparseable": False}
    if s and s[0] in letters:
        others = {ch for ch in s[1:] if ch in letters and ch != s[0]}
        if others:
            out["multi"] = True
        else:
            out["extracted"] = s[0]
            out["strict"] = (s == s[0])
        return out
    hits = [l for l, name in class_names.items() if name in s]
    if len(hits) == 1:
        out["extracted"] = hits[0]
    elif len(hits) > 1:
        out["multi"] = True
    else:
        out["unparseable"] = True
    return out
