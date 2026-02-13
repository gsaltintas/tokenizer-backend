from app.services.adapter import TokenizerAdapter

# Common English morphemes
PREFIXES = [
    "anti", "auto", "bi", "co", "counter", "de", "dis", "down", "ex", "extra",
    "fore", "hyper", "il", "im", "in", "inter", "ir", "mal", "micro", "mid",
    "mis", "mono", "multi", "non", "out", "over", "poly", "post", "pre", "pro",
    "re", "semi", "sub", "super", "tele", "trans", "tri", "ultra", "un", "under",
    "up",
]

SUFFIXES = [
    "able", "ible", "al", "ial", "an", "ian", "ance", "ence", "ant", "ent",
    "ary", "ery", "ory", "ate", "dom", "ed", "en", "er", "or", "ar",
    "est", "ful", "hood", "ic", "ical", "ile", "ing", "ion", "tion", "ation",
    "sion", "ious", "ous", "eous", "ish", "ism", "ist", "ity", "ive", "ative",
    "less", "like", "ly", "ment", "ness", "ous", "ship", "ty",
    "ward", "wards", "wise", "ize", "ise", "ify", "fy", "age", "ance", "ence",
    "ling", "let",
]

ROOTS = [
    "act", "aud", "bio", "cap", "ced", "cent", "chron", "cide", "claim",
    "clar", "cogn", "cord", "corp", "cosm", "crat", "cred", "crypt",
    "dict", "doc", "dom", "duc", "fact", "fer", "fid", "fig", "fin",
    "flex", "flu", "form", "fract", "gen", "geo", "grad", "graph", "grat",
    "grav", "hab", "hom", "hydr", "init", "ject", "jud", "junct",
    "jur", "lect", "leg", "liber", "liter", "loc", "log", "luc",
    "magn", "man", "mand", "mater", "med", "mem", "ment", "merc",
    "meter", "migr", "min", "mir", "miss", "mit", "mob", "mon", "morph",
    "mort", "mot", "multi", "nat", "neg", "nom", "norm", "not",
    "nov", "numer", "oper", "opt", "ord", "pac", "pass", "path",
    "ped", "pel", "pend", "phil", "phon", "photo", "plic", "pod",
    "pol", "pop", "port", "pos", "pot", "prim", "prob", "prot",
    "psych", "pub", "punct", "quer", "quest", "rect", "reg",
    "rupt", "sacr", "sanct", "sci", "scrib", "script", "sect", "sen",
    "sens", "sent", "sequ", "serv", "sign", "simil", "sol",
    "spec", "spir", "stab", "stat", "struct", "sum", "tact",
    "temp", "ten", "term", "terr", "therm", "tort", "tract",
    "trans", "trib", "turb", "typ", "ultima", "umbr", "uni",
    "vac", "val", "ven", "ver", "verb", "vest", "vid", "vis",
    "vit", "viv", "voc", "vol",
]

ALL_MORPHEMES = set(PREFIXES + SUFFIXES + ROOTS)


def _decompose_morphemes(word: str) -> tuple[str, list[str]]:
    """
    Attempt to decompose a word into known morphemes.
    Returns (type, morpheme_list).
    """
    if not word or not word.isalpha():
        return ("arbitrary", [])

    word_lower = word.lower()

    # Exact match
    if word_lower in ALL_MORPHEMES:
        return ("morpheme", [word_lower])

    # Try to decompose via longest-match greedy
    morphemes = []
    remaining = word_lower
    unmatched = 0

    while remaining:
        best_match = ""
        for length in range(min(len(remaining), 10), 0, -1):
            candidate = remaining[:length]
            if candidate in ALL_MORPHEMES:
                best_match = candidate
                break

        if best_match:
            morphemes.append(best_match)
            remaining = remaining[len(best_match):]
        else:
            # No morpheme match, consume one character
            unmatched += 1
            remaining = remaining[1:]

    if unmatched == 0 and len(morphemes) > 1:
        return ("morpheme_composite", morphemes)
    elif morphemes and unmatched < len(word_lower) / 2:
        return ("subword", morphemes)
    else:
        return ("arbitrary", [])


def compute_morpheme_analysis(adapter: TokenizerAdapter) -> list[dict]:
    """Analyze morphological structure of all vocabulary tokens."""
    vocab = adapter.get_vocab()
    results = []

    for token_str, token_id in vocab.items():
        # Strip space prefix for analysis
        clean = token_str.lstrip(" \u0120\u00a0▁")
        morpheme_type, morphemes = _decompose_morphemes(clean)

        results.append(
            {
                "token_str": token_str,
                "token_id": token_id,
                "morpheme_type": morpheme_type,
                "morphemes": morphemes,
            }
        )

    return results
