import unicodedata

from app.services.adapter import TokenizerAdapter

# Map Unicode script names to readable categories
SCRIPT_ALIASES = {
    "LATIN": "Latin",
    "CJK": "CJK",
    "HANGUL": "Korean",
    "HIRAGANA": "Japanese (Hiragana)",
    "KATAKANA": "Japanese (Katakana)",
    "ARABIC": "Arabic",
    "DEVANAGARI": "Devanagari",
    "CYRILLIC": "Cyrillic",
    "GREEK": "Greek",
    "HEBREW": "Hebrew",
    "THAI": "Thai",
    "GEORGIAN": "Georgian",
    "ARMENIAN": "Armenian",
    "BENGALI": "Bengali",
    "TAMIL": "Tamil",
    "TELUGU": "Telugu",
    "KANNADA": "Kannada",
    "MALAYALAM": "Malayalam",
    "GUJARATI": "Gujarati",
    "GURMUKHI": "Gurmukhi",
    "ETHIOPIC": "Ethiopic",
    "TIBETAN": "Tibetan",
    "MYANMAR": "Myanmar",
    "KHMER": "Khmer",
    "LAO": "Lao",
    "SINHALA": "Sinhala",
}


def _char_script(ch: str) -> str:
    """Get the script category of a character."""
    cat = unicodedata.category(ch)
    if cat.startswith("L") or cat.startswith("M"):
        try:
            name = unicodedata.name(ch, "UNKNOWN")
            # Extract script from Unicode name
            first_word = name.split(" ")[0]
            return SCRIPT_ALIASES.get(first_word, first_word.title())
        except ValueError:
            return "Unknown"
    elif cat.startswith("N"):
        return "Digit"
    elif cat.startswith("P"):
        return "Punctuation"
    elif cat.startswith("S"):
        return "Symbol"
    elif cat.startswith("Z"):
        return "Whitespace"
    elif cat.startswith("C"):
        return "Control"
    else:
        return "Other"


def compute_language_composition(adapter: TokenizerAdapter) -> dict:
    """Compute script/language composition of the vocabulary."""
    vocab = adapter.get_vocab()
    script_counts: dict[str, int] = {}
    script_examples: dict[str, list[str]] = {}
    mixed_count = 0
    total = 0

    for token_str in vocab:
        total += 1
        # Classify each character
        char_scripts = set()
        for ch in token_str:
            s = _char_script(ch)
            char_scripts.add(s)

        # Ignore whitespace/control-only tokens for script classification
        meaningful_scripts = char_scripts - {"Whitespace", "Control", "Other"}

        if len(meaningful_scripts) == 0:
            # All whitespace/control
            if "Whitespace" in char_scripts:
                script = "Whitespace"
            elif "Control" in char_scripts:
                script = "Control"
            else:
                script = "Other"
        elif len(meaningful_scripts) == 1:
            script = meaningful_scripts.pop()
        else:
            script = "Mixed"
            mixed_count += 1

        script_counts[script] = script_counts.get(script, 0) + 1
        if script not in script_examples:
            script_examples[script] = []
        if len(script_examples[script]) < 10:
            script_examples[script].append(token_str)

    categories = []
    for script, count in sorted(script_counts.items(), key=lambda x: -x[1]):
        categories.append(
            {
                "script": script,
                "token_count": count,
                "percentage": (count / max(total, 1)) * 100,
                "example_tokens": script_examples.get(script, []),
            }
        )

    return {
        "categories": categories,
        "total_tokens": total,
        "mixed_script_count": mixed_count,
    }
