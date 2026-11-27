"""Kazakh text normalization for audio8_tts.

Spells out numbers, ordinals, decimals, percentages, dates, times, Roman
century numerals and common abbreviations in the style of the KazakhTTS2
transcripts ("екі мың он тоғызыншы жылы", "тоғыз бүтін оннан бес"). It also
repairs Latin look-alike letters inside Cyrillic words. Text that needs no
change passes through unchanged apart from Unicode NFC.
"""

from __future__ import annotations

import re
import unicodedata

ONES = ["", "бір", "екі", "үш", "төрт", "бес", "алты", "жеті", "сегіз", "тоғыз"]
TENS = ["", "он", "жиырма", "отыз", "қырық", "елу", "алпыс", "жетпіс", "сексен", "тоқсан"]
SCALES = [(10**12, "триллион"), (10**9, "миллиард"), (10**6, "миллион"), (1000, "мың")]

BACK_VOWELS = set("аоұыя")
FRONT_VOWELS = set("әеөүіэ")
VOWELS = BACK_VOWELS | FRONT_VOWELS | set("иую")

MONTHS = [
    "қаңтар", "ақпан", "наурыз", "сәуір", "мамыр", "маусым",
    "шілде", "тамыз", "қыркүйек", "қазан", "қараша", "желтоқсан",
]

# Latin letters that look like Cyrillic ones; replaced only inside words that
# also contain Cyrillic letters.
_HOMOGLYPHS = str.maketrans(
    {
        "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
        "i": "і", "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н",
        "O": "О", "P": "Р", "C": "С", "T": "Т", "X": "Х", "I": "І",
        "ə": "ә", "Ə": "Ә",
    }
)

# Abbreviations matched as whole tokens (case-sensitive), longest first.
ABBREVIATIONS = {
    "км/сағ": "шақырым сағатына",
    "б.з.б.": "біздің заманымызға дейін",
    "б.з.д.": "біздің заманымызға дейін",
    "б.д.д.": "біздің дәуірімізге дейін",
    "б.з.": "біздің заманымыз",
    "т.с.с.": "тағы сол сияқты",
    "т.б.": "тағы басқа",
    "ҚР": "Қазақстан Республикасы",
    "БҰҰ": "Біріккен Ұлттар Ұйымы",
    "млрд": "миллиард",
    "млн": "миллион",
    "трлн": "триллион",
    "ж.": "жылы",
    "жж.": "жылдары",
    "ғ.": "ғасыр",
    "км": "шақырым",
    "кг": "килограмм",
    "см": "сантиметр",
    "мм": "миллиметр",
    "га": "гектар",
    "тг": "теңге",
}
FRACTIONS = {"½": "жарым", "¼": "төрттен бір", "¾": "төрттен үш", "⅓": "үштен бір", "⅔": "үштен екі"}
SYMBOLS = {"%": "пайыз", "₸": "теңге", "$": "доллар", "€": "еуро", "№": "нөмір", "°": "градус"}

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _harmony_is_back(word: str) -> bool:
    for char in reversed(word.lower()):
        if char in BACK_VOWELS:
            return True
        if char in FRONT_VOWELS:
            return False
    return False


def _under_thousand(n: int) -> list[str]:
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        if hundreds > 1:
            words.append(ONES[hundreds])
        words.append("жүз")
    tens, ones = divmod(rest, 10)
    if tens:
        words.append(TENS[tens])
    if ones:
        words.append(ONES[ones])
    return words


def cardinal(n: int) -> str:
    """Spell a non-negative integer as Kazakh words."""
    if n < 0:
        return "минус " + cardinal(-n)
    if n == 0:
        return "нөл"
    words: list[str] = []
    for value, name in SCALES:
        count, n = divmod(n, value)
        if count:
            # KazakhTTS2 reads 1991 as "мың тоғыз жүз..." with no leading "бір".
            if count > 1 or value > 1000:
                words.extend(_under_thousand(count))
            words.append(name)
    words.extend(_under_thousand(n))
    return " ".join(words)


def ordinal_word(word: str) -> str:
    """Turn the last number word into its ordinal form."""
    if word == "жиырма":
        return "жиырмасыншы"
    if word == "елу":
        return "елуінші"
    if word == "қырық":
        return "қырқыншы"
    back = _harmony_is_back(word)
    if word[-1] in VOWELS:
        return word + ("ншы" if back else "нші")
    return word + ("ыншы" if back else "інші")


def ordinal(n: int) -> str:
    words = cardinal(n).split()
    words[-1] = ordinal_word(words[-1])
    return " ".join(words)


def _ablative(word: str) -> str:
    """Ablative case of a number word (он → оннан, жүз → жүзден, мың → мыңнан)."""
    back = _harmony_is_back(word)
    if word[-1] in "мнң":
        return word + ("нан" if back else "нен")
    if word[-1] in VOWELS or word[-1] in "лрйузж":
        return word + ("дан" if back else "ден")
    return word + ("тан" if back else "тен")


def decimal(integer: str, fraction: str) -> str:
    denominator = {1: "он", 2: "жүз", 3: "мың"}[len(fraction)]
    numerator = cardinal(int(fraction))
    return f"{cardinal(int(integer))} бүтін {_ablative(denominator)} {numerator}"


def roman_to_int(value: str) -> int | None:
    if not value or not re.fullmatch(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})", value):
        return None
    total = 0
    for index, char in enumerate(value):
        number = ROMAN[char]
        if index + 1 < len(value) and ROMAN[value[index + 1]] > number:
            total -= number
        else:
            total += number
    return total or None


_ORDINAL_SUFFIX_RE = re.compile(r"^(ыншы|інші|ншы|нші|шы|ші)(.*)$")
_YEAR_WORD_RE = re.compile(r"^(жыл|ғасыр|сынып|курс|орын|қатар|тоқсан|бет|бап|тарау)", re.IGNORECASE)
_MONTH_RE = "|".join(MONTHS)


def _fix_homoglyphs(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        word = match.group()
        if re.search(r"[Ѐ-ӿ]", word):
            return word.translate(_HOMOGLYPHS)
        return word

    return re.sub(r"[\wəƏ]+", repl, text)


def _number_with_suffix(number: str, suffix: str) -> str:
    """Handle '5-ші', '2019-жылы', '10-нан', '1990-шы'."""
    value = int(number)
    match = _ORDINAL_SUFFIX_RE.match(suffix)
    if match:
        return ordinal(value) + match.group(2)
    if _YEAR_WORD_RE.match(suffix):
        return f"{ordinal(value)} {suffix}"
    # Case or plural suffix glued to the spoken form: 5-ке → беске.
    return cardinal(value) + suffix


def normalize_kk(text: str, *, lowercase: bool = False) -> str:
    """Normalize Kazakh text into the spoken form used for training.

    ``lowercase=True`` gives the exact input form of the Kazakh fine-tune,
    which was trained on lower-cased KazakhTTS2 text.
    """
    text = unicodedata.normalize("NFC", str(text))
    text = _fix_homoglyphs(text)
    text = text.replace("−", "-")

    # Dates: 15.03.2019 → он бесінші наурыз екі мың он тоғызыншы жыл
    def date_numeric(match: re.Match[str]) -> str:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return match.group()
        return f"{ordinal(year)} жылғы {ordinal(day)} {MONTHS[month - 1]}"

    text = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b(?:\s*ж\.)?", date_numeric, text)

    # Day before a month name: 15 наурыз → он бесінші наурыз
    text = re.sub(
        rf"\b(\d{{1,2}})\s+({_MONTH_RE})",
        lambda m: f"{ordinal(int(m.group(1)))} {m.group(2)}",
        text,
        flags=re.IGNORECASE,
    )

    # Times: 14:30 → он төрт отыз
    def clock(match: re.Match[str]) -> str:
        minutes = int(match.group(2))
        spoken = f"{cardinal(int(match.group(1)))} {cardinal(minutes) if minutes else 'нөл нөл'}"
        return spoken + (match.group(3) or "")

    text = re.sub(r"\b(\d{1,2}):(\d{2})\b(?:-([^\W\d_]+))?", clock, text)

    # Roman numerals before ғасыр / ғ. : XXI ғасыр → жиырма бірінші ғасыр
    def roman(match: re.Match[str]) -> str:
        value = roman_to_int(match.group(1))
        return f"{ordinal(value)} {match.group(2)}" if value else match.group()

    text = re.sub(r"\b([IVXLCDM]+)\s+(ғасыр\w*|ғ\.)", roman, text)

    # Ranges: 2019-2020 жылдары → екі мың он тоғызыншы-екі мың жиырмасыншы жылдары
    def number_range(match: re.Match[str]) -> str:
        left, right, word = match.group(1), match.group(2), match.group(3)
        spell = ordinal if _YEAR_WORD_RE.match(word) else cardinal
        return f"{spell(int(left))}-{spell(int(right))} {word}"

    text = re.sub(r"\b(\d+)\s*[-–—]\s*(\d+)\s+([^\W\d]+)", number_range, text)

    # Grouped thousands: 1 000 000 or 1,000,000 → 1000000
    text = re.sub(r"\b\d{1,3}(?:[   ]\d{3})+\b", lambda m: re.sub(r"\D", "", m.group()), text)

    # Decimals: 9,5 or 9.5
    text = re.sub(r"\b(\d+)[.,](\d{1,3})\b", lambda m: decimal(m.group(1), m.group(2)), text)

    # Number with a hyphenated suffix: 2019-жылы, 5-ші, 10-нан
    text = re.sub(
        r"\b(\d+)\s*-\s*([^\W\d_]+)",
        lambda m: _number_with_suffix(m.group(1), m.group(2)),
        text,
    )

    # Number followed by жыл/ғасыр... is an ordinal: 2019 жылы
    text = re.sub(
        r"\b(\d+)(\s+)(?=(жыл|ғасыр|сынып|курс)\w*)",
        lambda m: ordinal(int(m.group(1))) + m.group(2),
        text,
        flags=re.IGNORECASE,
    )

    # Symbols after numbers: 25% → 25 пайыз, $5 → 5 доллар
    text = re.sub(r"([$€])\s*(\d+)", lambda m: f"{m.group(2)} {SYMBOLS[m.group(1)]}", text)
    for symbol, word in SYMBOLS.items():
        text = text.replace(symbol, f" {word} ")

    # Vulgar fractions: 24½ → 24 жарым
    for fraction, word in FRACTIONS.items():
        text = text.replace(fraction, f" {word} ")

    # Remaining plain integers
    text = re.sub(r"\d+", lambda m: f" {cardinal(int(m.group()))} ", text)

    # Abbreviations as whole tokens
    for abbreviation, expansion in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        pattern = rf"(?<![\w.]){re.escape(abbreviation)}(?!\w)"
        text = re.sub(pattern, expansion, text)

    text = re.sub(r"\s+([,.!?;:»)])", r"\1", text)
    text = re.sub(r"([«(])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


if __name__ == "__main__":
    import sys

    lower = "--lowercase" in sys.argv[1:]
    for line in sys.stdin:
        print(normalize_kk(line, lowercase=lower))
