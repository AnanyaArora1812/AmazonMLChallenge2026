"""
normalization.py
=================
Reusable data-cleaning / normalization functions for the Business Entity
Resolution challenge (Source 1 / Source 2 / Source 3 records).

Design principles (per team requirements):
  1. NEVER destroy information needed for matching. Every function returns a
     *new* cleaned value; callers keep the original raw column alongside it.
     Anything ambiguous is preserved rather than deleted.
  2. Countries are an OPEN set of string labels. Nothing here special-cases
     or filters to {US, India} — France (present only in test) and any other
     future label must flow through untouched.
  3. No external APIs / databases / geocoders. All lookups used below
     (legal-suffix words, US state abbreviations, generic street-suffix
     abbreviations) are small, static, hand-written reference lists of
     *general public knowledge* (postal abbreviations, corporate suffixes),
     not a business-identity database, so they don't violate the "no
     external business lookup" rule.
  4. Every rule is conservative on purpose: F_0.5 punishes false merges 2x
     harder than missed matches, so when a transformation is ambiguous
     (e.g. "St" = Street vs. Saint) we default to leaving it alone rather
     than guessing and risking a corrupted token that causes a false match
     or a missed one.

Typical usage
-------------
    from normalization import preprocess_dataframe
    df = preprocess_dataframe(df)   # adds *_clean / *_core / *_tokens columns

Every individual function is also exported standalone so blocking / matching
code can call just the piece it needs (e.g. `normalize_business_name` alone).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover - keeps the module importable either way
    unidecode = None


# ---------------------------------------------------------------------------
# 0. Reference data (static, hand-curated, no external calls)
# ---------------------------------------------------------------------------

# Legal / corporate-FORM suffixes only — tokens that indicate how a business
# is legally incorporated and carry (in practice, in this dataset) no
# brand-differentiating meaning of their own. These are safe to auto-peel
# off the end of a name because "Acme Inc" and "Acme LLC" are overwhelmingly
# the same underlying entity recorded with a different legal form.
LEGAL_SUFFIXES = {
    # generic / US
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "llp", "ltd", "limited", "lp", "lllp", "pllc", "pc", "plc",
    # India
    "pvt", "private", "opc",
    # France
    "sarl", "sas", "sasu", "sa", "eurl", "cie", "gie", "sci",
}

# Generic business-type / descriptor words that are grammatically similar to
# legal suffixes (they often sit at the end of a name) but, unlike the set
# above, frequently ARE the brand-differentiating part of the name: "Acme
# Group" and "Acme Partners" and "Acme Ventures" are typically three
# DIFFERENT businesses, not the same one in different legal forms. EDA's
# "Surgical Partners LLC" (-> would wrongly reduce to a bare 'Surgical' core
# if 'Partners' were peeled too) is exactly this failure mode. We therefore
# deliberately do NOT auto-peel these into the suffix field — they stay in
# `core` — trading a slightly less-aggressive core-name match for protection
# against false merges, consistent with F_0.5's 2x penalty on precision.
GENERIC_DESCRIPTOR_WORDS = {
    "group", "holdings", "enterprises", "associates", "partners", "ventures",
    "trust", "union",
}

# Suffix tokens that are safe to *canonicalize* (many spellings -> one form)
# because they are unambiguous corporate abbreviations. We deliberately do
# NOT canonicalize every suffix (e.g. we leave "Co" as-is) because some short
# tokens double as ordinary words; the canonical map below only touches
# tokens with no common alternate meaning.
SUFFIX_CANON_MAP = {
    "incorporated": "inc",
    "corporation": "corp",
    "limited": "ltd",
    "private": "pvt",
    "llp": "llp",
    "company": "co",
}

# Markers that introduce a "doing business as" trade name. EDA showed a
# recurring pattern: "<random-looking legal name> DBA <real trade name>",
# where the left side is often synthetic noise and the right side is the
# name that actually recurs across sources. We split on these rather than
# discard either half, since either can be the one that matches.
DBA_PATTERN = re.compile(r"\b(?:d[./]?\s?b[./]?\s?a\.?:?|doing business as)\b", re.IGNORECASE)

# Literal placeholder strings that show up as address *tokens* meaning
# "field missing", not real address content (confirmed in EDA: 'null',
# '<null>', 'n/a' embedded mid-address at ~2-3% frequency).
MISSING_PLACEHOLDER_RE = re.compile(r"(?<![a-z])(null|n/?a|none|unknown)(?![a-z])", re.IGNORECASE)

# Honorific / convenience prefixes seen prepended to Indian business names.
# We *tag* these rather than delete them outright (see normalize_business_name).
NAME_PREFIX_HONORIFICS = re.compile(r"^(m/s\.?|mr\.?|mrs\.?|ms\.?|smt\.?|sri\.?|shri\.?|dr\.?)\s+", re.IGNORECASE)

# Street-suffix abbreviation -> full word. Only the UNAMBIGUOUS ones are
# expanded automatically. "St" and "Dr" are handled separately with a
# position-aware heuristic because they collide with "Saint" / "Doctor".
UNAMBIGUOUS_ADDR_ABBREV = {
    "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "ln": "lane", "hwy": "highway", "apt": "apartment", "ste": "suite",
    "fl": "floor", "flr": "floor", "ct": "court", "pl": "place",
    "sq": "square", "ter": "terrace", "cir": "circle", "pkwy": "parkway",
    "hts": "heights", "twp": "township",
}

US_STATE_ABBREV = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok",
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy",
}

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
NON_LATIN_RE = re.compile(r"[^\x00-\x7F\u00C0-\u024F]")  # outside ASCII + Latin-extended


# ---------------------------------------------------------------------------
# 1. Low-level building blocks
# ---------------------------------------------------------------------------

def clean_missing_placeholder(text: str) -> str:
    """Remove literal placeholder tokens ('null', 'n/a', 'none', 'unknown')
    that appear as stand-ins for a missing address component.

    Why: EDA found ~2-3% of addresses contain a literal 'null' (or '<null>',
    'N/A') as one comma-separated component, e.g.
    '067 PRODUCTION CT, NULL, INDEPENDENCE, KY'. Left in place, this token
    pollutes token-overlap / TF-IDF features with a high-frequency word that
    carries zero real signal, and can even accidentally match another
    record's genuine word if not filtered. We remove the token (and a
    leftover empty comma-separated slot) but keep everything else untouched.
    """
    if text is None:
        return ""
    text = MISSING_PLACEHOLDER_RE.sub("", text)
    # collapse the empty slot / stray punctuation the removal can leave behind
    text = re.sub(r"[<>]", "", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"^\s*,\s*|\s*,\s*$", "", text)
    return text.strip()


def normalize_unicode(text: str, transliterate: bool = False) -> str:
    """Normalize Unicode form and optionally transliterate non-Latin scripts.

    Why: EDA found (a) genuine accented characters in French text
    ('LÈGE-CAP-FERRET'), (b) *noise*-injected accents on ordinary English
    words ('Rápid', 'Héritage', 'Órange') that are typos rather than
    linguistic content, and (c) names written entirely in Devanagari,
    Telugu, Malayalam, Bengali, etc. in Source 2/3 while Source 1 (and other
    sources) may hold the Latin-script name for the same business.

    We NFKD-normalize and strip combining diacritical marks — but ONLY on
    Latin-script characters (Basic Latin, Latin-1 Supplement, Latin
    Extended A/B/Additional). This is safe there: it never merges two
    *different* words, it only removes accent noise ('Rápid' -> 'Rapid',
    'Héritage' -> 'Heritage'). It is critically NOT applied to Indic (or
    other non-Latin) scripts: an early version of this function ran NFKD
    over the whole string, which silently strips the combining vowel signs
    (matras) that Devanagari/Bengali/Telugu/Malayalam etc. depend on to be
    readable at all — e.g. 'পশ্চিমবঙ্গ' (West Bengal) was corrupted to
    'পশচিমবঙগ'. Those combining marks are load-bearing content there, not
    noise, so we leave every non-Latin character completely untouched.

    Transliteration of non-Latin scripts (c) is a strictly *lossy*, one-way
    operation (e.g. Devanagari -> ASCII romanization is approximate), so it
    is OFF by default and only applied when the caller explicitly asks for
    it (transliterate=True) — typically only for building a fuzzy-matching
    key, never for a "clean but keep info" display column.
    """
    if text is None:
        return ""
    # Latin-ish codepoint ranges where stripping a combining diacritic is
    # safe: Basic Latin, Latin-1 Supplement, Latin Extended-A/B, IPA
    # Extensions, combining diacriticals themselves, Latin Extended
    # Additional (Vietnamese etc). Anything at/above Greek (0x0370) and
    # outside these extra Latin blocks is left completely alone.
    def _is_latinish(cp: int) -> bool:
        return cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF

    out_chars = []
    for ch in text:
        if _is_latinish(ord(ch)):
            decomposed = unicodedata.normalize("NFKD", ch)
            out_chars.append("".join(c for c in decomposed if not unicodedata.combining(c)))
        else:
            out_chars.append(ch)
    stripped = "".join(out_chars)
    if transliterate and unidecode is not None:
        stripped = unidecode(stripped)
    return stripped


def normalize_whitespace_and_punct(text: str) -> str:
    """Collapse whitespace, strip bracket wrappers, normalize quotes/dashes.

    Why: EDA found bracket-wrapped single words that are otherwise normal
    tokens ('[Limited] Next Traders', 'Global [Procesong]') — the brackets
    themselves carry no meaning, so we drop the bracket characters but keep
    their contents (never delete the word inside). We also collapse repeated
    whitespace (~10-15% of names/addresses have double spaces) and unify
    smart quotes/dashes to their plain-ASCII equivalents so 'O’Brien' and
    "O'Brien" compare equal downstream.
    """
    if text is None:
        return ""
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # drop bracket characters but keep their contents
    text = re.sub(r"[\[\]{}()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_ampersand(text: str) -> str:
    """Canonicalize '&' and stand-alone 'and' to the same token ('and').

    Why: EDA found both forms used interchangeably as a name conjunction
    ('Spencer, Hill and Sullivan' vs. 'Jpa & Brothers ...'). Since '&' has
    no other use in these names, canonicalizing is safe and removes a very
    common source of otherwise-identical names failing to match.
    """
    if text is None:
        return ""
    text = re.sub(r"\s*&\s*", " and ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_dba(name: str) -> tuple[str, str | None]:
    """Split a "<legal name> DBA <trade name>" string into (primary, alias).

    Why: EDA found this pattern in a few percent of Source 3 (and some
    Source 2) names, and the pre-DBA half is frequently a synthetic-looking
    string (e.g. 'Rizahaloquo', 'Umbrafayevio') that will never match
    anything, while the post-DBA half is a normal, matchable business name.
    Rather than guessing which half is "the" name, we keep BOTH: `primary`
    (post-DBA if present, else the whole string) is what should drive the
    main matching key; `alias` (pre-DBA, or None) is kept as a secondary
    field so a matcher can optionally also compare against it. No
    information is discarded.
    """
    if name is None:
        return "", None
    parts = DBA_PATTERN.split(name, maxsplit=1)
    if len(parts) == 2:
        legal_part = parts[0].strip(" ,-.:")
        trade_part = parts[1].strip(" ,-.:")
        if trade_part:
            return trade_part, (legal_part or None)
    return name, None


def strip_name_honorific(name: str) -> tuple[str, str | None]:
    """Remove a leading honorific/convenience prefix (M/s, Mr, Mrs, Smt, Sri...).

    Why: EDA found Indian-style prefixes like 'M/s', 'Smt', 'Sri' attached to
    some business names ('Smt Jeevan Tmh Private Limited'). These are
    conventional honorifics, not part of the registered brand, and their
    presence/absence differs across sources for the same entity. We return
    the prefix separately (never drop it silently) so it can still be used
    as a weak feature if useful, while the de-prefixed name becomes the
    primary matching key.
    """
    if name is None:
        return "", None
    m = NAME_PREFIX_HONORIFICS.match(name)
    if m:
        return name[m.end():].strip(), m.group(1).rstrip(".").lower()
    return name, None


def split_legal_suffix(name_tokens: list[str]) -> tuple[list[str], list[str]]:
    """Peel trailing legal-suffix tokens off a tokenized name.

    Why: 'Acme Corp' and 'Acme LLC' and 'Acme' are frequently the *same*
    real-world business recorded with a different (or missing) legal form
    across sources. Separating the suffix lets a matcher compare the "core"
    brand name on its own (high-signal) while still keeping the suffix
    available as a secondary, lower-weight feature — we never discard it.
    Only whole tokens in LEGAL_SUFFIXES are peeled, and only from the end
    inward, so a suffix word appearing mid-name ("Corp Solutions Inc") is
    left in the core where it might be semantically load-bearing.
    """
    tokens = list(name_tokens)
    suffixes = []
    while tokens:
        last = tokens[-1].lower().strip(".")
        if last in LEGAL_SUFFIXES:
            suffixes.insert(0, tokens.pop().rstrip("."))
            continue
        # handle hyphen-joined suffixes like "Private-Limited" (a single
        # whitespace token whose hyphen-separated parts are BOTH known
        # suffix words) without touching hyphens anywhere else (e.g. "U-3")
        if "-" in last:
            parts = [p for p in last.split("-") if p]
            if parts and all(p in LEGAL_SUFFIXES for p in parts):
                suffixes[0:0] = parts
                tokens.pop()
                continue
        break
    return tokens, suffixes


def canonicalize_suffix_tokens(suffix_tokens: list[str]) -> list[str]:
    """Map suffix synonyms to one canonical spelling (Incorporated -> Inc, etc).

    Why: only applied to the *already-separated* suffix list (see
    split_legal_suffix), never to the core name, so there's no risk of a
    canonicalization rule accidentally rewriting an ordinary word.
    """
    return [SUFFIX_CANON_MAP.get(t.lower(), t.lower()) for t in suffix_tokens]


def normalize_addr_abbreviations(text: str) -> str:
    """Expand unambiguous street-type abbreviations to a canonical full word.

    Why: EDA showed the *same* street type spelled inconsistently across
    sources ('Road' vs 'Rd', 'Street' vs 'St', 'Avenue' vs 'Ave', 'Drive'
    vs 'Dr'). Expanding to one canonical form raises token-overlap between
    matching addresses. We ONLY expand tokens with no other common meaning
    (Rd, Ave, Blvd, Ln, Hwy, ...). 'St' and 'Dr' are deliberately excluded
    here and handled by `disambiguate_st_dr` instead, because 'St' also
    means 'Saint' (as in 'St Louis') and 'Dr' also means 'Doctor' (as a
    name-title in Source 2/3 business names) — blindly expanding those two
    would corrupt real place/person names and manufacture false matches or
    breaks, which is worse than leaving an occasional abbreviation
    unexpanded (F_0.5 punishes false positives twice as hard as misses).
    """
    if text is None:
        return ""

    def repl(m):
        tok = m.group(0)
        full = UNAMBIGUOUS_ADDR_ABBREV.get(tok.lower().rstrip("."))
        if full is None:
            return tok
        return full.capitalize() if tok[0].isupper() else full

    return re.sub(r"\b[A-Za-z]{1,5}\.?\b", repl, text)


def disambiguate_st_dr(text: str) -> str:
    """Context-aware handling of the 'St'/'Dr' street-vs-other ambiguity.

    Why: 'St' means 'Street' when it TRAILS a segment (end of string, or
    right before a comma) — e.g. '123 Main St, Springfield' — but means
    'Saint' when it's followed by another capitalized word in the same
    segment — e.g. 'St Louis', 'St Petersburg'. Similarly 'Dr' at the very
    start of a business name is almost always the honorific 'Doctor'
    ('Dr Rb Design'), while 'Dr' trailing an address segment is 'Drive'.
    This heuristic only fires in the unambiguous positional cases; anywhere
    else the token is left exactly as written rather than guessed at,
    consistent with the "don't risk false matches" requirement.
    """
    if text is None:
        return ""
    # trailing "St" at end of a comma-segment -> Street
    text = re.sub(r"\bSt\.?(?=\s*(,|$))", "Street", text)
    text = re.sub(r"\bst\.?(?=\s*(,|$))", "street", text)
    # trailing "Dr" at end of a comma-segment -> Drive
    text = re.sub(r"\bDr\.?(?=\s*(,|$))", "Drive", text)
    text = re.sub(r"\bdr\.?(?=\s*(,|$))", "drive", text)
    # "St" followed by a capitalized word (not end/comma) -> Saint
    text = re.sub(r"\bSt\.?\s+(?=[A-Z])", "Saint ", text)
    return text


def extract_postal_code(address: str) -> str | None:
    """Pull out a 5-6 digit postal / PIN code if one is present, else None.

    Why: both US ZIP (5 digits) and Indian PIN (6 digits) codes are strong,
    low-noise blocking keys when present (~7% of addresses in EDA). We only
    *extract* it as an auxiliary field for blocking; we do not remove it
    from the address text, so nothing is lost for later use.
    """
    if not address:
        return None
    m = re.search(r"\b(\d{5,6})\b", address)
    return m.group(1) if m else None


def normalize_country(country: str) -> str:
    """Trim/title-case a country label without assuming a closed set.

    Why: the challenge explicitly requires treating country as an open set
    (test adds 'France', unseen in training) — this function only fixes
    incidental casing/whitespace, it never maps, filters, or validates
    against a fixed list of countries.
    """
    if country is None:
        return ""
    return country.strip()


# ---------------------------------------------------------------------------
# 2. Composed, field-level pipelines
# ---------------------------------------------------------------------------

@dataclass
class NormalizedName:
    raw: str
    clean: str                 # de-noised, still human-readable
    core: str                  # clean with legal suffix peeled off
    suffix: str                # canonicalized legal suffix, '' if none
    alias: str | None = None   # pre-DBA legal name, if a DBA pattern was found
    honorific: str | None = None
    tokens: list = field(default_factory=list)
    fuzzy_key: str = ""        # aggressive, transliterated key for approximate matching


def normalize_business_name(raw_name: str) -> NormalizedName:
    """Full name-normalization pipeline. Returns every intermediate value
    so downstream code can pick whichever field suits a given feature
    (exact-token overlap vs. fuzzy string similarity vs. blocking key).

    Pipeline (each step's rationale is documented on the function it calls):
      1. Unicode clean (strip injected/real accents, NFKD)
      2. Whitespace/bracket/quote normalization
      3. Split off a DBA trade name if present (keep both halves)
      4. Strip a leading honorific/M-s prefix (kept separately)
      5. Ampersand canonicalization
      6. Tokenize, peel + canonicalize a trailing legal suffix
      7. Build a separate, transliterated `fuzzy_key` for cross-script
         approximate matching (lossy — only for similarity scoring, never
         for display or exact-match logic)
    """
    raw_name = raw_name or ""
    step1 = normalize_unicode(raw_name, transliterate=False)
    step2 = normalize_whitespace_and_punct(step1)
    primary, alias = strip_dba(step2)
    primary, honorific = strip_name_honorific(primary)
    primary = normalize_ampersand(primary)
    primary = re.sub(r"\s+", " ", primary).strip()

    tokens = primary.split(" ") if primary else []
    core_tokens, suffix_tokens = split_legal_suffix(tokens)
    suffix_tokens = canonicalize_suffix_tokens(suffix_tokens)

    core = " ".join(core_tokens)
    suffix = " ".join(suffix_tokens)
    clean = (core + (" " + suffix if suffix else "")).strip()

    fuzzy_source = normalize_unicode(raw_name, transliterate=True)
    fuzzy_source = normalize_whitespace_and_punct(fuzzy_source)
    fuzzy_source, _ = strip_dba(fuzzy_source)
    fuzzy_source, _ = strip_name_honorific(fuzzy_source)
    fuzzy_key = re.sub(r"[^a-z0-9 ]", "", fuzzy_source.lower()).strip()
    fuzzy_key = re.sub(r"\s+", " ", fuzzy_key)

    return NormalizedName(
        raw=raw_name, clean=clean, core=core, suffix=suffix,
        alias=alias, honorific=honorific, tokens=core_tokens, fuzzy_key=fuzzy_key,
    )


@dataclass
class NormalizedAddress:
    raw: str
    clean: str
    tokens: list = field(default_factory=list)
    postal_code: str | None = None
    has_state_abbrev: bool = False
    fuzzy_key: str = ""


def normalize_business_address(raw_address: str, country: str | None = None) -> NormalizedAddress:
    """Full address-normalization pipeline.

    `country` is accepted purely as an optional hint for logging/analysis;
    it never filters or branches into country-specific hard-coded lookups
    beyond the *generic, public* US-state-abbreviation set used only to set
    the informational `has_state_abbrev` flag (not used to alter the text).
    This keeps the function honest about the "don't assume only US/India"
    and "no external data" constraints while still surfacing a useful
    signal the modeling team can weight per-country if they choose to.
    """
    raw_address = raw_address or ""
    postal_code = extract_postal_code(raw_address)

    step1 = clean_missing_placeholder(raw_address)
    step2 = normalize_unicode(step1, transliterate=False)
    step3 = normalize_whitespace_and_punct(step2)
    step4 = disambiguate_st_dr(step3)
    step5 = normalize_addr_abbreviations(step4)
    clean = re.sub(r"\s*,\s*", ", ", step5).strip(" ,")
    clean = re.sub(r"\s+", " ", clean)

    tokens = [t.strip(",.") for t in clean.split(" ") if t.strip(",.")]
    # Only trust a token as a US state code if it appears in ALL CAPS in the
    # cleaned text (real state codes are consistently written that way in
    # this dataset) AND sits in the last two tokens (states trail an
    # address). This avoids false positives like French "la"/"de" (lower-
    # case function words) matching Louisiana/Delaware's abbreviations,
    # which the case-insensitive version of this check was doing.
    has_state = any(
        t.isupper() and t.lower() in US_STATE_ABBREV
        for t in tokens[-2:]
    )

    fuzzy_source = normalize_unicode(step1, transliterate=True)
    fuzzy_source = normalize_whitespace_and_punct(fuzzy_source)
    fuzzy_key = re.sub(r"[^a-z0-9 ]", "", fuzzy_source.lower())
    fuzzy_key = re.sub(r"\s+", " ", fuzzy_key).strip()

    return NormalizedAddress(
        raw=raw_address, clean=clean, tokens=tokens,
        postal_code=postal_code, has_state_abbrev=has_state, fuzzy_key=fuzzy_key,
    )


# ---------------------------------------------------------------------------
# 3. DataFrame-level convenience wrapper
# ---------------------------------------------------------------------------

def preprocess_dataframe(df, name_col="business_name", addr_col="business_address",
                          country_col="country"):
    """Apply the full normalization pipeline to a source dataframe.

    Adds new columns; never overwrites or drops the original raw columns,
    so nothing is destroyed and every step remains auditable/reversible.
    Memory-optimized single-pass list allocation.
    """
    out = df.copy()
    n_len = len(out)

    name_vals = out[name_col].fillna("").astype(str).values
    name_clean = [None] * n_len
    name_core = [None] * n_len
    name_suffix = [None] * n_len
    name_alias = [None] * n_len
    name_honorific = [None] * n_len
    name_fuzzy_key = [None] * n_len

    for i in range(n_len):
        n = normalize_business_name(name_vals[i])
        name_clean[i] = n.clean
        name_core[i] = n.core
        name_suffix[i] = n.suffix
        name_alias[i] = n.alias
        name_honorific[i] = n.honorific
        name_fuzzy_key[i] = n.fuzzy_key

    out["name_clean"] = name_clean
    out["name_core"] = name_core
    out["name_suffix"] = name_suffix
    out["name_alias"] = name_alias
    out["name_honorific"] = name_honorific
    out["name_fuzzy_key"] = name_fuzzy_key

    if country_col in out.columns:
        country_vals = out[country_col].fillna("").astype(str).values
        country_clean_list = [normalize_country(c) for c in country_vals]
        out["country_clean"] = country_clean_list
    else:
        country_clean_list = [""] * n_len
        out["country_clean"] = ""

    addr_vals = out[addr_col].fillna("").astype(str).values

    addr_clean = [None] * n_len
    address_postal_code = [None] * n_len
    address_has_state_abbrev = [None] * n_len
    address_fuzzy_key = [None] * n_len

    for i in range(n_len):
        a = normalize_business_address(addr_vals[i], country_clean_list[i])
        addr_clean[i] = a.clean
        address_postal_code[i] = a.postal_code
        address_has_state_abbrev[i] = a.has_state_abbrev
        address_fuzzy_key[i] = a.fuzzy_key

    out["address_clean"] = addr_clean
    out["address_postal_code"] = address_postal_code
    out["address_has_state_abbrev"] = address_has_state_abbrev
    out["address_fuzzy_key"] = address_fuzzy_key

    return out


__all__ = [
    "clean_missing_placeholder", "normalize_unicode", "normalize_whitespace_and_punct",
    "normalize_ampersand", "strip_dba", "strip_name_honorific", "split_legal_suffix",
    "canonicalize_suffix_tokens", "normalize_addr_abbreviations", "disambiguate_st_dr",
    "extract_postal_code", "normalize_country", "normalize_business_name",
    "normalize_business_address", "preprocess_dataframe",
    "NormalizedName", "NormalizedAddress",
    "LEGAL_SUFFIXES", "GENERIC_DESCRIPTOR_WORDS", "SUFFIX_CANON_MAP",
    "UNAMBIGUOUS_ADDR_ABBREV", "US_STATE_ABBREV",
]
