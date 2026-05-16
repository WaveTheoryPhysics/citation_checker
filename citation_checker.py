#!/usr/bin/env python3
"""
citation_checker.py
-------------------
Check academic citations for existence and correctness.

Methodology mirrors Zhao et al. (2025) "LLM hallucinations in the wild":
  1. Auto-detect input format and parse into structured fields
  2. Query Semantic Scholar, CrossRef, and OpenAlex (all free, no key needed)
  3. Compute fuzzy string-similarity match scores
  4. Flag citations as VERIFIED / SUSPICIOUS / NOT_FOUND
  5. For found papers, check author/year correctness

Supported input formats (auto-detected by extension and content)
-----------------------------------------------------------------
  .bib          BibTeX  (@article, @book, @inproceedings, etc.)
  .ris          RIS     (TY/AU/TI/ER tags)
  .txt / other  Plain text, one reference per line

Usage
-----
  python citation_checker.py references.bib
  python citation_checker.py references.ris
  python citation_checker.py references.txt
  python citation_checker.py --demo                 # live API demo
  python citation_checker.py --mock                 # offline demo (no API calls)
  python citation_checker.py refs.bib --output results.json
  python citation_checker.py refs.bib --workers 8  # faster with more threads

Dependencies
------------
  pip install requests rapidfuzz tqdm colorama

APIs used (free, no key required)
-----------------------------------
  Semantic Scholar  https://api.semanticscholar.org/graph/v1/
  CrossRef          https://api.crossref.org/works/
  OpenAlex          https://api.openalex.org/works
"""

import argparse
import json
import os
import re
import sys
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
from rapidfuzz import fuzz
from tqdm import tqdm

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class _Dummy:
        def __getattr__(self, _): return ""
    Fore = Style = _Dummy()


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ParsedRef:
    raw: str                            # original string / entry for display
    title: Optional[str] = None
    authors: list = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None          # non-DOI URL (GitHub, Zenodo page, etc.)
    arxiv_id: Optional[str] = None     # arXiv ID from eprint= field, e.g. "1402.7073"
    entry_key: Optional[str] = None    # BibTeX cite key, if available
    entry_type: Optional[str] = None   # article / book / etc., if available


@dataclass
class CheckResult:
    raw: str
    parsed: ParsedRef
    status: str = "NOT_CHECKED"        # VERIFIED | SUSPICIOUS | NOT_FOUND | ERROR
    title_score: float = 0.0
    author_score: float = 0.0
    year_match: Optional[bool] = None
    matched_title: Optional[str] = None
    matched_authors: list = field(default_factory=list)
    matched_year: Optional[int] = None
    matched_doi: Optional[str] = None
    matched_url: Optional[str] = None  # verified URL for software/web references
    matched_venue: Optional[str] = None
    source: Optional[str] = None
    notes: list = field(default_factory=list)


# =============================================================================
# Format detection
# =============================================================================

def detect_format(path: str, text: str) -> str:
    """Return 'bibtex', 'ris', or 'plaintext'."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.bib':
        return 'bibtex'
    if ext == '.ris':
        return 'ris'
    # Sniff content even if extension is wrong
    head = text[:2000]
    if re.search(r'@\w+\s*\{', head):
        return 'bibtex'
    if re.search(r'^TY\s+-\s+\w+', head, re.MULTILINE):
        return 'ris'
    return 'plaintext'


# =============================================================================
# BibTeX parser
# =============================================================================

def load_bibtex(text: str) -> list:
    """
    Parse a .bib file into a list of ParsedRef objects.
    Handles nested braces and all standard entry types.
    Skips @comment, @string, @preamble.
    """
    SKIP_TYPES = {'comment', 'string', 'preamble'}
    entry_start_re = re.compile(r'@(\w+)\s*\{', re.IGNORECASE)
    refs = []
    i = 0

    while i < len(text):
        m = entry_start_re.search(text, i)
        if not m:
            break

        entry_type = m.group(1).lower()
        if entry_type in SKIP_TYPES:
            i = m.end()
            continue

        # Find the matching closing brace
        brace_pos = m.end() - 1   # position of '{'
        depth = 0
        j = brace_pos
        while j < len(text):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1

        body = text[brace_pos + 1 : j]

        # Extract cite key (first token before the first comma)
        key_m = re.match(r'\s*([^,\s]+)\s*,', body)
        entry_key = key_m.group(1).strip() if key_m else ''

        # Extract fields
        title   = strip_bibtex_markup(_bib_field(body, 'title') or '') or None
        author  = _bib_field(body, 'author')
        year_s  = _bib_field(body, 'year')
        _raw_doi = _bib_field(body, 'doi')
        _raw_url = _bib_field(body, 'url')
        doi, url = _classify_doi_or_url(_raw_doi or _raw_url)

        # arXiv ID from eprint= / archivePrefix= fields
        arxiv_id = _extract_arxiv_id(
            _bib_field(body, 'eprint'),
            _bib_field(body, 'archiveprefix') or _bib_field(body, 'archivePrefix'),
        )

        authors = _split_bib_authors(author) if author else []
        year    = int(year_s) if year_s and year_s.strip().isdigit() else None

        # Reconstruct a readable raw string for display
        raw = f"@{entry_type}{{{entry_key}}} {title or '?'} ({year or '?'})"  # title already stripped

        ref = ParsedRef(
            raw=raw,
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            url=url,
            arxiv_id=arxiv_id,
            entry_key=entry_key,
            entry_type=entry_type,
        )
        refs.append(ref)
        i = j + 1

    return refs


def _bib_field(body: str, field_name: str) -> Optional[str]:
    """
    Extract a BibTeX field value from an entry body.
    Handles:  field = {possibly {nested} braces}
              field = "double quoted"
              field = 1992   (bare number, for year)
    """
    pat = re.compile(
        r'(?:^|,)\s*' + re.escape(field_name) + r'\s*=\s*',
        re.IGNORECASE | re.MULTILINE,
    )
    m = pat.search(body)
    if not m:
        return None

    rest = body[m.end():].lstrip()
    if not rest:
        return None

    if rest[0] == '{':
        # Brace-delimited: count depth
        depth = 0
        buf = []
        for ch in rest:
            if ch == '{':
                depth += 1
                if depth > 1:
                    buf.append(ch)
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    break
                buf.append(ch)
            else:
                buf.append(ch)
        return ''.join(buf).strip()

    elif rest[0] == '"':
        # Quote-delimited
        end = rest.index('"', 1)
        return rest[1:end].strip()

    else:
        # Bare value (e.g. year = 1992)
        bare_m = re.match(r'[\w\-]+', rest)
        return bare_m.group(0) if bare_m else None


def _split_bib_authors(author_str: str) -> list:
    """
    Split a BibTeX author field on ' and ' and normalise "Last, First" order.
    """
    parts = re.split(r'\s+and\s+', author_str, flags=re.IGNORECASE)
    result = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if ',' in p:
            last, *rest = p.split(',', 1)
            p = rest[0].strip() + ' ' + last.strip()
        result.append(p.strip())
    return result



def strip_bibtex_markup(s: str) -> str:
    """
    Strip BibTeX/LaTeX markup from a title string before sending it to APIs.

    BibTeX uses braces for two purposes that both corrupt API queries:
      - Capitalisation protection: {R}iemannian  ->  Riemannian
      - Abbreviation protection:   {THINGS}      ->  THINGS
    LaTeX math/commands also appear in titles:
      - Superscripts:  $^{\\mathrm{3D}}$  ->  (removed)
      - Spacing cmds:  {H\\,I}            ->  HI
      - Emphasis:      \\emph{word}       ->  word
    """
    import re as _re
    # 1. Extract superscript/subscript math that contains meaningful text.
    #    $^{\mathrm{3D}}$ -> 3D,  $^{3D}$ -> 3D  (no space, attaches to word)
    s = _re.sub(r'\$\^\{\\?(?:\w+)\{([^}]+)\}\}\$', r'\1', s)  # $^{\mathrm{3D}}$ -> 3D
    s = _re.sub(r'\$\^\{([^}$]+)\}\$', r'\1', s)                     # $^{3D}$ -> 3D
    s = _re.sub(r'\$_\{([^}$]+)\}\$', r'\1', s)                       # $_{eff}$ -> eff
    # 2. Remove remaining math spans entirely (conditions, equations)
    s = _re.sub(r'\$[^$]*\$', '', s)
    # 2. Strip LaTeX single-char commands: \, \; \! etc.
    s = _re.sub(r'\\\\[^a-zA-Z\s{]', '', s)
    # 3. Strip named LaTeX commands, keeping their brace content: \mathrm{3D} -> 3D
    for _ in range(5):
        s = _re.sub(r'\\\\[a-zA-Z]+\{([^{}]*)\}', r'\1', s)
    # 4. Strip bare LaTeX commands with no braces: \emph, \it, etc.
    s = _re.sub(r'\\\\[a-zA-Z]+\s*', '', s)
    # 6. Strip protective braces, keeping content: {THINGS} -> THINGS  (repeat for nested)
    for _ in range(5):
        s = _re.sub(r'\{([^{}]*)\}', r'\1', s)
    # 7. Remove any remaining stray braces
    s = s.replace('{', '').replace('}', '')
    # 8. Normalise em-dashes and whitespace
    s = _re.sub(r'\s*---\s*', ' - ', s)
    s = _re.sub(r'\s+', ' ', s).strip()
    return s


# Patterns for classifying reference URLs
_GITHUB_RE  = re.compile(r'https?://github\.com/', re.IGNORECASE)
_DOI_URL_RE = re.compile(r'https?://(?:dx\.)?doi\.org/(10\.[\d.]+/.+)', re.IGNORECASE)
_DOI_BARE   = re.compile(r'^10\.\d{4,9}/')


def _classify_doi_or_url(raw: Optional[str]) -> tuple:
    """
    Given a raw string from a BibTeX doi= or url= field, return (doi, url) where
    exactly one is non-None:
      - A bare DOI  (10.xxxx/...)            -> (doi, None)
      - A doi.org URL                        -> (doi, None)   [bare DOI extracted]
      - A GitHub/software/general HTTP URL   -> (None, url)
      - None / empty                         -> (None, None)
    """
    if not raw:
        return None, None
    raw = raw.strip()
    # Bare DOI already
    if _DOI_BARE.match(raw):
        return raw, None
    # doi.org URL -> extract bare DOI
    m = _DOI_URL_RE.match(raw)
    if m:
        return m.group(1), None
    # Any other HTTP URL (GitHub, Zenodo landing page, arXiv, etc.)
    if raw.startswith('http'):
        return None, raw
    return None, None


def verify_url(url: str) -> tuple:
    """
    Verify a non-DOI URL (GitHub file, software repo, dataset page, etc.)
    by issuing an HTTP HEAD request.

    Returns (status_code, ok) where ok is True for 2xx responses.
    Follows redirects. Returns (-1, False) on network error.
    """
    try:
        r = requests.head(
            url, timeout=8, allow_redirects=True,
            headers={"User-Agent": HEADERS["User-Agent"]},
        )
        return r.status_code, (200 <= r.status_code < 300)
    except Exception:
        return -1, False

def _url_source_label(url: str) -> str:
    """Human-readable source label for a URL-verified reference."""
    if _GITHUB_RE.match(url):
        # e.g. github.com/Org/Repo/blob/main/file.py -> GitHub (Org/Repo)
        parts = url.split('/')
        if len(parts) >= 5:
            return f"GitHub ({parts[3]}/{parts[4]})"
        return "GitHub"
    if 'zenodo.org' in url:
        return "Zenodo"
    if 'arxiv.org' in url:
        return "arXiv"
    return "Web URL"




# =============================================================================
# RIS parser
# =============================================================================

def load_ris(text: str) -> list:
    """
    Parse a .ris file into a list of ParsedRef objects.
    Handles multi-value AU tags and optional trailing value on ER line.
    """
    refs = []
    current: dict = {}

    for line in text.splitlines():
        # RIS lines: "XX  - value" where XX is the tag
        m = re.match(r'^([A-Z][A-Z0-9])\s+-\s*(.*)', line.rstrip())
        if not m:
            continue
        tag, value = m.group(1), m.group(2).strip()

        if tag == 'ER':
            if current:
                refs.append(_ris_to_parsed_ref(current))
                current = {}
        elif tag in ('AU', 'A1', 'A2'):
            current.setdefault('_authors', []).append(value)
        else:
            current[tag] = value

    # File with no trailing ER
    if current:
        refs.append(_ris_to_parsed_ref(current))

    return refs


def _ris_to_parsed_ref(entry: dict) -> ParsedRef:
    title   = entry.get('TI') or entry.get('T1') or entry.get('CT', '')
    doi     = entry.get('DO') or entry.get('DI') or entry.get('UR', None)
    year_s  = entry.get('PY') or entry.get('Y1', '')
    authors_raw = entry.get('_authors', [])

    # RIS authors are typically "Last, First"
    authors = []
    for a in authors_raw:
        if ',' in a:
            last, first = a.split(',', 1)
            authors.append(f"{first.strip()} {last.strip()}")
        else:
            authors.append(a)

    year = None
    if year_s:
        ym = re.search(r'\d{4}', year_s)
        if ym:
            year = int(ym.group())

    if doi:
        doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi).strip()

    raw = f"[RIS] {title or '?'} ({year or '?'})"

    return ParsedRef(
        raw=raw,
        title=title or None,
        authors=authors,
        year=year,
        doi=doi or None,
        entry_type=entry.get('TY', 'unknown').lower(),
    )


# =============================================================================
# Plain-text parser  (one reference per line, heuristic)
# =============================================================================

_DOI_RE   = re.compile(r'10\.\d{4,9}/[^\s,\]"]+')
_YEAR_RE  = re.compile(r'\b((19|20)\d{2})\b')
_TITLE_RE = re.compile(
    r'"([^"]{10,})"'                     # ASCII double-quoted
    r'|\u201c([^\u201d]{10,})\u201d'     # curly double-quoted
    r'|\btitle[=:]\s*\{([^}]{10,})\}'   # BibTeX title={...}
)


def load_plaintext(text: str) -> list:
    """One reference per non-empty line."""
    return [parse_plaintext_ref(line.strip())
            for line in text.splitlines()
            if line.strip()]


def parse_plaintext_ref(raw: str) -> ParsedRef:
    """Heuristic parser for free-form reference strings."""
    ref = ParsedRef(raw=raw)

    doi_m = _DOI_RE.search(raw)
    if doi_m:
        ref.doi = doi_m.group(0).rstrip('.')

    years = _YEAR_RE.findall(raw)
    if years:
        ref.year = int(years[0][0])

    title_m = _TITLE_RE.search(raw)
    if title_m:
        ref.title = next(g for g in title_m.groups() if g).strip().rstrip(',')
    else:
        ref.title = _heuristic_title(raw)

    if ref.doi and ref.title and ref.title.lower().startswith("doi:"):
        ref.title = None

    ref.authors = _heuristic_authors(raw, ref.title, ref.year)
    return ref


def _heuristic_title(raw: str) -> Optional[str]:
    cleaned = re.sub(r'^\s*[\[\(]\w+[\]\)]\s*', '', raw).strip()
    parts = cleaned.split(',')
    if len(parts) >= 2:
        candidate = parts[1].strip().strip('"').strip('\u201c').strip('\u201d').strip()
        if 8 < len(candidate) < 250:
            return candidate
    if 10 < len(cleaned) < 300:
        return cleaned
    return None


def _heuristic_authors(raw: str, title: Optional[str], year: Optional[int]) -> list:
    anchor = raw
    if year:
        pos = anchor.find(str(year))
        if pos > 0:
            anchor = anchor[:pos].strip()
    if title and title in anchor:
        anchor = anchor[:anchor.find(title)].strip()
    names = re.split(r'\band\b|;|,\s+(?=[A-Z])', anchor)
    names = [n.strip().strip(',').strip() for n in names if 2 < len(n.strip()) < 60]
    return names[:10]


# =============================================================================
# Unified file loader
# =============================================================================

def load_references(path: str) -> tuple:
    """
    Load a reference file of any supported format.
    Returns (fmt, list_of_ParsedRef).
    """
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()

    fmt = detect_format(path, text)

    if fmt == 'bibtex':
        refs = load_bibtex(text)
    elif fmt == 'ris':
        refs = load_ris(text)
    else:
        refs = load_plaintext(text)

    return fmt, refs


# =============================================================================
# API clients
# =============================================================================

HEADERS = {
    "User-Agent": "CitationChecker/1.0 (research; contact: user@example.com)"
}
TIMEOUT = 10

# Per-host rate limiters (requests per second).
# Values are conservative to stay well within free-tier limits:
#   Semantic Scholar: ~1 req/s unauthenticated
#   CrossRef:        ~5 req/s (polite pool)
#   OpenAlex:        ~5 req/s
_RL_SS  = None   # initialised in _init_rate_limiters()
_RL_CR  = None
_RL_OA  = None
_RL_LOCK = threading.Lock()   # guards 429 back-off per host


class _RateLimiter:
    """
    Thread-safe sliding-window token bucket.
    Allows at most `rate` calls per `per` seconds across all threads.
    """
    def __init__(self, rate: float, per: float = 1.0):
        self.rate  = rate
        self.per   = per
        self._lock = threading.Lock()
        self._calls: deque = deque()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and self._calls[0] < now - self.per:
                    self._calls.popleft()
                if len(self._calls) < self.rate:
                    self._calls.append(now)
                    return
                wait = self.per - (now - self._calls[0])
            time.sleep(max(wait, 0.01))


def _init_rate_limiters():
    """Create rate limiters (called once from main, after workers=N is known)."""
    global _RL_SS, _RL_CR, _RL_OA
    _RL_SS = _RateLimiter(rate=1.0, per=1.0)   # Semantic Scholar: 1 req/s
    _RL_CR = _RateLimiter(rate=4.0, per=1.0)   # CrossRef:         4 req/s
    _RL_OA = _RateLimiter(rate=4.0, per=1.0)   # OpenAlex:         4 req/s


def _get(url: str, params: dict, limiter: Optional['_RateLimiter'] = None) -> Optional[dict]:
    """HTTP GET with optional rate-limiter and 429 back-off."""
    if limiter:
        limiter.acquire()
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 2 ** (attempt + 1)   # 2 s, 4 s, 8 s
                time.sleep(wait)
                continue
            return None
        except Exception:
            return None
    return None


def query_semantic_scholar(title: str) -> Optional[dict]:
    data = _get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {"query": title, "fields": "title,authors,year,externalIds,venue", "limit": 3},
        limiter=_RL_SS,
    )
    if data and data.get("data"):
        return data["data"][0]
    return None


def query_crossref(title: str, author: Optional[str] = None) -> Optional[dict]:
    params = {
        "query.title": title, "rows": 3,
        "select": "title,author,published,DOI,container-title",
    }
    if author:
        params["query.author"] = author
    data = _get("https://api.crossref.org/works", params, limiter=_RL_CR)
    if data and data.get("message", {}).get("items"):
        return data["message"]["items"][0]
    return None


def query_openalex(title: str) -> Optional[dict]:
    data = _get(
        "https://api.openalex.org/works",
        {"search": title, "per-page": 3,
         "select": "title,authorships,publication_year,doi,primary_location"},
        limiter=_RL_OA,
    )
    if data and data.get("results"):
        return data["results"][0]
    return None


def lookup_doi(doi: str) -> Optional[dict]:
    data = _get(f"https://api.crossref.org/works/{doi}", {}, limiter=_RL_CR)
    if data and data.get("message"):
        return data["message"]
    return None



def _extract_arxiv_id(eprint: Optional[str], prefix: Optional[str]) -> Optional[str]:
    """
    Normalise a BibTeX eprint= value into a bare arXiv ID.
    Handles:
      eprint={1402.7073}  archivePrefix={arXiv}  ->  "1402.7073"
      eprint={astro-ph/0507092}                  ->  "astro-ph/0507092"
      eprint={https://arxiv.org/abs/1234.5678}   ->  "1234.5678"
    Returns None if the entry is not an arXiv preprint.
    """
    if not eprint:
        return None
    eprint = eprint.strip()
    # Strip full URL if someone put the URL in eprint=
    eprint = re.sub(r'https?://arxiv\.org/abs/', '', eprint).strip()
    # Accept if prefix says arXiv, or if the ID looks like an arXiv ID
    _ARXIV_ID_RE = re.compile(r'^(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})$', re.IGNORECASE)
    is_arxiv_prefix = (prefix or '').lower() in ('arxiv', 'arxiv.org')
    if is_arxiv_prefix or _ARXIV_ID_RE.match(eprint):
        return eprint
    return None


def lookup_arxiv_ss(arxiv_id: str) -> Optional[dict]:
    """
    Look up a paper by arXiv ID using Semantic Scholar's direct paper endpoint.
    This is deterministic — no fuzzy matching, no ambiguity.
    Returns the same dict shape as query_semantic_scholar().

    Note: fields must be appended directly to the URL, not passed as a params
    dict — requests encodes commas as %2C which breaks SS field parsing.
    """
    fields = "title,authors,year,externalIds,venue"
    data = _get(
        f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields={fields}",
        {},   # no params — everything is already in the URL
        limiter=_RL_SS,
    )
    # Direct lookup returns the paper object directly, not wrapped in {"data": [...]}
    if data and data.get("title"):
        return data
    return None


# =============================================================================
# Matching / scoring
# =============================================================================

TITLE_THRESHOLD = 85


def _norm(s: str) -> str:
    return re.sub(r'[^\w\s]', '', s.lower()).strip()


def title_similarity(a: str, b: str) -> float:
    return fuzz.token_sort_ratio(_norm(a), _norm(b))


def author_similarity(parsed: list, found: list) -> float:
    if not parsed or not found:
        return 0.0
    pa = ' '.join(_norm(a) for a in parsed[:3])
    fa = ' '.join(_norm(a) for a in found[:3])
    return fuzz.token_set_ratio(pa, fa)


def _ss_authors(hit: dict) -> list:
    return [a.get("name", "") for a in hit.get("authors", [])]


def _cr_authors(hit: dict) -> list:
    return [f"{a.get('given','')} {a.get('family','')}".strip()
            for a in hit.get("author", [])]


def _oa_authors(hit: dict) -> list:
    return [a.get("author", {}).get("display_name", "")
            for a in hit.get("authorships", [])]


def _cr_year(hit: dict) -> Optional[int]:
    parts = hit.get("published", {}).get("date-parts", [[]])
    return int(parts[0][0]) if parts and parts[0] else None


# =============================================================================
# Main checker
# =============================================================================

def check_citation(ref: ParsedRef) -> CheckResult:
    result = CheckResult(raw=ref.raw, parsed=ref)

    # 0. Software / web reference: verify by HTTP HEAD, skip academic DBs entirely
    if ref.url and not ref.doi:
        http_code, ok = verify_url(ref.url)
        if ok:
            result.status      = "VERIFIED"
            result.matched_url = ref.url
            result.source      = _url_source_label(ref.url)
            result.notes.append(f"URL reachable (HTTP {http_code}).")
        elif http_code == 404:
            result.status = "NOT_FOUND"
            result.notes.append(f"URL returned HTTP 404 — file or page does not exist.")
        elif http_code == -1:
            result.status = "SUSPICIOUS"
            result.notes.append("URL could not be reached (network error). Cannot verify.")
        else:
            result.status = "SUSPICIOUS"
            result.notes.append(f"URL returned HTTP {http_code} — cannot confirm existence.")
        result.matched_url = ref.url
        return result

    if not ref.title and not ref.doi:
        result.status = "ERROR"
        result.notes.append("Could not parse a title or DOI from this reference.")
        return result

    # 1. DOI fast-path
    if ref.doi:
        hit = lookup_doi(ref.doi)
        if hit:
            found_title   = (hit.get("title") or [""])[0]
            found_authors = _cr_authors(hit)
            found_year    = _cr_year(hit)
            ts = title_similarity(ref.title or "", found_title) if ref.title else 100.0

            result.matched_title   = found_title
            result.matched_authors = found_authors
            result.matched_year    = found_year
            result.matched_doi     = ref.doi
            result.matched_venue   = (hit.get("container-title") or [""])[0]
            result.title_score     = ts
            result.author_score    = author_similarity(ref.authors, found_authors)
            result.year_match      = (ref.year == found_year) if ref.year and found_year else None
            result.source          = "CrossRef (DOI)"
            result.status          = "VERIFIED" if (ts >= TITLE_THRESHOLD or not ref.title) else "SUSPICIOUS"
            if result.status == "SUSPICIOUS":
                result.notes.append(f"DOI resolves but title mismatch (score={ts:.0f}).")
            return result
        else:
            result.notes.append("DOI not found in CrossRef.")

    title_query  = ref.title or ""
    first_author = ref.authors[0] if ref.authors else None

    # 1b. arXiv ID fast-path (deterministic — no fuzzy matching)
    if ref.arxiv_id:
        hit = lookup_arxiv_ss(ref.arxiv_id)
        if hit:
            found_title   = hit.get("title", "")
            found_authors = _ss_authors(hit)
            found_year    = hit.get("year")
            ts = title_similarity(title_query, found_title) if title_query else 100.0

            result.matched_title   = found_title
            result.matched_authors = found_authors
            result.matched_year    = found_year
            result.matched_doi     = (hit.get("externalIds") or {}).get("DOI")
            result.matched_venue   = hit.get("venue", "")
            result.title_score     = ts
            result.author_score    = author_similarity(ref.authors, found_authors)
            result.year_match      = (ref.year == found_year) if ref.year and found_year else None
            result.source          = f"Semantic Scholar (arXiv:{ref.arxiv_id})"
            result.status          = _correctness_status(result)
            # arXiv ID is an exact identifier — it already confirms the right paper.
            # Titles legitimately differ: abbreviated subtitles, superscripts,
            # journal vs preprint wording. Accept score >= 70 (vs 85 for title
            # search) when found via arXiv ID, but still flag author mismatches.
            ARXIV_TITLE_THRESHOLD = 70
            if result.status == "SUSPICIOUS" and ts < TITLE_THRESHOLD and ts >= ARXIV_TITLE_THRESHOLD and result.author_score >= 50:
                result.notes.append(
                    f"Title differs from arXiv version (score={ts:.0f})"
                    f" — likely abbreviated subtitle or journal vs preprint wording."
                )
                result.status = "VERIFIED"
            return result
        else:
            result.notes.append(f"arXiv:{ref.arxiv_id} not found in Semantic Scholar.")

    # 2. Semantic Scholar title search
    hit_ss = query_semantic_scholar(title_query)
    if hit_ss:
        found_title = hit_ss.get("title", "")
        ts = title_similarity(title_query, found_title)
        if ts >= TITLE_THRESHOLD:
            result.matched_title   = found_title
            result.matched_authors = _ss_authors(hit_ss)
            result.matched_year    = hit_ss.get("year")
            result.matched_doi     = (hit_ss.get("externalIds") or {}).get("DOI")
            result.matched_venue   = hit_ss.get("venue", "")
            result.title_score     = ts
            result.author_score    = author_similarity(ref.authors, result.matched_authors)
            result.year_match      = (ref.year == result.matched_year) if ref.year and result.matched_year else None
            result.source          = "Semantic Scholar"
            result.status          = _correctness_status(result)
            return result

    # 3. CrossRef
    hit_cr = query_crossref(title_query, first_author)
    if hit_cr:
        found_title = (hit_cr.get("title") or [""])[0]
        ts = title_similarity(title_query, found_title)
        if ts >= TITLE_THRESHOLD:
            result.matched_title   = found_title
            result.matched_authors = _cr_authors(hit_cr)
            result.matched_year    = _cr_year(hit_cr)
            result.matched_doi     = hit_cr.get("DOI")
            result.matched_venue   = (hit_cr.get("container-title") or [""])[0]
            result.title_score     = ts
            result.author_score    = author_similarity(ref.authors, result.matched_authors)
            result.year_match      = (ref.year == result.matched_year) if ref.year and result.matched_year else None
            result.source          = "CrossRef"
            result.status          = _correctness_status(result)
            return result

    # 4. OpenAlex
    hit_oa = query_openalex(title_query)
    if hit_oa:
        found_title = hit_oa.get("title", "") or ""
        ts = title_similarity(title_query, found_title)
        if ts >= TITLE_THRESHOLD:
            result.matched_title   = found_title
            result.matched_authors = _oa_authors(hit_oa)
            result.matched_year    = hit_oa.get("publication_year")
            result.matched_doi     = hit_oa.get("doi", "")
            result.title_score     = ts
            result.author_score    = author_similarity(ref.authors, result.matched_authors)
            result.year_match      = (ref.year == result.matched_year) if ref.year and result.matched_year else None
            result.source          = "OpenAlex"
            result.status          = _correctness_status(result)
            return result

    # 5. Not found
    result.status = "NOT_FOUND"
    result.notes.append("Title not matched in Semantic Scholar, CrossRef, or OpenAlex.")
    return result


# Year-difference tolerance per entry type.
# Books: editions and reprints routinely differ by 1-3 years.
# Articles: arXiv preprint vs final publication, or updated living reviews.
_YEAR_TOL = {
    'book':          3,
    'inbook':        3,
    'incollection':  3,
    'article':       3,
    'inproceedings': 2,
    'conference':    2,
}
_YEAR_TOL_DEFAULT = 1   # misc, techreport, etc.


def _year_tolerance(entry_type: Optional[str]) -> int:
    return _YEAR_TOL.get((entry_type or '').lower(), _YEAR_TOL_DEFAULT)


def _correctness_status(r: CheckResult) -> str:
    issues = []
    if r.author_score > 0 and r.author_score < 50:
        issues.append(f"Author mismatch (score={r.author_score:.0f})")

    if r.year_match is False and r.parsed.year and r.matched_year:
        diff = abs(r.parsed.year - r.matched_year)
        tol  = _year_tolerance(r.parsed.entry_type)
        if diff <= tol:
            # Within tolerance: likely a different edition or arXiv/print gap
            label = "edition" if r.parsed.entry_type in ('book','inbook','incollection') else "version"
            r.notes.append(
                f"Year differs by {diff} (claimed {r.parsed.year}, found {r.matched_year}) "
                f"— probably a different {label} or preprint/publication gap."
            )
            # Do NOT add to issues; treat as verified
        else:
            issues.append(f"Year mismatch (claimed {r.parsed.year}, found {r.matched_year})")

    if issues:
        r.notes.extend(issues)
        return "SUSPICIOUS"
    return "VERIFIED"


# =============================================================================
# Batch runner  (parallel)
# =============================================================================

DEFAULT_WORKERS = 5   # good default: stays under all three APIs' rate limits

def check_references(refs: list, verbose: bool = True, workers: int = DEFAULT_WORKERS) -> list:
    """
    Check all references in parallel using a thread pool.

    workers: number of concurrent threads.
      - Each thread makes HTTP calls, so the GIL is released and true
        parallelism is achieved.
      - The rate limiters ensure we never exceed per-host request budgets
        regardless of how many workers are running.
      - Recommended range: 3–8. Higher values don't help much because
        Semantic Scholar caps unauthenticated requests at ~1 req/s total,
        and that's usually the bottleneck.

    Results are returned in the same order as the input list.
    """
    results = [None] * len(refs)          # pre-allocate to preserve order
    lock    = threading.Lock()            # guards tqdm updates

    bar = tqdm(
        total=len(refs),
        desc=f"Checking citations (workers={workers})",
        disable=not verbose,
        unit="ref",
    )

    def _task(index: int, ref: ParsedRef):
        result = check_citation(ref)
        results[index] = result
        with lock:
            bar.update(1)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, i, ref): i for i, ref in enumerate(refs)}
        # Drain futures so exceptions propagate rather than being silently lost
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                idx = futures[future]
                results[idx] = CheckResult(
                    raw=refs[idx].raw, parsed=refs[idx],
                    status="ERROR",
                    notes=[f"Unexpected error: {exc}"],
                )

    bar.close()
    return results


# =============================================================================
# Report printer
# =============================================================================

STATUS_COLOR = {
    "VERIFIED":    Fore.GREEN,
    "SUSPICIOUS":  Fore.YELLOW,
    "NOT_FOUND":   Fore.RED,
    "ERROR":       Fore.MAGENTA,
    "NOT_CHECKED": Fore.WHITE,
}


def print_report(results: list) -> None:
    counts = {"VERIFIED": 0, "SUSPICIOUS": 0, "NOT_FOUND": 0, "ERROR": 0}
    print()
    print("=" * 72)
    print("  CITATION CHECKER REPORT")
    print("=" * 72)

    for i, r in enumerate(results, 1):
        color = STATUS_COLOR.get(r.status, "")
        reset = Style.RESET_ALL if HAS_COLOR else ""
        badge = f"{color}[{r.status}]{reset}"

        # Use cite key as label if available, else index
        label = r.parsed.entry_key or str(i)
        print(f"\n{i:>4}. {badge}  {label}")

        if r.parsed.title:
            print(f"      Title   : {r.parsed.title[:100]}")
        if r.parsed.year:
            print(f"      Year    : {r.parsed.year}")
        if r.parsed.authors:
            print(f"      Authors : {', '.join(r.parsed.authors[:3])}")

        if r.matched_url and not r.matched_title:
            # Software / web reference: show URL + source label
            print(f"      URL     : {r.matched_url}")
            if r.source:
                print(f"      Source  : {r.source}")
        if r.matched_title:
            print(f"      Found   : {r.matched_title[:100]}  [{r.source}]")
            year_sym = "Y" if r.year_match else ("N" if r.year_match is False else "?")
            print(f"      Scores  : title={r.title_score:.0f}  authors={r.author_score:.0f}  year={year_sym}")
            if r.matched_doi:
                print(f"      DOI     : {r.matched_doi}")
            if r.matched_venue:
                print(f"      Venue   : {r.matched_venue}")

        for note in r.notes:
            print(f"      !  {note}")

        counts[r.status] = counts.get(r.status, 0) + 1

    total = len(results)
    print()
    print("-" * 72)
    print(f"  SUMMARY  ({total} references)")
    print("-" * 72)
    for status in ["VERIFIED", "SUSPICIOUS", "NOT_FOUND", "ERROR"]:
        n = counts.get(status, 0)
        if n == 0:
            continue
        pct = 100 * n / total if total else 0
        color = STATUS_COLOR.get(status, "")
        reset = Style.RESET_ALL if HAS_COLOR else ""
        bar = "#" * int(pct / 2)
        print(f"  {color}{status:<12}{reset}  {n:>4} ({pct:5.1f}%)  {bar}")
    print("=" * 72)
    print()


# =============================================================================
# Demo references
# =============================================================================

DEMO_REFS_RAW = [
    '[1] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, "Detecting hallucinations in large language models using semantic entropy," Nature, vol. 630, pp. 625-630, 2024.',
    '[2] S. Wuchty, B. F. Jones, and B. Uzzi, "The increasing dominance of teams in production of knowledge," Science, vol. 316, pp. 1036-1039, 2007.',
    '[3] S. Noy and W. Zhang, "Experimental evidence on the productivity effects of generative artificial intelligence," Science, vol. 381, pp. 187-192, 2023.',
    '[4] I. Shumailov et al., "AI models collapse when trained on recursively generated data," Nature, vol. 631, pp. 755-759, 2022.',
    '[5] J. Smith and A. Jones, "Quantum entanglement effects on deep learning convergence rates in transformer architectures," Neural Computation, vol. 45, pp. 112-145, 2023.',
    '[6] R. Chen, M. Lee, "Emergent consciousness in large language models: a topological framework," Nature Machine Intelligence, 2024.',
    '[7] doi:10.1038/s41586-024-07421-0',
]

DEMO_BIB = r"""
@article{Farquhar2024,
  author  = {Sebastian Farquhar and Jannik Kossen and Lorenz Kuhn and Yarin Gal},
  title   = {Detecting hallucinations in large language models using semantic entropy},
  journal = {Nature},
  volume  = {630},
  pages   = {625--630},
  year    = {2024},
  doi     = {10.1038/s41586-024-07421-0}
}
@article{Wuchty2007,
  author  = {Stefan Wuchty and Benjamin F. Jones and Brian Uzzi},
  title   = {The increasing dominance of teams in production of knowledge},
  journal = {Science},
  volume  = {316},
  pages   = {1036--1039},
  year    = {2007}
}
@article{Shumailov2022wrong,
  author  = {Ilia Shumailov and Zakhar Shumaylov},
  title   = {AI models collapse when trained on recursively generated data},
  journal = {Nature},
  year    = {2022}
}
@article{hallucinated2024,
  author  = {John Q. Smith and Alice Jones},
  title   = {Quantum entanglement effects on deep learning convergence rates in transformer architectures},
  journal = {Neural Computation},
  year    = {2023}
}
"""


def _mock_results(refs: list) -> list:
    """Simulated results for offline demo."""
    mock_data = [
        dict(status="VERIFIED",  title_score=97,  author_score=82, year_match=True,
             matched_title="Detecting hallucinations in large language models using semantic entropy",
             matched_authors=["Sebastian Farquhar","Jannik Kossen","Lorenz Kuhn","Yarin Gal"],
             matched_year=2024, matched_doi="10.1038/s41586-024-07421-0",
             matched_venue="Nature", source="CrossRef (DOI)"),
        dict(status="VERIFIED",  title_score=99,  author_score=91, year_match=True,
             matched_title="The increasing dominance of teams in production of knowledge",
             matched_authors=["Stefan Wuchty","Benjamin F. Jones","Brian Uzzi"],
             matched_year=2007, matched_doi="10.1126/science.1136099",
             matched_venue="Science", source="Semantic Scholar"),
        dict(status="SUSPICIOUS", title_score=98, author_score=75, year_match=False,
             matched_title="AI models collapse when trained on recursively generated data",
             matched_authors=["Ilia Shumailov","Zakhar Shumaylov","Yiren Zhao"],
             matched_year=2024, matched_doi="10.1038/s41586-024-07566-y",
             matched_venue="Nature", source="Semantic Scholar",
             notes=["Year mismatch (claimed 2022, found 2024)"]),
        dict(status="NOT_FOUND", notes=["Title not matched in Semantic Scholar, CrossRef, or OpenAlex."]),
    ]
    results = []
    for ref, data in zip(refs, mock_data):
        notes = data.pop('notes', [])
        r = CheckResult(raw=ref.raw, parsed=ref, **data)
        r.notes = notes
        results.append(r)
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Check academic citations for existence and correctness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported input formats (auto-detected):
  .bib    BibTeX
  .ris    RIS
  .txt    Plain text, one reference per line

Examples:
  python citation_checker.py myrefs.bib
  python citation_checker.py myrefs.bib --output audit.json
  python citation_checker.py --mock
  python citation_checker.py --demo-bib
""")
    parser.add_argument("input_file", nargs="?",
                        help="Reference file (.bib, .ris, or .txt)")
    parser.add_argument("--demo",     action="store_true",
                        help="Live API demo on plain-text references")
    parser.add_argument("--demo-bib", action="store_true",
                        help="Live API demo on BibTeX references")
    parser.add_argument("--mock",     action="store_true",
                        help="Offline demo using BibTeX sample (no API calls)")
    parser.add_argument("--output",   metavar="FILE",
                        help="Save JSON results to this file")
    parser.add_argument("--workers",  type=int, default=DEFAULT_WORKERS, metavar="N",
                        help=f"Parallel threads (default: {DEFAULT_WORKERS}; range 1-10)")
    parser.add_argument("--quiet",    action="store_true",
                        help="Suppress progress bar")
    args = parser.parse_args()

    if args.mock:
        refs = load_bibtex(DEMO_BIB)
        print(f"(Mock mode -- no API calls)\n")
        print(f"Loaded {len(refs)} BibTeX entries from built-in demo.\n")
        results = _mock_results(refs)
        print_report(results)
        return

    if args.demo_bib:
        refs = load_bibtex(DEMO_BIB)
        fmt = 'bibtex'
    elif args.demo:
        refs = [parse_plaintext_ref(r) for r in DEMO_REFS_RAW]
        fmt = 'plaintext'
    elif args.input_file:
        fmt, refs = load_references(args.input_file)
        print(f"Detected format : {fmt.upper()}")
    else:
        parser.print_help()
        sys.exit(0)

    print(f"Loaded {len(refs)} references.")
    print(f"Checking against Semantic Scholar, CrossRef, OpenAlex...\n")
    _init_rate_limiters()
    results = check_references(refs, verbose=not args.quiet, workers=args.workers)
    print_report(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        print(f"JSON results saved to {args.output}")


if __name__ == "__main__":
    main()
