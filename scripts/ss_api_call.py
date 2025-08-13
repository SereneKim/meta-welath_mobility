import os, time, random, requests
import pandas as pd
import logging
import requests
from typing import Iterable,Optional, List, Dict, Any
import re
import unicodedata

SS_BASE   = "https://api.semanticscholar.org/graph/v1"
SS_APIKEY = os.getenv("SS_API_KEY")
HEADERS   = {"x-api-key": SS_APIKEY} if SS_APIKEY else {}

NOT_FOUND_PATH = "ss_not_found_dois.txt"  # persistent cache on disk

# ---------- utilities ----------
def normalize_doi(s):
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip().lower()
    for pref in ("https://doi.org/", "http://doi.org/", "doi:"):
        if s.startswith(pref):
            s = s[len(pref):]
    return s

def _respect_rate_limit(last_call, rps=1.0):
    # keep >= 1 rps for intro tier
    min_interval = max(1.0 / rps, 1.0)
    delay = min_interval - (time.monotonic() - last_call)
    if delay > 0:
        time.sleep(delay)

def _get(sess, url, params, last_call, rps=1.0):
    while True:
        _respect_rate_limit(last_call, rps=rps)
        r = sess.get(url, headers=HEADERS, params=params, timeout=30)
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 1.5))
            time.sleep(wait + random.uniform(0, 0.5))
            last_call = time.monotonic()
            continue
        # 10 MB response cap
        if r.status_code == 400 and "Response would exceed maximum size" in (r.text or ""):
            raise RuntimeError("Response too large; reduce page size.")
        r.raise_for_status()
        return r, time.monotonic()

# def _load_not_found(path=NOT_FOUND_PATH):
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             return {d for d in (normalize_doi(x) for x in f) if d}
#     except FileNotFoundError:
#         return set()

# def _save_not_found(not_found, path=NOT_FOUND_PATH):
#     tmp = path + ".tmp"
#     with open(tmp, "w", encoding="utf-8") as f:
#         for d in sorted(not_found):
#             f.write(d + "\n")
#     os.replace(tmp, path)

def _load_not_found(path=NOT_FOUND_PATH):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        return set()

def _save_not_found(ids, path=NOT_FOUND_PATH):
    with open(path, "w", encoding="utf-8") as f:
        for pid in sorted(ids):
            f.write(pid + "\n")
 
    
# ---------- search for paper by semantic scholar ID ----------
def fetch_semantic_scholar_batch(
    identifiers: Iterable[str],
    fields: List[str] = ["paperId", "title", "externalIds"],
    batch_size: int = 500,
    timeout: int = 60,
    sleep_between: float = 0.8,
) -> List[Dict[str, Any]]:
    """
    Look up papers by DOI (or other supported ID) using the batch endpoint.
    Returns a list aligned with your input order: each item is either a dict (paper) or None.
    """
    ids = [normalize_doi(x) for x in identifiers if x]
    if not ids:
        return []

    headers = HEADERS
    params = {"fields": ",".join(fields)}
    url = f"{SS_BASE}/paper/batch"

    out: List[Dict[str, Any] | None] = []
    session = requests.Session()

    for i in range(0, len(ids), batch_size):
        chunk = ids[i:i + batch_size]
        # POST (not GET) and send the list directly under "ids"
        r = session.post(url, headers=headers, params=params, json={"ids": chunk}, timeout=timeout)
        if r.status_code == 429:  # gentle backoff once
            time.sleep(2.5)
            r = session.post(url, headers=headers, params=params, json={"ids": chunk}, timeout=timeout)
        r.raise_for_status()
        out.extend(r.json())
        if i + batch_size < len(ids):
            time.sleep(sleep_between)

    return out

# Convenience: return just a mapping from DOI -> paperId (None if not found)
def dois_to_paper_ids(dois: Iterable[str], **kwargs) -> Dict[str, str | None]:
    results = fetch_semantic_scholar_batch(dois, **kwargs)
    mapping: Dict[str, str | None] = {}
    for orig, rec in zip(dois, results):
        pid = rec.get("paperId") if isinstance(rec, dict) else None
        mapping[str(orig).lower()] = pid
    return mapping



# ---------- fetch paper meta by Title ----------
import re, time, unicodedata, random, requests, html
from difflib import SequenceMatcher
from typing import Iterable, Optional, Dict, Any, List

def _req_with_backoff(
    sess: requests.Session,
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    params: Dict[str, Any] | None = None,
    json: Dict[str, Any] | None = None,
    timeout: int = 30,
    max_retries: int = 6,
    base_delay: float = 1.0,
) -> requests.Response:
    """HTTP request with Retry-After handling and exponential backoff + jitter."""
    last = None
    for attempt in range(max_retries):
        r = sess.request(method, url, headers=headers, params=params, json=json, timeout=timeout)
        last = r
        # Back off on 429 or transient 5xx
        if r.status_code == 429 or 500 <= r.status_code < 600:
            ra = r.headers.get("Retry-After")
            try:
                delay = float(ra) if ra is not None else base_delay * (2 ** attempt)
            except ValueError:
                delay = base_delay * (2 ** attempt)
            time.sleep(delay + random.uniform(0, 0.5))
            continue
        # Let caller handle 404/400 (used by /search/match)
        if r.status_code in (200, 400, 404):
            return r
        r.raise_for_status()
        return r
    # Exhausted retries
    if last is not None:
        last.raise_for_status()
    raise RuntimeError("Request failed without response")


TAG_RE = re.compile(r"<[^>]*>")

def _strip_tags_and_entities(s: str) -> str:
    # Remove HTML tags, decode entities, collapse spaces
    s = TAG_RE.sub("", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()

def _norm_title(s: str) -> str:
    if not s:
        return ""
    # strip tags/entities too, because some titles come with markup
    s = _strip_tags_and_entities(s)
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _best_normalized_match(target: str, candidates: List[Dict[str, Any]], threshold: float = 0.92):
    """Pick the closest normalized title when no exact-normalized match exists."""
    t = _norm_title(target)
    best, best_score = None, 0.0
    for rec in candidates:
        cand = _norm_title(rec.get("title", ""))
        score = SequenceMatcher(None, t, cand).ratio()
        if score > best_score:
            best, best_score = rec, score
    return best if best_score >= threshold else None

def get_paper_by_title(
    title: str,
    year: Optional[int] = None,
    fields: List[str] = ["paperId", "title", "year"],  # keep payload lean
    timeout: int = 30,
    sess: Optional[requests.Session] = None,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return the best matching paper dict (or None) for a given title."""
    if not title:
        return None

    # Clean the query we actually send
    query = _strip_tags_and_entities(title)
    # print(f"Searching for title: {query!r}") #debugging
    params: Dict[str, Any] = {"query": query, "fields": ",".join(fields)}
    if year is not None:
        params["year"] = year

    _sess = sess or requests.Session()

    # 1) Try single-best title match
    url_match = f"{SS_BASE}/paper/search/match"
    r = _req_with_backoff(_sess, "GET", url_match, headers=HEADERS, params=params, timeout=timeout)
    print("r result:",r.status_code, r.text) # debugging
    if r.status_code == 200:
        return r.json()

    # 2) General search (with year if provided)
    url_search = f"{SS_BASE}/paper/search"
    r2 = _req_with_backoff(_sess, "GET", url_search, headers=HEADERS,
                           params={**params, "limit": 20}, timeout=timeout)
    print("r2 result:", r2.status_code, r2.text)  # debugging
    
    if r2.status_code != 200:
        r2.raise_for_status()

    items = (r2.json() or {}).get("data", []) or []

    # If nothing came back and we had a year filter, retry without year
    if not items and year is not None:
        params_no_year = {"query": query, "fields": ",".join(fields), "limit": 20}
        r3 = _req_with_backoff(_sess, "GET", url_search, headers=HEADERS,
                               params=params_no_year, timeout=timeout)
        if r3.status_code == 200:
            items = (r3.json() or {}).get("data", []) or []

    if not items:
        if debug:
            print(f"[MISS] No results for: {title!r} (query={query!r}, year={year})")
        return None

    # Prefer exact-normalized match
    target = _norm_title(title)
    for rec in items:
        if _norm_title(rec.get("title", "")) == target:
            return rec

    # Fallback: best fuzzy-normalized match
    approx = _best_normalized_match(title, items)
    if approx is not None:
        print(f"[FALLBACK] Best match for {title!r} is {approx.get('title', '')!r} (score={approx.get('score', 0.0)})")
        return approx

    if debug:
        print(f"[MISS] Only non-exact low-similarity results for: {title!r}")
    return items[0]  # last resort: top-ranked result

def titles_to_paper_ids(
    titles: Iterable[str],
    year_hints: Optional[Dict[str, int]] = None,
    sleep_between: float = 0.8,
    debug: bool = False,
) -> Dict[str, Optional[str]]:
    """Map each title -> paperId (None if not found). Dedupe titles to cut API calls."""
    titles = [t for t in titles if isinstance(t, str) and t.strip()]
    unique_titles = list(dict.fromkeys(titles))

    out_unique: Dict[str, Optional[str]] = {}
    sess = requests.Session()

    for t in unique_titles:
        try:
            rec = get_paper_by_title(t, year=(year_hints or {}).get(t), sess=sess, debug=debug)
            print(f"This is rec: {rec}")  # Debugging
            if isinstance(rec, dict) and "data" in rec:
                out_unique[t] = rec["data"][0]["paperId"]
            elif isinstance(rec, dict) and "paperId" in rec:
                out_unique[t] = rec["paperId"]
            else:
                out_unique[t] = None
            print(f"output: {out_unique[t]}")  # Debugging
        except requests.HTTPError as e:
            if debug:
                print(f"[ERROR] {t!r}: {e}")
        if sleep_between:
            time.sleep(sleep_between)
    return {t: out_unique.get(t) for t in titles}


# # ---------- paper meta (for influentialCitationCount) ----------
# def fetch_influential_counts(ids, rps=1.0, session=None, not_found=None):
#     """
#     Return dict {paperId: influentialCitationCount} (missing -> None)
#     Records 400/404 DOIs into not_found (if provided) and skips them next time.
#     """
#     sess = session or requests.Session()
#     out = {}
#     last = 0.0
#     nf = not_found if not_found is not None else set()

#     for id in ids:
#         url = f"{SS_BASE}/paper/{id}"
#         params = {"fields": "influentialCitationCount"}
#         try:
#             r, last = _get(sess, url, params, last, rps=rps)
#             j = r.json() or {}
#             out[id] = j.get("influentialCitationCount", None)
#         except requests.HTTPError as e:
#             status = getattr(e.response, "status_code", None)
#             if status in (400, 404):
#                 # invalid / unknown id at S2
#                 nf.add(id)
#                 out[id] = None
#                 logging.info(f"S2 not found (meta): {id}")
#                 continue
#             raise
#     return out

# # ---------- references edge fetch (citing -> cited) ----------
# def fetch_reference_edges_for_citing(citing_id, include_contexts=False, rps=1.0, session=None, not_found=None):
#     """
#     For **one** citing paperId, fetch ALL reference edges and return list of dicts:
#       {cited_paperId, intents, edge_isInfluential, contextsWithIntent?}
#     If citing paperId is 400/404, record in not_found (if provided) and return [].
#     """
#     sess = session or requests.Session()
#     nf = not_found if not_found is not None else set()
    
#     if not citing_id or citing_id in nf:
#         return []

#     url = f"{SS_BASE}/paper/{citing_id}/references"
#     fields = ["paperId", "intents", "isInfluential"]
#     if include_contexts:
#         fields.append("contexts") # check if this is correct, as the original code had "contextsWithIntent"
#     params = {"fields": ",".join(fields), "offset": 0, "limit": 1000}

#     edges = []
#     last = 0.0
#     try:
#         while True:
#             r, last = _get(sess, url, params, last, rps=rps)
#             batch = r.json()
#             if not isinstance(batch, dict):
#                 logging.warning(f"Unexpected /references payload for {citing_id}: {batch}")
#                 break
#             raw = batch.get("data")
#             if not raw:  # None or empty list -> no edges
#                 break
#             for e in raw:
#                 cited_id = e.get("citedPaper", {}).get("paperId", None)
#                 edges.append({
#                     "cited_paperId": cited_id,
#                     "intents": e.get("intents", []) or [],
#                     "edge_isInfluential": e.get("isInfluential", False),
#                     "contexts": e.get("contexts") if include_contexts else None,
#                 })
#             nxt = batch.get("next")
#             if nxt is None:
#                 break
#             params["offset"] = nxt
#     except requests.HTTPError as e:
#         status = getattr(e.response, "status_code", None)
#         if status in (400, 404):
#             nf.add(citing_id)
#             logging.info(f"S2 not found (refs): {citing_id}")
#             return []
#         raise
#     return [e for e in edges if e["cited_paperId"]]

# # ---------- main: enrich your cit dataframe ----------
# def enrich_with_intents_and_influential(cit_df, include_contexts=False, rps=1.0, not_found_path=NOT_FOUND_PATH):
#     """
#     cit_df must have columns: citingPaperId, citedPaperId
#     Returns (df, info) where:
#       - df includes: intents, edge_isInfluential, contexts?,
#                      citing_influentialCitationCount, cited_influentialCitationCount
#       - info: dict with keys:
#           'not_found_all', 'not_found_new', 'errors'
#     """
#     df = cit_df.copy()

#     # Load persistent not-found cache
#     not_found = _load_not_found(not_found_path) if not_found_path else set()
#     baseline_nf_size = len(not_found)

#     # 1) Fetch all references once per unique citing DOI (skipping known-missing)
#     sess = requests.Session()
#     ref_map = {}
#     errors = []
#     unique_citing = [d for d in df["citingPaperId"].dropna().unique()]
#     for citing in unique_citing:
#         if citing in not_found:
#             ref_map[citing] = {}
#             continue
#         try:
#             edges = fetch_reference_edges_for_citing(
#                 citing, include_contexts=include_contexts, rps=rps, session=sess, not_found=not_found
#             )
#         except RuntimeError:
#             # response too big -> refetch with smaller pages
#             edges = []
#             url = f"{SS_BASE}/paper/{citing}/references"
#             fields = ["paperId", "intents", "isInfluential"]
#             if include_contexts:
#                 fields.append("contexts") # check if this is correct, as the original code had "contextsWithIntent"
#             params = {"fields": ",".join(fields), "offset": 0, "limit": 200}
#             last = 0.0
#             try:
#                 while True:
#                     r, last = _get(sess, url, params, last, rps=rps)
#                     batch = r.json() or {}
#                     raw = batch.get("data") or []
#                     for e in raw:
#                         edges.append({
#                             "cited_paperId": e.get("citedPaper", {}).get("paperId", None),
#                             "intents": e.get("intents", []) or [],
#                             "edge_isInfluential": e.get("isInfluential", False),
#                             "contexts": e.get("contexts") if include_contexts else None,
#                         })
#                     nxt = batch.get("next")
#                     if nxt is None: break
#                     params["offset"] = nxt
#             except requests.HTTPError as e:
#                 status = getattr(e.response, "status_code", None)
#                 if status in (400, 404):
#                     not_found.add(citing)
#                     edges = []
#                 else:
#                     errors.append({"citing": citing, "error": str(e)})
#                     edges = []
#         except Exception as e:
#             errors.append({"citing": citing, "error": str(e)})
#             edges = []

#         ref_map[citing] = {e["cited_paperId"]: e for e in edges if e["cited_paperId"]}

#     # 2) Map edge info to each row
#     def _lookup_edge(row):
#         m = ref_map.get(row["citingPaperId"], {})
#         e = m.get(row["citedPaperId"], None)
#         print(f"Processing citing={row['citingPaperId']} cited={row['citedPaperId']} -> {e}")  # Debugging
#         if e:
#             print(f"hit: citing={row['citingPaperId']} cited={row['citedPaperId']}") # Debugging
#             return pd.Series({
#                 "intents": e.get("intents", []),
#                 "edge_isInfluential": e.get("edge_isInfluential", False),
#                 "contexts": e.get("contexts"),
#             })
#         else:
#             return pd.Series({"intents": [], "edge_isInfluential": None, "contexts": None})
        

#     edge_cols = df.apply(_lookup_edge, axis=1)
#     df = pd.concat([df, edge_cols], axis=1)

#     # 3) Add paper-level influentialCitationCount for both sides (skip known-missing)
#     all_ids = pd.unique(pd.concat([df["citingPaperId"], df["citedPaperId"]], ignore_index=True).dropna())
#     want = [d for d in all_ids if d not in not_found]
#     infl_map = fetch_influential_counts(want, rps=rps, session=sess, not_found=not_found)
#     # also ensure missing ones map to None
#     for d in all_ids:
#         if d not in infl_map:
#             infl_map[d] = None
#     df["citing_influentialCitationCount"] = df["citingPaperId"].map(infl_map)
#     df["cited_influentialCitationCount"]  = df["citedPaperId"].map(infl_map)

#     # Persist not-found set if it changed
#     if not_found_path and len(not_found) > baseline_nf_size:
#         _save_not_found(not_found, not_found_path)

#     info = {
#         "not_found_all": sorted(not_found),
#         "not_found_new": sorted(list(not_found))[baseline_nf_size:],
#         "errors": errors,
#     }
#     return df, info, ref_map

# ---------- paper meta (for influentialCitationCount) ----------
def fetch_influential_counts(ids, rps=1.0, session=None, not_found=None):
    """
    Return dict {paperId: influentialCitationCount} (missing -> None)
    Records 400/404 DOIs into not_found (if provided) and skips them next time.
    """
    sess = session or requests.Session()
    out = {}
    last = 0.0
    nf = not_found if not_found is not None else set()

    for id in ids:
        url = f"{SS_BASE}/paper/{id}"
        params = {"fields": "influentialCitationCount"}
        try:
            r, last = _get(sess, url, params, last, rps=rps)
            j = r.json() or {}
            out[id] = j.get("influentialCitationCount", None)
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status in (400, 404):
                # invalid / unknown id at S2
                nf.add(id)
                out[id] = None
                logging.info(f"S2 not found (meta): {id}")
                continue
            raise
    return out

# ---------- citations edge fetch (citing -> cited) ----------
def fetch_citation_edges_for_cited(cited_id, include_contexts=False, rps=1.0, session=None, not_found=None, stats=None):
    """
    CHANGE TO "CITATIONS" NOT "REFERENCES"
    For **one** cited paperId, fetch ALL CITATIONS edges and return list of dicts:
      {citing_paperId, intents, edge_isInfluential, contextsWithIntent?}
    If cited paperId is 400/404, record in not_found (if provided) and return [].
    """
    sess = session or requests.Session()
    nf = not_found if not_found is not None else set()
    
    if not cited_id or cited_id in nf:
        return []

    url = f"{SS_BASE}/paper/{cited_id}/citations"
    fields = ["intents", "isInfluential", "paperId"]
    if include_contexts:
        fields.append("contexts") # check if this is correct, as the original code had "contextsWithIntent"
    params = {"fields": ",".join(fields), "offset": 0, "limit": 1000}

    edges = []
    last = 0.0
    
    # debugging stats
    total_cits = 0
    with_pid = 0
    empty_intents = 0
    influ_true = 0
    
    try:
        while True:
            r, last = _get(sess, url, params, last, rps=rps)
            batch = r.json()
            if not isinstance(batch, dict):
                logging.warning(f"Unexpected /citations payload for {cited_id}: {batch}")
                break
            raw = batch.get("data")
            if not raw:  # None or empty list -> no edges
                break
            for e in raw:
                # debugging stats-----------------
                total_cits += 1
                citing_id = e.get("citingPaper", {}).get("paperId", None)
                intents = e.get("intents") or []
                is_influ = bool(e.get("isInfluential", False))
                empty_intents += (len(intents) == 0)
                influ_true += (is_influ is True)
                # ---------------------------------
                if citing_id:
                    with_pid += 1
                    edges.append({
                        "citing_paperId": citing_id,
                        "intents": e.get("intents", []) or [],
                        "edge_isInfluential": e.get("isInfluential", False),
                        "contexts": e.get("contexts") if include_contexts else None,
                    })
            nxt = batch.get("next")
            if nxt is None:
                break
            params["offset"] = nxt
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (400, 404):
            nf.add(citing_id)
            logging.info(f"S2 not found (refs): {citing_id}")
            return []
        raise
    finally:
        if stats is not None:
            stats[citing_id] = {
                "total_cits": total_cits,
                "with_paperId": with_pid,
                "no_paperId": total_cits - with_pid,
                "empty_intents": empty_intents,
                "influ_true": influ_true
            }   
    return [e for e in edges if e["citing_paperId"]]
    # return [e for e in edges if e["citing_paperId"]], stats if stats is not None else None # Debugging

# ---------- main: enrich your cit dataframe ----------
def enrich_with_intents_and_influential(cit_df, include_contexts=False, rps=1.0, not_found_path=NOT_FOUND_PATH):
    """
    cit_df must have columns: citingPaperId, citedPaperId
    Returns (df, info) where:
      - df includes: intents, edge_isInfluential, contexts?,
                     citing_influentialCitationCount, cited_influentialCitationCount
      - info: dict with keys:
          'not_found_all', 'not_found_new', 'errors'
    """
    df = cit_df.copy()

    # Load persistent not-found cache
    not_found = _load_not_found(not_found_path) if not_found_path else set()
    baseline_nf_size = len(not_found)

    # 1) Fetch all references once per unique citing DOI (skipping known-missing)
    sess = requests.Session()
    ref_map = {}
    errors = []
    unique_cited = [d for d in df["citedPaperId"].dropna().unique()]
    for cited in unique_cited:
        if cited in not_found:
            ref_map[cited] = {}
            continue
        try:
            ref_stats = {}
            edges = fetch_citation_edges_for_cited(
                cited, include_contexts=include_contexts, rps=rps, session=sess, not_found=not_found, stats=ref_stats
            )
        except RuntimeError:
            # response too big -> refetch with smaller pages
            edges = []
            url = f"{SS_BASE}/paper/{cited}/citations"
            fields = ["intents","isInfluential", "paperId"]
            if include_contexts:
                fields.append("contexts") # check if this is correct, as the original code had "contextsWithIntent"
            params = {"fields": ",".join(fields), "offset": 0, "limit": 200}
            last = 0.0
            try:
                while True:
                    r, last = _get(sess, url, params, last, rps=rps)
                    batch = r.json() or {}
                    raw = batch.get("data") or []
                    for e in raw:
                        edges.append({
                            "citing_paperId": e.get("citingPaper", {}).get("paperId", None),
                            "intents": e.get("intents", []) or [],
                            "edge_isInfluential": e.get("isInfluential", False),
                            "contexts": e.get("contexts") if include_contexts else None,
                        })
                    nxt = batch.get("next")
                    if nxt is None: break
                    params["offset"] = nxt
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status in (400, 404):
                    not_found.add(cited)
                    edges = []
                else:
                    errors.append({"cited": cited, "error": str(e)})
                    edges = []
        except Exception as e:
            errors.append({"cited": cited, "error": str(e)})
            edges = []

        ref_map[cited] = {e["citing_paperId"]: e for e in edges if e["citing_paperId"]}

    # 2) Map edge info to each row
    def _lookup_edge(row):
        m = ref_map.get(row["citedPaperId"], {})
        e = m.get(row["citingPaperId"], None)
        if e:
            print(f"hit: cited={row['citedPaperId']} citing={row['citingPaperId']}") # Debugging
            return pd.Series({
                "intents": e.get("intents", []),
                "edge_isInfluential": e.get("edge_isInfluential", False),
                "contexts": e.get("contexts"),
            })
        else:
            return pd.Series({"intents": [], "edge_isInfluential": None, "contexts": None})
        

    edge_cols = df.apply(_lookup_edge, axis=1)
    df = pd.concat([df, edge_cols], axis=1)

    # 3) Add paper-level influentialCitationCount for both sides (skip known-missing)
    all_ids = pd.unique(pd.concat([df["citingPaperId"], df["citedPaperId"]], ignore_index=True).dropna())
    want = [d for d in all_ids if d not in not_found]
    infl_map = fetch_influential_counts(want, rps=rps, session=sess, not_found=not_found)
    # also ensure missing ones map to None
    for d in all_ids:
        if d not in infl_map:
            infl_map[d] = None
    df["citing_influentialCitationCount"] = df["citingPaperId"].map(infl_map)
    df["cited_influentialCitationCount"]  = df["citedPaperId"].map(infl_map)

    # Persist not-found set if it changed
    if not_found_path and len(not_found) > baseline_nf_size:
        _save_not_found(not_found, not_found_path)

    info = {
        "not_found_all": sorted(not_found),
        "not_found_new": sorted(list(not_found))[baseline_nf_size:],
        "errors": errors,
    }
    return df, info, ref_map, ref_stats if 'ref_stats' in locals() else None