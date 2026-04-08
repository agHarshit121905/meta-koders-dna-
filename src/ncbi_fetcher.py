"""
NCBI E-utilities fetcher.
All sequence data comes directly from NCBI – no synthetic data is ever generated.
API: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
"""
import re
import time
import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "crispr_openenv_agent"
EMAIL = "hackathon@openenv.ai"


# Curated list of well-characterised human disease-gene accessions (NCBI RefSeq)
GENE_CATALOG = {
    "BRCA1 (Breast Cancer Gene 1)":       "NM_007294",
    "TP53 (Tumour Suppressor p53)":        "NM_000546",
    "CFTR (Cystic Fibrosis)":             "NM_000492",
    "HTT (Huntington Disease)":            "NM_002111",
    "EGFR (Lung Cancer Target)":           "NM_005228",
    "VEGFA (Angiogenesis Factor)":         "NM_001025366",
    "TNF (Tumour Necrosis Factor)":        "NM_000594",
    "IL6 (Interleukin-6)":                 "NM_000600",
    "APOE (Alzheimer Risk Gene)":          "NM_000041",
    "ACE2 (SARS-CoV-2 Receptor)":         "NM_021804",
}


def _get(url: str, params: dict) -> requests.Response:
    params.update({"tool": TOOL, "email": EMAIL})
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    time.sleep(0.35)          # NCBI rate-limit: ≤3 req/sec
    return r


def search_gene(accession: str) -> str | None:
    """Return NCBI UID for a RefSeq accession (e.g. NM_007294)."""
    r = _get(f"{EUTILS_BASE}/esearch.fcgi", {
        "db": "nucleotide", "term": f"{accession}[accn]", "retmode": "json"
    })
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    return ids[0] if ids else None


def fetch_sequence(uid: str, start: int = 0, length: int = 500) -> dict:
    """
    Fetch a sub-sequence from NCBI nucleotide by UID.
    Returns {'accession', 'title', 'sequence', 'start', 'length'}.
    FASTA format – no synthetic data.
    """
    r = _get(f"{EUTILS_BASE}/efetch.fcgi", {
        "db": "nucleotide", "id": uid, "rettype": "fasta", "retmode": "text",
        "seq_start": start + 1,          # NCBI is 1-based
        "seq_stop": start + length,
    })
    text = r.text.strip()
    lines = text.splitlines()
    header = lines[0]                    # >accession description
    seq = "".join(lines[1:]).upper()
    # Extract accession from header
    acc = header.split()[0].lstrip(">")
    title = " ".join(header.split()[1:])
    return {"accession": acc, "title": title, "sequence": seq,
            "start": start, "length": len(seq)}


def fetch_gene_info_offline(accession: str) -> dict | None:
    """Return pre-fetched NCBI sequence when network is unavailable."""
    from bundled_sequences import BUNDLED
    entry = BUNDLED.get(accession)
    if not entry:
        return None
    return {
        "accession": accession,
        "title": entry["title"] + "  [offline — bundled NCBI sequence]",
        "sequence": entry["sequence"],
        "start": entry["region_start"],
        "length": len(entry["sequence"]),
    }


def fetch_gene_info(accession: str, region_start: int = 200,
                    region_length: int = 400) -> dict | None:
    """
    High-level helper: search accession → fetch region → return data dict.
    Returns None if NCBI is unreachable.

    Set OFFLINE_MODE=1 in environment to skip NCBI entirely and use
    bundled sequences (recommended for containerised / judge environments).
    """
    import os
    if os.getenv("OFFLINE_MODE", "0").strip() in ("1", "true", "yes"):
        return fetch_gene_info_offline(accession)
    try:
        uid = search_gene(accession)
        if not uid:
            return fetch_gene_info_offline(accession)
        data = fetch_sequence(uid, start=region_start, length=region_length)
        return data
    except Exception:
        return fetch_gene_info_offline(accession)