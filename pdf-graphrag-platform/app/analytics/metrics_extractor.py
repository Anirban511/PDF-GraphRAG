"""
metrics_extractor.py — Pull structured financial metrics out of documents.

WHY THIS EXISTS:
  The RAG layer answers questions. The ANALYTICS layer turns the same
  documents into structured, quantitative output a business user can act on
  — KPIs, trends, and comparisons. This module is the bridge: it extracts
  numeric financial facts (revenue, profit, growth %, costs) into a clean
  table that the reporting layer can chart and summarise.

DESIGN:
  • Uses the local LLM to extract metrics as strict JSON (value + unit +
    period + entity), because financial figures appear in wildly different
    phrasings ("$2.5M", "2.5 million dollars", "USD 2,500,000").
  • Every extracted metric keeps its source (filename + page) so the
    resulting analytics remain auditable — a non-negotiable for finance.
  • Output is a list of MetricRecord dataclasses, trivially convertible to a
    pandas DataFrame for aggregation and charting.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict

from app.generation.llm import call_llm
from app.ingestion.chunker import Chunk
from app.utils.logger import logger

_METRIC_SYSTEM = """You extract financial metrics from business/financial text.
For each metric found, capture: entity, metric name, value, unit, and time period.

CRITICAL — ENTITY NAME:
Use ONLY entity names that appear explicitly in the text. Copy the exact
company or segment name from the text. Do NOT invent or substitute entity names.
If a segment is mentioned as "Cloud Services segment", use "Cloud Services".

CRITICAL — VALUE NORMALISATION (most important rule):
Always output values as plain integers. Never use M, B, K suffixes.
  "$4.85 billion"  -> 4850000000
  "$920 million"   -> 920000000
  "$3.62 billion"  -> 3620000000
  "$18.4 billion"  -> 18400000000
  "$420 million"   -> 420000000

CRITICAL — EXTRACT BOTH CURRENT AND COMPARISON FIGURES:
When a sentence says "X grew to $A from $B in [prior period]", extract BOTH:
  1. Current period: value=A, period=current
  2. Prior period:   value=B, period=prior

CRITICAL — ONLY EXTRACT THESE METRIC TYPES (ignore everything else):
  revenue, operating_income, net_income, total_assets, cash,
  long_term_debt, research_and_development, capital_expenditure,
  acquisition_cost, free_cash_flow
DO NOT extract: EPS, earnings per share, dividends, percentages, ratios,
  margins, guidance ranges, segment percentages, or year numbers.

Respond with ONLY valid JSON matching this exact shape, no other text:
{"metrics": [
  {"entity": "<exact name from text>", "metric": "revenue",
   "value": 4850000000, "unit": "USD", "period": "FY2024"},
  {"entity": "<exact name from text>", "metric": "revenue",
   "value": 4250000000, "unit": "USD", "period": "FY2023"}
]}
If no matching metrics found, return {"metrics": []}."""

_COMPARISON_SYSTEM = """You extract ONLY prior-period comparison figures from financial text.
Look specifically for phrases like:
  "up from X in [prior period]", "compared to X in [year]",
  "from $X in fiscal [year]", "versus $X in the prior year"

CRITICAL — Use ONLY entity names that appear explicitly in the text.
Do NOT invent entity names. Copy them exactly as they appear.

Apply the same normalisation: "$920 million" = 920000000, "$1.88 billion" = 1880000000.
Only extract: revenue, operating_income, net_income, total_assets, cash,
  long_term_debt, research_and_development, capital_expenditure.
Do NOT extract EPS, dividends, percentages, or guidance figures.

Respond ONLY with valid JSON:
{"metrics": [
  {"entity": "<exact name from text>", "metric": "revenue",
   "value": 4250000000, "unit": "USD", "period": "FY2023"}
]}
If no comparison figures found, return {"metrics": []}."""

_METRIC_USER = "Extract financial metrics from:\n\n{text}"
_COMPARISON_USER = "Extract prior-period comparison figures from:\n\n{text}"

@dataclass
class MetricRecord:
    entity: str
    metric: str
    value: float
    unit: str
    period: str
    filename: str
    page_num: int


def _coerce_value(v) -> float | None:
    """Turn '2.5M', '$1,200', 3000 etc. into a float."""
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.replace(",", "").replace("$", "").strip().upper()
    mult = 1.0
    if s.endswith("M"):
        mult, s = 1_000_000, s[:-1]
    elif s.endswith("B"):
        mult, s = 1_000_000_000, s[:-1]
    elif s.endswith("K"):
        mult, s = 1_000, s[:-1]
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) * mult if m else None


def extract_metrics(chunks: list[Chunk]) -> list[MetricRecord]:
    """Extract financial metrics from chunks into structured records.

    Shows a live progress bar with ETA — one LLM call per chunk, so this is
    the slowest stage of the pipeline (especially on CPU).
    """
    from tqdm import tqdm
    from app.config import settings
    # Optional cap: LLM-per-chunk is the slowest stage; cap it on large docs
    if settings.max_extraction_chunks and len(chunks) > settings.max_extraction_chunks:
        logger.warning(
            f"Capping metric extraction at {settings.max_extraction_chunks} "
            f"of {len(chunks)} chunks (set MAX_EXTRACTION_CHUNKS=0 to disable)."
        )
        chunks = chunks[: settings.max_extraction_chunks]
    records: list[MetricRecord] = []
    for i, chunk in enumerate(tqdm(chunks, desc="Extracting metrics", unit="chunk"), 1):
        try:
            raw = call_llm(system=_METRIC_SYSTEM,
                           user=_METRIC_USER.format(text=chunk.text),
                           max_tokens=1024, temperature=0.0)
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
        except Exception as exc:
            logger.debug(f"Metric extraction skipped on {chunk.chunk_id}: {exc}")
            continue

        for m in data.get("metrics", []):
            val = _coerce_value(m.get("value"))
            if val is None:
                continue
            records.append(MetricRecord(
                entity=str(m.get("entity", "Unknown")).strip(),
                metric=str(m.get("metric", "")).strip().lower(),
                value=val,
                unit=str(m.get("unit", "")).strip(),
                period=str(m.get("period", "")).strip(),
                filename=chunk.filename,
                page_num=chunk.page_num,
            ))

    logger.success(f"Extracted {len(records)} financial metrics.")
    return records


def extract_comparison_figures(chunks: list[Chunk]) -> list[MetricRecord]:
    """
    Second-pass extraction targeting prior-period comparison figures.

    WHY THIS EXISTS (Path 3 improvement):
      The first pass focuses on the headline metric in each sentence. When a
      sentence says "revenue grew to $4.85B from $4.25B in FY2023", the first
      pass reliably extracts the $4.85B but frequently drops the $4.25B.
      This pass uses a dedicated prompt that specifically looks for the
      "from X in prior year" patterns — phrases the main extraction misses.

      Running both passes and merging (deduplicated) gives materially better
      recall on prior-period figures, which make up the majority of false
      negatives in small-model extraction.
    """
    from tqdm import tqdm
    from app.config import settings

    if settings.max_extraction_chunks and len(chunks) > settings.max_extraction_chunks:
        chunks = chunks[: settings.max_extraction_chunks]

    records: list[MetricRecord] = []
    for chunk in tqdm(chunks, desc="Second pass (comparison figures)", unit="chunk"):
        # Only run on chunks that contain comparison language — avoids wasted calls
        comparison_signals = ["from $", "up from", "compared to", "versus",
                              "prior year", "prior period", "increased from",
                              "decreased from", "grew from"]
        text_lower = chunk.text.lower()
        if not any(sig in text_lower for sig in comparison_signals):
            continue

        try:
            raw = call_llm(system=_COMPARISON_SYSTEM,
                           user=_COMPARISON_USER.format(text=chunk.text),
                           max_tokens=512, temperature=0.0)
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
        except Exception as exc:
            logger.debug(f"Comparison pass skipped on {chunk.chunk_id}: {exc}")
            continue

        for m in data.get("metrics", []):
            val = _coerce_value(m.get("value"))
            if val is None:
                continue
            records.append(MetricRecord(
                entity=str(m.get("entity", "Unknown")).strip(),
                metric=str(m.get("metric", "")).strip().lower(),
                value=val,
                unit=str(m.get("unit", "")).strip(),
                period=str(m.get("period", "")).strip(),
                filename=chunk.filename,
                page_num=chunk.page_num,
            ))

    logger.success(f"Comparison pass: {len(records)} additional figures found.")
    return records


def _normalise_entity(s: str) -> str:
    """Lowercase, strip punctuation, compress whitespace for dedup keying."""
    import re
    s = s.lower().replace("_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s)).strip()


def validate_entity_against_chunks(records: list[MetricRecord],
                                    chunks) -> list[MetricRecord]:
    """
    Drop records whose entity name does not appear in any chunk text.

    WHY THIS EXISTS:
      Small models hallucinate entity names from prompt examples (a known
      failure called "example leakage"). By checking that the extracted
      entity name appears somewhere in the source chunks, we filter out
      entities the model invented rather than read.

    MATCHING:
      Case-insensitive. Checks if a 4+ character prefix of the entity
      name appears in any chunk text. This handles "NexaCore Technologies"
      appearing when the text says "NexaCore" — the prefix "nexac" matches.
      Minimum 4 chars to avoid matching short common words.
    """
    all_text = " ".join(c.text.lower() for c in chunks)
    validated = []
    for rec in records:
        entity_norm = rec.entity.lower().strip()
        # Try progressively shorter prefixes to handle name variations
        matched = False
        words = entity_norm.split()
        for word in words:
            if len(word) >= 4 and word in all_text:
                matched = True
                break
        if matched:
            validated.append(rec)
        else:
            logger.debug(
                f"Entity validation removed: '{rec.entity}' "
                f"(not found in source text)"
            )
    removed = len(records) - len(validated)
    if removed:
        logger.info(
            f"Entity validation: removed {removed} records with "
            f"hallucinated entity names"
        )
    return validated


def deduplicate_records(records: list[MetricRecord]) -> list[MetricRecord]:
    """
    Collapse duplicate extractions caused by overlapping chunks.

    WHY THIS EXISTS:
      With chunk_size=512 and overlap=64, a financial sentence often appears
      in 2-3 consecutive chunks. Each chunk independently extracts the same
      metric, producing duplicates. A 22-figure document can yield 40-50+
      raw extractions before deduplication.

    ALGORITHM:
      Group by (entity_normalised, metric_normalised, period).
      Within each group, keep the record with the LARGEST value.
      Rationale: scale errors always produce a SMALLER number (billion read
      as million = 1000x smaller). The largest value in a group is almost
      always the correctly-scaled one. For genuine duplicates (same value
      extracted twice), any pick is correct.

    ADDITIONAL FILTERS:
      - Drop records with value < 1000 — these are ratios, percentages,
        EPS figures, or page numbers that slipped through (e.g. "14.2%"
        parsed as 14.2, "$6.84 EPS" parsed as 6.84).
      - Drop records with no entity or no metric.
    """
    import re

    MIN_VALUE = 1_000   # filter out percentages, EPS, ratios

    # First filter: remove obviously non-financial values
    filtered = [
        r for r in records
        if r.value >= MIN_VALUE
        and r.entity.strip()
        and r.metric.strip()
    ]

    # Group by (entity_norm, metric_norm, period)
    groups: dict[tuple, list[MetricRecord]] = {}
    for r in filtered:
        key = (
            _normalise_entity(r.entity),
            _normalise_entity(r.metric),
            str(r.period).strip().upper(),
        )
        groups.setdefault(key, []).append(r)

    # Keep the record with the largest value in each group
    deduped = [max(grp, key=lambda x: x.value) for grp in groups.values()]

    logger.info(
        f"Deduplication: {len(records)} raw → {len(filtered)} after "
        f"value filter → {len(deduped)} after group dedup"
    )
    return deduped


def extract_metrics_full(chunks: list[Chunk]) -> list[MetricRecord]:
    """
    Run both passes, validate entities, merge, and deduplicate.

    Pipeline:
      1. First pass  — headline metrics per chunk
      2. Second pass — prior-period comparison figures
      3. Entity validation — drop records whose entity is not in source text
         (catches hallucinated entity names from prompt example leakage)
      4. Deduplication — collapse chunk-overlap duplicates, keep largest value
    """
    first  = extract_metrics(chunks)
    second = extract_comparison_figures(chunks)
    raw    = first + second

    # Step 3: entity validation — removes "Acme Corp" type hallucinations
    validated = validate_entity_against_chunks(raw, chunks)

    # Step 4: dedup — collapses overlapping-chunk duplicates
    merged = deduplicate_records(validated)

    logger.info(
        f"Full extraction: {len(first)} (pass 1) + {len(second)} (pass 2) "
        f"→ {len(validated)} after entity validation "
        f"→ {len(merged)} after dedup"
    )
    return merged

