"""
evaluator.py — Compare extracted metrics against ground truth.

WHY THIS EXISTS:
  Extraction accuracy is the honest gap in the previous pipeline. This
  module closes it by defining precisely what "correct" means for a
  financial metric and computing standard information-retrieval metrics:
  precision, recall, and F1.

WHAT "CORRECT" MEANS HERE:
  Matching an extracted metric against a ground truth entry requires
  comparing across three dimensions simultaneously:

  1. ENTITY MATCH — did we attribute it to the right company?
     Uses fuzzy matching (lowercase, normalised) because the LLM might
     extract "Acme Corporation" while the ground truth says "Acme Corp."

  2. METRIC MATCH — did we identify the right metric type?
     "revenue" vs "total revenue" vs "net revenue" — fuzzy substring match.

  3. VALUE MATCH — did we extract the right number?
     Uses a configurable tolerance (default 1%) to handle floating point
     and minor rounding differences.

TWO STRICTNESS LEVELS:
  STRICT  — entity + metric + value must all match within tolerance
  LENIENT — entity + metric match, value within 5% (for imprecise PDFs)

  Reporting both gives a more honest picture: strict shows how accurate
  the system is end-to-end; lenient shows how often it found the right
  fact even if it read the scale slightly wrong.

METRICS — standard information retrieval:
  Precision = of what we extracted, what fraction was correct?
              (measures false positives — hallucinated/wrong extractions)
  Recall    = of the true facts, what fraction did we find?
              (measures false negatives — missed facts)
  F1        = harmonic mean of precision and recall
              (single number that balances both)

WHY THESE THREE AND NOT JUST ACCURACY:
  "Accuracy" is ambiguous when there are false positives AND false negatives.
  Precision tells you about quality; recall tells you about coverage.
  F1 is the standard single-number summary used in NLP extraction tasks,
  which is exactly what this is — so F1 is the headline metric you cite.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

from app.analytics.metrics_extractor import MetricRecord
from app.evaluation.ground_truth import GroundTruthEntry
from app.utils.logger import logger


# ── Matching helpers ──────────────────────────────────────────────────

def _normalise(s: str) -> str:
    """Lowercase, replace underscores with spaces, strip punctuation, compress whitespace."""
    s = s.lower().replace("_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s)).strip()


def _entity_match(extracted: str, truth: str) -> bool:
    """
    True if the extracted entity name fuzzy-matches the ground truth.
    Accepts either as a substring of the other (handles abbreviations,
    "Corp" vs "Corporation", dropped "Inc.", etc.)
    """
    e, t = _normalise(extracted), _normalise(truth)
    return e == t or e in t or t in e


def _metric_match(extracted: str, truth: str) -> bool:
    """
    True if metric types are compatible.
    "total revenue" should match "revenue"; "net income" should match "income".
    """
    e, t = _normalise(extracted), _normalise(truth)
    return e == t or e in t or t in e


def _value_match(extracted: float, truth: float, tolerance: float = 0.01) -> bool:
    """
    True if extracted value is within *tolerance* (relative) of truth.
    Default 1% handles minor rounding. Avoids division by zero.
    """
    if truth == 0:
        return abs(extracted) < 1e-6
    return abs(extracted - truth) / abs(truth) <= tolerance


# ── Match result ──────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """Records the outcome of comparing one extracted record to ground truth."""
    extracted:       MetricRecord
    truth:           GroundTruthEntry | None   # None = no matching ground truth
    entity_matched:  bool = False
    metric_matched:  bool = False
    value_matched_strict:  bool = False        # within 1%
    value_matched_lenient: bool = False        # within 5%

    @property
    def strict_hit(self) -> bool:
        return self.entity_matched and self.metric_matched and self.value_matched_strict

    @property
    def lenient_hit(self) -> bool:
        return self.entity_matched and self.metric_matched and self.value_matched_lenient

    @property
    def label(self) -> str:
        if self.strict_hit:
            return "CORRECT (strict)"
        if self.lenient_hit:
            return "CORRECT (lenient)"
        if self.entity_matched and self.metric_matched:
            return "WRONG VALUE"
        if self.entity_matched:
            return "WRONG METRIC"
        return "NO MATCH"


# ── Evaluation results ────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Full evaluation report for one run."""
    total_ground_truth:  int
    total_extracted:     int
    matches:             list[MatchResult] = field(default_factory=list)

    # Strict metrics (1% value tolerance)
    true_positives_strict:  int = 0
    false_positives_strict: int = 0
    false_negatives_strict: int = 0

    # Lenient metrics (5% value tolerance)
    true_positives_lenient:  int = 0
    false_positives_lenient: int = 0
    false_negatives_lenient: int = 0

    def _safe_div(self, a: int, b: int) -> float:
        return round(a / b, 4) if b else 0.0

    # ── Strict metrics ──
    @property
    def precision_strict(self) -> float:
        return self._safe_div(self.true_positives_strict,
                              self.true_positives_strict + self.false_positives_strict)

    @property
    def recall_strict(self) -> float:
        return self._safe_div(self.true_positives_strict,
                              self.true_positives_strict + self.false_negatives_strict)

    @property
    def f1_strict(self) -> float:
        p, r = self.precision_strict, self.recall_strict
        return self._safe_div(2 * p * r, p + r)

    # ── Lenient metrics ──
    @property
    def precision_lenient(self) -> float:
        return self._safe_div(self.true_positives_lenient,
                              self.true_positives_lenient + self.false_positives_lenient)

    @property
    def recall_lenient(self) -> float:
        return self._safe_div(self.true_positives_lenient,
                              self.true_positives_lenient + self.false_negatives_lenient)

    @property
    def f1_lenient(self) -> float:
        p, r = self.precision_lenient, self.recall_lenient
        return self._safe_div(2 * p * r, p + r)

    @property
    def extraction_accuracy_strict(self) -> float:
        """% of extracted records that were strictly correct."""
        return self._safe_div(self.true_positives_strict, self.total_extracted)

    @property
    def extraction_accuracy_lenient(self) -> float:
        return self._safe_div(self.true_positives_lenient, self.total_extracted)

    def summary(self) -> str:
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════╗",
            "║          EXTRACTION ACCURACY EVALUATION REPORT          ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"  Ground truth entries  : {self.total_ground_truth}",
            f"  Extracted records     : {self.total_extracted}",
            "",
            "  ── STRICT (value within 1% of truth) ──────────────────",
            f"  Extraction accuracy   : {self.extraction_accuracy_strict:.1%}  "
            f"({self.true_positives_strict}/{self.total_extracted} correct)",
            f"  Precision             : {self.precision_strict:.1%}",
            f"  Recall                : {self.recall_strict:.1%}",
            f"  F1 score              : {self.f1_strict:.1%}  ← HEADLINE METRIC",
            "",
            "  ── LENIENT (value within 5% of truth) ──────────────────",
            f"  Extraction accuracy   : {self.extraction_accuracy_lenient:.1%}  "
            f"({self.true_positives_lenient}/{self.total_extracted} correct)",
            f"  Precision             : {self.precision_lenient:.1%}",
            f"  Recall                : {self.recall_lenient:.1%}",
            f"  F1 score              : {self.f1_lenient:.1%}",
            "",
            "  ── PER-RECORD BREAKDOWN ────────────────────────────────",
        ]
        for m in self.matches:
            tag = "✓" if m.strict_hit else ("~" if m.lenient_hit else "✗")
            entity = (m.extracted.entity or "?")[:25]
            metric = (m.extracted.metric or "?")[:15]
            val = f"{m.extracted.value:,.0f}"
            truth_val = f"{m.truth.true_value:,.0f}" if m.truth else "no GT"
            lines.append(
                f"  [{tag}] {entity:<25} {metric:<15} "
                f"extracted={val:<15} truth={truth_val:<15} → {m.label}"
            )
        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ── Core evaluation logic ─────────────────────────────────────────────

def evaluate(extracted: list[MetricRecord],
             ground_truth: list[GroundTruthEntry],
             strict_tol: float = 0.01,
             lenient_tol: float = 0.05) -> EvaluationResult:
    """
    Compare extracted MetricRecords against ground truth entries.

    Algorithm:
      For each extracted record, find the best-matching ground truth entry
      (greedy: entity match first, then metric, then value). Each ground
      truth entry can only be matched once — prevents double-counting.
      Unmatched ground truth entries become false negatives.
      Unmatched extracted records become false positives.
    """
    result = EvaluationResult(
        total_ground_truth=len(ground_truth),
        total_extracted=len(extracted),
    )

    unmatched_gt = list(ground_truth)   # shrinks as entries are consumed
    matches: list[MatchResult] = []

    for rec in extracted:
        best: GroundTruthEntry | None = None
        best_score = -1

        for gt in unmatched_gt:
            em = _entity_match(rec.entity, gt.entity)
            mm = _metric_match(rec.metric, gt.metric)
            vm_strict  = _value_match(rec.value, gt.true_value, strict_tol)
            vm_lenient = _value_match(rec.value, gt.true_value, lenient_tol)
            score = int(em) * 4 + int(mm) * 2 + int(vm_strict) + int(vm_lenient) * 0.5
            if score > best_score and em:   # entity must match to be a candidate
                best_score = score
                best = gt

        if best is not None:
            unmatched_gt.remove(best)
            em = _entity_match(rec.entity, best.entity)
            mm = _metric_match(rec.metric, best.metric)
            vms = _value_match(rec.value, best.true_value, strict_tol)
            vml = _value_match(rec.value, best.true_value, lenient_tol)
            match = MatchResult(
                extracted=rec, truth=best,
                entity_matched=em, metric_matched=mm,
                value_matched_strict=vms, value_matched_lenient=vml,
            )
        else:
            match = MatchResult(extracted=rec, truth=None)

        matches.append(match)

        # Tally TP/FP per strictness
        if match.strict_hit:
            result.true_positives_strict += 1
        else:
            result.false_positives_strict += 1

        if match.lenient_hit:
            result.true_positives_lenient += 1
        else:
            result.false_positives_lenient += 1

    # Remaining unmatched ground truth = false negatives
    result.false_negatives_strict  = len(unmatched_gt)
    result.false_negatives_lenient = len(unmatched_gt)
    result.matches = matches

    logger.info(
        f"Evaluation: {result.true_positives_strict} strict hits, "
        f"{result.false_positives_strict} FP, "
        f"{result.false_negatives_strict} FN "
        f"→ F1 {result.f1_strict:.1%}"
    )
    return result
