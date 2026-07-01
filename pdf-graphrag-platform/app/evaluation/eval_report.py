"""
eval_report.py — Save evaluation results to disk and generate a summary.

WHY THIS EXISTS:
  The evaluation should produce a persistent artefact — a report you can
  open, read, and cite numbers from — not just terminal output that scrolls
  away. This module saves both a machine-readable JSON (for further analysis)
  and a human-readable text report.

  The JSON is also used by the API's /evaluate/results endpoint so the
  Streamlit UI can display the accuracy numbers.
"""

from __future__ import annotations
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.evaluation.evaluator import EvaluationResult
from app.utils.logger import logger

EVAL_DIR = Path("data/evaluation")


def save_evaluation(result: EvaluationResult,
                    run_name: str | None = None) -> tuple[Path, Path]:
    """
    Save the evaluation result as JSON + text report.
    Returns (json_path, text_path).
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{run_name}_{ts}" if run_name else ts

    # ── JSON (machine-readable) ──
    json_data = {
        "run": tag,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_ground_truth":          result.total_ground_truth,
            "total_extracted":             result.total_extracted,
            "f1_strict":                   result.f1_strict,
            "precision_strict":            result.precision_strict,
            "recall_strict":               result.recall_strict,
            "extraction_accuracy_strict":  result.extraction_accuracy_strict,
            "f1_lenient":                  result.f1_lenient,
            "precision_lenient":           result.precision_lenient,
            "recall_lenient":              result.recall_lenient,
            "extraction_accuracy_lenient": result.extraction_accuracy_lenient,
            "true_positives_strict":       result.true_positives_strict,
            "false_positives_strict":      result.false_positives_strict,
            "false_negatives_strict":      result.false_negatives_strict,
        },
        "per_record": [
            {
                "entity":              m.extracted.entity,
                "metric":              m.extracted.metric,
                "extracted_value":     m.extracted.value,
                "extracted_unit":      m.extracted.unit,
                "extracted_period":    m.extracted.period,
                "source_page":         m.extracted.page_num,
                "truth_value":         m.truth.true_value if m.truth else None,
                "truth_entity":        m.truth.entity     if m.truth else None,
                "strict_hit":          m.strict_hit,
                "lenient_hit":         m.lenient_hit,
                "label":               m.label,
            }
            for m in result.matches
        ],
    }
    json_path = EVAL_DIR / f"eval_{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # ── Text report (human-readable) ──
    text_path = EVAL_DIR / f"eval_{tag}.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"PDF GraphRAG Platform — Extraction Accuracy Report\n")
        f.write(f"Run: {tag}\n")
        f.write(result.summary())

    logger.success(f"Evaluation saved → {json_path.name}")
    return json_path, text_path


def load_latest_evaluation() -> dict | None:
    """Load the most recent saved evaluation JSON for the UI."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(EVAL_DIR.glob("eval_*.json"), reverse=True)
    if not files:
        return None
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)
