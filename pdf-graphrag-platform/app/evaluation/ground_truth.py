"""
ground_truth.py — Ground truth schema, loader, and template generator.

WHY THIS EXISTS:
  This is what transforms the project from "pipeline that produces output"
  to "evaluated system with a measurable accuracy." Ground truth is the
  human-verified set of correct answers the pipeline is measured against.

WHAT GROUND TRUTH IS:
  A JSON file you fill in by reading your PDF and writing down the real
  financial figures. It is the source of truth: "the PDF actually says
  Acme Corp revenue was $4.2B in FY2024 on page 12." The evaluator then
  compares what the pipeline extracted against this.

SCHEMA — one file per document set:
  {
    "meta": {
      "description": "...",
      "document_set": ["annual_report.pdf"],
      "verified_by": "manual reading",
      "date_verified": "2025-01-01"
    },
    "entries": [
      {
        "entity":        "Acme Corp",          <- company / org name
        "metric":        "revenue",            <- metric type (lowercase)
        "true_value":    4200000000,           <- normalised float (no M/B/K)
        "unit":          "USD",
        "period":        "FY2024",
        "source_page":   12,                   <- page you found it on
        "source_sentence": "Revenue grew to $4.2 billion for fiscal 2024"
      }
    ]
  }

HOW TO FILL IT IN (5-step process):
  1. Open your PDF.
  2. Find any sentence with a financial figure — revenue, profit, costs, etc.
  3. Write the entity (company name), metric type, the number normalised to
     a plain float (so "$4.2 billion" becomes 4200000000), the unit, and
     the time period.
  4. Record the page number and copy the sentence verbatim.
  5. Repeat for 10-20 entries — more gives a more reliable accuracy estimate.

10-20 entries is enough for a defensible preliminary accuracy number.
It takes about 15-20 minutes for a typical 20-page annual report.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.utils.logger import logger

# Where ground truth files live
GT_DIR = Path("data/ground_truth")


@dataclass
class GroundTruthEntry:
    """One verified financial fact from the source document."""
    entity:          str
    metric:          str
    true_value:      float
    unit:            str
    period:          str
    source_page:     int
    source_sentence: str = ""


@dataclass
class GroundTruthFile:
    """Full ground truth file for a document set."""
    meta:    dict
    entries: list[GroundTruthEntry] = field(default_factory=list)

    @property
    def document_set(self) -> list[str]:
        return self.meta.get("document_set", [])


def load_ground_truth(path: Path) -> GroundTruthFile:
    """Load and validate a ground truth JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = []
    for i, e in enumerate(raw.get("entries", [])):
        try:
            entries.append(GroundTruthEntry(
                entity=str(e["entity"]).strip(),
                metric=str(e["metric"]).strip().lower(),
                true_value=float(e["true_value"]),
                unit=str(e.get("unit", "")).strip(),
                period=str(e.get("period", "")).strip(),
                source_page=int(e.get("source_page", 0)),
                source_sentence=str(e.get("source_sentence", "")),
            ))
        except (KeyError, ValueError) as exc:
            logger.warning(f"Ground truth entry {i} skipped (bad format): {exc}")

    logger.info(f"Loaded {len(entries)} ground truth entries from {path.name}")
    return GroundTruthFile(meta=raw.get("meta", {}), entries=entries)


def load_all_ground_truth() -> list[GroundTruthFile]:
    """Load every *.json file in the ground truth directory."""
    GT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(GT_DIR.glob("*.json"))
    if not files:
        return []
    return [load_ground_truth(f) for f in files]


def generate_template(output_path: Path | None = None) -> Path:
    """
    Write a blank ground truth template the user fills in.
    Call this once to create the file, then edit it manually.
    """
    template = {
        "meta": {
            "description": "Ground truth for extraction accuracy evaluation",
            "document_set": ["your_document.pdf"],
            "verified_by": "manual reading",
            "date_verified": "YYYY-MM-DD",
            "notes": "Fill in 10-20 entries for a reliable accuracy estimate."
        },
        "entries": [
            {
                "entity": "Company Name Here",
                "metric": "revenue",
                "true_value": 4200000000,
                "unit": "USD",
                "period": "FY2024",
                "source_page": 12,
                "source_sentence": "Revenue grew to $4.2 billion for fiscal 2024."
            },
            {
                "entity": "Company Name Here",
                "metric": "net_income",
                "true_value": 850000000,
                "unit": "USD",
                "period": "FY2024",
                "source_page": 14,
                "source_sentence": "Net income was $850 million, up 8% year-over-year."
            },
            {
                "entity": "Company Name Here",
                "metric": "revenue",
                "true_value": 3800000000,
                "unit": "USD",
                "period": "FY2023",
                "source_page": 12,
                "source_sentence": "Revenue grew to $4.2 billion from $3.8 billion in 2023."
            }
        ]
    }
    output_path = output_path or (GT_DIR / "ground_truth_template.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
    logger.success(f"Template written → {output_path}")
    return output_path
