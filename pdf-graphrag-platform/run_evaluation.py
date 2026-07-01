"""
run_evaluation.py — Run the extraction accuracy evaluation end-to-end.

This is the script that produces your real, defensible accuracy number.

USAGE
-----
Step 1 — Generate the ground truth template (first time only):
    python run_evaluation.py template

Step 2 — Fill in data/ground_truth/ground_truth_template.json
    Open the file and add real entries from your PDF.
    10-20 entries is enough for a reliable number.
    Each entry = one financial figure you verified by hand from the source.

Step 3 — Run the evaluation:
    python run_evaluation.py

    This will:
      a) Ingest all PDFs in data/raw_pdfs/
      b) Run the extraction pipeline
      c) Compare extracted metrics against your ground truth
      d) Print a full accuracy report with F1, precision, recall
      e) Save results to data/evaluation/

Step 4 — Read your number:
    The headline metric is F1 (strict).
    The report also shows per-record breakdowns so you can see exactly
    which figures the pipeline got right and which it missed.

HOW TO READ THE RESULTS:
    ✓ = correctly extracted (value within 1% of truth)
    ~ = lenient match (value within 5% — found the right fact, off by scale)
    ✗ = missed or wrong

    F1 (strict) is your primary CV metric.
    Precision tells you about false positives (hallucinated figures).
    Recall tells you about false negatives (real figures the system missed).

WHAT TO DO WITH A LOW F1:
    If F1 is below ~0.6:
      - Check the per-record breakdown for the failure pattern
      - If most are "WRONG VALUE": the model is finding figures but reading
        scale wrong (billion vs million) — the most common small-model error
      - If most are "NO MATCH": the model is missing entities / facts entirely
      - Try LLM_PROVIDER=openai in .env to see if a stronger model fixes it
      - Try MAX_EXTRACTION_CHUNKS=0 to ensure all chunks are processed

    A low F1 is a real and interesting finding for an interview —
    "the small local model achieved F1=0.58 vs F1=0.84 with gpt-4o-mini"
    is a better story than a fabricated high number.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path


def main():
    args = sys.argv[1:]

    # ── Template generation mode ──────────────────────────────────────
    if args and args[0] == "template":
        from app.evaluation.ground_truth import generate_template
        path = generate_template()
        print(f"\nTemplate created: {path}")
        print("Open it and fill in real financial figures from your PDF.")
        print("Then run: python run_evaluation.py")
        return

    # ── Full evaluation mode ──────────────────────────────────────────
    from app.config import settings
    from app.ingestion.loader import load_all_pdfs
    from app.ingestion.chunker import chunk_pages
    from app.ingestion.embedder import embed_chunks
    from app.ingestion.vector_store import VectorStore
    from app.analytics.metrics_extractor import extract_metrics_full
    from app.evaluation.ground_truth import load_all_ground_truth
    from app.evaluation.evaluator import evaluate
    from app.evaluation.eval_report import save_evaluation

    print("=" * 64)
    print("  PDF GraphRAG Platform — Extraction Accuracy Evaluation")
    print("=" * 64)

    # 1. Load ground truth
    print("\n▶ Loading ground truth…")
    gt_files = load_all_ground_truth()
    if not gt_files:
        print("\n!! No ground truth files found in data/ground_truth/")
        print("   Run: python run_evaluation.py template")
        print("   Then fill in the template and rerun.")
        return

    all_entries = [e for gtf in gt_files for e in gtf.entries]
    print(f"  ✓ {len(all_entries)} ground truth entries from {len(gt_files)} file(s)")

    # 2. Ingest + extract
    print("\n▶ Ingesting PDFs…")
    pdfs = sorted(settings.raw_pdfs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"  !! No PDFs in {settings.raw_pdfs_dir}")
        return

    pages  = load_all_pdfs()
    # overlap=0 for extraction — each sentence processed exactly once.
    # Overlap is kept for Q&A retrieval chunks (different code path).
    # Using overlap here causes the same sentence to appear in 2-3 chunks,
    # each generating a duplicate MetricRecord, which destroys precision.
    chunks = chunk_pages(pages, overlap=0)
    print(f"  ✓ {len(pages)} pages → {len(chunks)} chunks "
          f"(overlap=0, no duplicates) from {len(pdfs)} PDF(s)")

    print("\n▶ Embedding + indexing (for completeness)…")
    vectors = embed_chunks(chunks)
    store   = VectorStore()
    store.build(chunks, vectors)
    print(f"  ✓ {vectors.shape[0]} vectors indexed")

    print("\n▶ Running extraction (both passes + deduplication)…")
    t0 = time.perf_counter()
    metrics = extract_metrics_full(chunks)
    elapsed = time.perf_counter() - t0
    print(f"  ✓ {len(metrics)} metrics extracted in {elapsed:.1f}s")

    if not metrics:
        print("\n!! No metrics extracted. Possible reasons:")
        print("   - PDF has no financial text (try a real annual report)")
        print("   - LLM returned empty JSON (try LLM_PROVIDER=openai)")
        print("   - MAX_EXTRACTION_CHUNKS is too low")
        return

    # 3. Evaluate
    print("\n▶ Evaluating against ground truth…")
    result = evaluate(metrics, all_entries)

    # 4. Print report
    print(result.summary())

    # 5. Save
    json_path, text_path = save_evaluation(result)
    print(f"\n  Results saved:")
    print(f"    JSON : {json_path}")
    print(f"    Text : {text_path}")

    # 6. CV-ready summary
    print("\n" + "=" * 64)
    print("  CV-READY NUMBERS — copy these:")
    print("=" * 64)
    print(f"  Extraction F1 (strict, ±1%)  : {result.f1_strict:.1%}")
    print(f"  Extraction F1 (lenient, ±5%) : {result.f1_lenient:.1%}")
    print(f"  Precision (strict)           : {result.precision_strict:.1%}")
    print(f"  Recall (strict)              : {result.recall_strict:.1%}")
    print(f"  Accuracy (strict)            : {result.extraction_accuracy_strict:.1%}  "
          f"({result.true_positives_strict}/{result.total_extracted} correct)")
    print(f"  Ground truth entries used    : {result.total_ground_truth}")
    print(f"  Extracted records evaluated  : {result.total_extracted}")
    print("=" * 64)
    print()
    print("  Suggested CV bullet (fill in your numbers):")
    print(f"  Achieved F1 {result.f1_strict:.0%} extraction accuracy across "
          f"{result.total_ground_truth} hand-verified financial figures by")
    print(f"  integrating LLM structured extraction with pandas aggregation")
    print()


if __name__ == "__main__":
    main()
