"""
create_test_pdf.py - Generate a realistic financial report PDF with known values.

This PDF contains deliberately realistic financial text so pdfplumber extracts
clean, readable sentences - the same quality you'd get from a real digital
(non-scanned) annual report. The values are fixed and known, which is exactly
what we need for ground truth verification.
"""

from fpdf import FPDF
from pathlib import Path


class FinancialReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, "NEXACORE TECHNOLOGIES INC. - ANNUAL REPORT 2024", align="C")
        self.ln(4)
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 60, 120)
        self.ln(4)
        self.cell(0, 10, title)
        self.ln(2)
        self.set_draw_color(30, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)
        self.set_text_color(40, 40, 40)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def kv_line(self, label, value, bold_value=True):
        self.set_font("Helvetica", "", 10)
        self.cell(90, 7, label)
        if bold_value:
            self.set_font("Helvetica", "B", 10)
        self.cell(0, 7, value)
        self.ln()
        self.set_font("Helvetica", "", 10)


def build_pdf(out_path: Path):
    pdf = FinancialReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)

    # ── PAGE 1: Executive Overview ────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("Executive Overview")
    pdf.body_text(
        "NexaCore Technologies Inc. is pleased to report strong financial performance "
        "for the fiscal year ended December 31, 2024. Total revenue reached "
        "$4.85 billion, representing a 14.2% increase from $4.25 billion in fiscal "
        "year 2023. This marks the fourth consecutive year of double-digit revenue growth "
        "for the company."
    )
    pdf.body_text(
        "Operating income for fiscal 2024 was $1.12 billion, up from $920 million "
        "in the prior year. Net income attributable to shareholders was $847 million, "
        "compared to $710 million in fiscal 2023, representing a year-over-year increase "
        "of 19.3%."
    )
    pdf.body_text(
        "Earnings per share (diluted) reached $6.84, compared to $5.72 in fiscal 2023. "
        "The board of directors has approved a quarterly dividend of $0.42 per share, "
        "totalling $1.68 per share annually."
    )

    # ── PAGE 1: Segment Performance ───────────────────────────────────
    pdf.chapter_title("Segment Performance")
    pdf.body_text(
        "NexaCore operates through three primary business segments: Cloud Services, "
        "Enterprise Software, and Professional Services."
    )
    pdf.body_text(
        "The Cloud Services segment generated revenue of $2.31 billion in fiscal 2024, "
        "a 22.7% increase from $1.88 billion in fiscal 2023. Cloud Services now represents "
        "47.6% of total company revenue, up from 44.2% in the prior year. Segment operating "
        "margin improved to 34.2% from 31.8%."
    )
    pdf.body_text(
        "Enterprise Software revenue was $1.74 billion, broadly flat compared to "
        "$1.71 billion in fiscal 2023 as the company continues to transition customers "
        "to subscription-based pricing. Recurring revenue within this segment now accounts "
        "for 78% of segment total."
    )
    pdf.body_text(
        "Professional Services contributed $800 million in revenue, up 6.7% from "
        "$750 million in fiscal 2023, driven by strong demand for implementation and "
        "integration services associated with the company's cloud migration offerings."
    )

    # ── PAGE 2: Balance Sheet ─────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("Balance Sheet Summary (as of December 31, 2024)")
    pdf.body_text(
        "Total assets at fiscal year-end were $18.4 billion, compared to $16.2 billion "
        "at December 31, 2023. Cash and cash equivalents were $3.62 billion, up from "
        "$2.85 billion at the prior year-end."
    )
    pdf.body_text(
        "Total liabilities were $8.9 billion. Long-term debt outstanding was $4.2 billion, "
        "unchanged from the prior year following the company's decision not to undertake "
        "additional debt issuance in fiscal 2024. Total shareholders equity was $9.5 billion."
    )

    # ── PAGE 2: R&D and Capital Expenditure ───────────────────────────
    pdf.chapter_title("Research and Development")
    pdf.body_text(
        "NexaCore invested $680 million in research and development in fiscal 2024, "
        "representing 14.0% of total revenue. This compares to R&D expenditure of "
        "$590 million in fiscal 2023. The increase reflects accelerated investment in "
        "artificial intelligence and machine learning capabilities across all product lines."
    )
    pdf.body_text(
        "Capital expenditure for fiscal 2024 totalled $420 million, up from $370 million "
        "in fiscal 2023. The majority of capital spending was directed toward data center "
        "infrastructure to support growth in the Cloud Services segment."
    )

    # ── PAGE 3: Subsidiary Performance ───────────────────────────────
    pdf.add_page()
    pdf.chapter_title("Subsidiary and Acquired Entity Performance")
    pdf.body_text(
        "DataStream Analytics, acquired by NexaCore in March 2023 for $520 million, "
        "contributed $185 million in revenue during fiscal 2024, ahead of the $150 million "
        "target established at acquisition. The integration is substantially complete and "
        "DataStream is expected to achieve breakeven operating income in fiscal 2025."
    )
    pdf.body_text(
        "NexaCore Asia Pacific Pte. Ltd., the company's Singapore-based regional subsidiary, "
        "reported revenue of $340 million in fiscal 2024, a 28.3% increase from "
        "$265 million in fiscal 2023, reflecting strong demand across the APAC region."
    )

    # ── PAGE 3: Outlook ───────────────────────────────────────────────
    pdf.chapter_title("Fiscal 2025 Guidance")
    pdf.body_text(
        "For fiscal year 2025, NexaCore expects total revenue in the range of "
        "$5.30 billion to $5.50 billion, representing growth of 9% to 13% over fiscal 2024. "
        "Operating income is expected to be between $1.25 billion and $1.35 billion. "
        "The company expects to generate free cash flow of approximately $1.1 billion."
    )
    pdf.body_text(
        "Capital expenditure for fiscal 2025 is expected to be in the range of "
        "$480 million to $520 million as the company continues to invest in "
        "infrastructure to support its cloud growth strategy."
    )

    pdf.output(str(out_path))
    print(f"PDF created: {out_path} ({out_path.stat().st_size:,} bytes)")
    return out_path


if __name__ == "__main__":
    out = Path("data/raw_pdfs/nexacore_annual_report_2024.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(out)
