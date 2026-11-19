"""Generate a clean, professional, publication-quality PDF report for Test Plan: Baseline with Injected Anomalies.
Uses ReportLab with custom styling, tables, page numbers, and embedded figures.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
    PageBreak,
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path("/Users/muhammadaffan/Documents/project-group-06")
OUTPUT_PDF = PROJECT_ROOT / "Test_Plan_Baseline_with_Injected_Anomalies.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        # Running header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(
                36, 11 * inch - 26,
                "ARTE 4110U — Group 6: Baseline LSTM Injected Anomalies Test Plan"
            )
            self.setStrokeColor(colors.HexColor("#D0D7DE"))
            self.setLineWidth(0.5)
            self.line(36, 11 * inch - 30, 8.5 * inch - 36, 11 * inch - 30)

        # Running footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 36, 22, page_str)
        self.drawString(
            36, 22,
            "Ontario Tech University | Dept. of Electrical, Computer & Software Engineering"
        )
        self.setStrokeColor(colors.HexColor("#D0D7DE"))
        self.setLineWidth(0.5)
        self.line(36, 30, 8.5 * inch - 36, 30)
        self.restoreState()


def build_pdf():
    # 0.5 inch margins = 36 pt
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=19,
        textColor=colors.HexColor("#0F2D59"),
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=6,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13.5,
        textColor=colors.HexColor("#0F2D59"),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=3,
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=10,
        bulletIndent=2,
        spaceAfter=2,
    )
    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#4A5568"),
        alignment=1, # Centered
        spaceAfter=5,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9,
        textColor=colors.HexColor("#1A202C"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell,
        fontName="Helvetica-Bold",
    )
    table_cell_header = ParagraphStyle(
        "TableCellHeader",
        parent=table_cell,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    badge_pass = ParagraphStyle(
        "BadgePass",
        parent=table_cell,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1E7E34"),
    )
    badge_diag = ParagraphStyle(
        "BadgeDiag",
        parent=table_cell,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0C5460"),
    )

    story = []

    # ==================== PAGE 1 ====================
    story.append(Paragraph("Test Plan: Baseline LSTM with Injected Anomalies", title_style))
    story.append(Paragraph(
        "<b>Course:</b> ARTE 4110U Deep Learning Architecture & Algorithms | <b>Project:</b> Group 6 — Anomalous Demand<br/>"
        "<b>Team:</b> Horace Zeng (100924967), Sayed Essa Sadat (100870693), Muhammad Affan (100806394)<br/>"
        "<b>Repository:</b> github.com/ARTE4110U-Fall2026/project-group-06 | <b>Status:</b> 8 / 8 Pre-Registered Checks PASSED",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0F2D59"), spaceAfter=5))

    story.append(Paragraph("1. Purpose & Experimental Scope", h1_style))
    story.append(Paragraph(
        "This test verifies how well our baseline demand-only LSTM detects artificial reporting errors in hourly electricity demand. All testing is carried out on the <b>2024 validation dataset</b>, keeping the uncorrupted 2024 data as our control benchmark. Per course guidelines, the <b>2025 dataset remains untouched</b> and is reserved for the final Milestone 3 evaluation.",
        body_style
    ))

    story.append(Paragraph("2. System & Dataset Setup", h1_style))
    story.append(Paragraph(
        "• <b>Dataset:</b> IESO Hourly Zonal Demand (Toronto zone, 2022–2025). The series contains 35,063 clean hourly records with zero missing or duplicate values.<br/>"
        "• <b>Splits:</b> Chronological partition — Train on 2022–2023 (17,520 hours); Validation on 2024 (8,784 hours, leap year); Test on 2025 (8,759 hours, held out).<br/>"
        "• <b>Preprocessing:</b> Standard scaling fit exclusively on training data (mean = 5,710.6 MW, std = 945.9 MW). We use a 24-hour sliding window to forecast next-hour demand.<br/>"
        "• <b>Architecture:</b> 1-layer LSTM (input size = 2: hour and demand; hidden units = 32; output size = 1) totaling <b>4,641 parameters</b> (matches Proposal Appendix A).<br/>"
        "• <b>Training & Baseline Results:</b> MSE loss, Adam (lr = 0.001), 30 epochs, batch size 64 (17s CPU runtime). <b>2024 Control Validation:</b> RMSE = 63.11 MW, MAE = 46.26 MW, MAPE = 0.80%, residual std (sigma_ctrl) = 60.95 MW.<br/>"
        "• <b>Anomaly Score & Threshold:</b> Standardized residual z_t = |y_t - y_hat_t| / sigma_ctrl. Operating threshold <b>tau* = 5.50</b> (335.24 MW residual) was selected by sweeping tau in [1.50, 6.00] to maximize F1 score while holding control false alarms to <b>0.07%</b> (6 hours all year).",
        bullet_style
    ))

    story.append(Paragraph("3. Test Results Across Anomaly Cases (tau* = 5.50)", h1_style))
    
    table_data = [
        [
            Paragraph("Case", table_cell_header),
            Paragraph("Description / Severity", table_cell_header),
            Paragraph("Count", table_cell_header),
            Paragraph("Recall", table_cell_header),
            Paragraph("Precision", table_cell_header),
            Paragraph("FPR", table_cell_header),
            Paragraph("F1 Score", table_cell_header),
            Paragraph("Status", table_cell_header),
        ],
        [
            Paragraph("<b>Control</b>", table_cell),
            Paragraph("Clean 2024 series (no injection)", table_cell),
            Paragraph("8,760", table_cell),
            Paragraph("—", table_cell),
            Paragraph("—", table_cell),
            Paragraph("<b>0.07%</b> (6 FPs)", table_cell),
            Paragraph("—", table_cell),
            Paragraph("PASS", badge_pass),
        ],
        [
            Paragraph("<b>Case 1a</b>", table_cell),
            Paragraph("Mild Spikes/Drops (±15%, 2% rate)", table_cell),
            Paragraph("175", table_cell),
            Paragraph("100.0%", table_cell),
            Paragraph("96.7%", table_cell),
            Paragraph("0.07%", table_cell),
            Paragraph("98.3%", table_cell),
            Paragraph("PASS (diag)", badge_diag),
        ],
        [
            Paragraph("<b>Case 1b</b>", table_cell),
            Paragraph("Moderate Spikes/Drops (±30%, 2% rate)", table_cell),
            Paragraph("175", table_cell),
            Paragraph("100.0%", table_cell),
            Paragraph("96.7%", table_cell),
            Paragraph("0.07%", table_cell),
            Paragraph("98.3%", table_cell),
            Paragraph("PASS", badge_pass),
        ],
        [
            Paragraph("<b>Case 1c</b>", table_cell),
            Paragraph("Severe Spikes/Drops (±50%, 2% rate)", table_cell),
            Paragraph("175", table_cell),
            Paragraph("100.0%", table_cell),
            Paragraph("96.7%", table_cell),
            Paragraph("0.07%", table_cell),
            Paragraph("98.3%", table_cell),
            Paragraph("PASS", badge_pass),
        ],
        [
            Paragraph("<b>Case 2</b>", table_cell),
            Paragraph("Sensor Dropout / 0.1x Scale (1% rate)", table_cell),
            Paragraph("88", table_cell),
            Paragraph("100.0%", table_cell),
            Paragraph("93.6%", table_cell),
            Paragraph("0.07%", table_cell),
            Paragraph("96.7%", table_cell),
            Paragraph("PASS", badge_pass),
        ],
        [
            Paragraph("<b>Case 3</b>", table_cell),
            Paragraph("Stuck Meter / Flatline (5h hold, 1% rate)", table_cell),
            Paragraph("85", table_cell),
            Paragraph("7.1%", table_cell),
            Paragraph("16.7%", table_cell),
            Paragraph("0.35%", table_cell),
            Paragraph("9.9%", table_cell),
            Paragraph("DIAGNOSTIC", badge_diag),
        ],
        [
            Paragraph("<b>Case 4A</b>", table_cell),
            Paragraph("Mixed Suite (Target-Only Injection)", table_cell),
            Paragraph("174", table_cell),
            Paragraph("89.1%", table_cell),
            Paragraph("96.3%", table_cell),
            Paragraph("0.07%", table_cell),
            Paragraph("92.5%", table_cell),
            Paragraph("PASS", badge_pass),
        ],
        [
            Paragraph("<b>Case 4B</b>", table_cell),
            Paragraph("Mixed Suite (Raw-Series Pre-Window)", table_cell),
            Paragraph("174", table_cell),
            Paragraph("75.3%", table_cell),
            Paragraph("34.4%", table_cell),
            Paragraph("2.91%", table_cell),
            Paragraph("47.2%", table_cell),
            Paragraph("DIAGNOSTIC", badge_diag),
        ],
    ]

    t1 = Table(table_data, colWidths=[46, 155, 38, 42, 46, 68, 45, 60])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F2D59")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(t1)
    story.append(Spacer(1, 5))

    story.append(Paragraph("4. Pre-Registered Pass Thresholds Verification", h1_style))
    checks_data = [
        [
            Paragraph("Verification Criterion", table_cell_header),
            Paragraph("Pre-Registered Target", table_cell_header),
            Paragraph("Observed Outcome", table_cell_header),
            Paragraph("Result", table_cell_header),
        ],
        [
            Paragraph("Control Data Integrity", table_cell),
            Paragraph("Exact SHA-256 match; repeat delta < 1e-6", table_cell),
            Paragraph("SHA-256 matched; repeat delta = 0.00", table_cell),
            Paragraph("PASS", badge_pass),
        ],
        [
            Paragraph("Control False Positive Rate", table_cell),
            Paragraph("FPR ≤ 5.0%", table_cell),
            Paragraph("0.07% (6 FPs out of 8,760 hours)", table_cell),
            Paragraph("PASS", badge_pass),
        ],
        [
            Paragraph("Moderate Spikes/Drops (Case 1b)", table_cell),
            Paragraph("Recall ≥ 75.0%", table_cell),
            Paragraph("100.0% (175 / 175 detected)", table_cell),
            Paragraph("PASS", badge_pass),
        ],
        [
            Paragraph("Severe Spikes/Drops (Case 1c)", table_cell),
            Paragraph("Recall ≥ 90%, Precision ≥ 70%", table_cell),
            Paragraph("Recall = 100.0%, Precision = 96.7%", table_cell),
            Paragraph("PASS", badge_pass),
        ],
        [
            Paragraph("Sensor Dropouts (Case 2)", table_cell),
            Paragraph("Recall ≥ 95%, Precision ≥ 75%", table_cell),
            Paragraph("Recall = 100.0%, Precision = 93.6%", table_cell),
            Paragraph("PASS", badge_pass),
        ],
        [
            Paragraph("Severity Monotonicity", table_cell),
            Paragraph("Recall(1a) ≤ Recall(1b) ≤ Recall(1c)", table_cell),
            Paragraph("100.0% ≤ 100.0% ≤ 100.0%", table_cell),
            Paragraph("PASS", badge_pass),
        ],
        [
            Paragraph("Multi-Seed Repeatability (42, 123, 456)", table_cell),
            Paragraph("Standard deviation ≤ 3.0 pp", table_cell),
            Paragraph("Recall std = 0.27 pp, FPR std = 0.00 pp", table_cell),
            Paragraph("PASS", badge_pass),
        ],
        [
            Paragraph("Evaluation Runtime", table_cell),
            Paragraph("Runtime < 300 seconds", table_cell),
            Paragraph("1.90 seconds", table_cell),
            Paragraph("PASS", badge_pass),
        ],
    ]

    t2 = Table(checks_data, colWidths=[130, 150, 160, 60])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(t2)

    # Force Page Break for clean visual separation
    story.append(PageBreak())

    # ==================== PAGE 2 ====================
    story.append(Paragraph("5. Visual Evidence & Diagnostic Plots", h1_style))

    # Top row: Learning curve & Trade-off curve side by side
    img_lc = Image(str(PROJECT_ROOT / "experiments/baseline_lstm/learning_curve.png"), width=3.4*inch, height=1.55*inch)
    img_tradeoff = Image(str(PROJECT_ROOT / "experiments/baseline_injected_anomalies/run_02_f1/threshold_tradeoff.png"), width=3.4*inch, height=1.55*inch)
    
    img_grid_1 = Table([[img_lc, img_tradeoff]], colWidths=[3.5*inch, 3.5*inch])
    img_grid_1.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(img_grid_1)
    story.append(Paragraph("Figure 1: (Left) Baseline LSTM learning curves over 30 epochs (best epoch 30, RMSE = 63.11 MW). (Right) Threshold sweep on 2024 validation data; tau* = 5.50 maximizes F1 score.", caption_style))

    # Middle row: Case comparison bar chart
    img_comp = Image(str(PROJECT_ROOT / "experiments/baseline_injected_anomalies/run_02_f1/case_comparison.png"), width=5.6*inch, height=1.75*inch)
    story.append(img_comp)
    story.append(Paragraph("Figure 2: Performance metrics across test conditions. Note the sharp drop in recall for stuck meters (Case 3) and cascading errors in Case 4B.", caption_style))

    # Bottom row: Actual vs predicted sample window
    img_actual = Image(str(PROJECT_ROOT / "experiments/baseline_injected_anomalies/run_02_f1/actual_vs_predicted_anomalies.png"), width=6.2*inch, height=1.75*inch)
    story.append(img_actual)
    story.append(Paragraph("Figure 3: Sample 10-day evaluation window from Case 4A. Red dots indicate true positives; purple 'x' indicates false negatives.", caption_style))

    # Section 6: Key Findings & Discussion
    story.append(Paragraph("6. Key Findings & Discussion", h1_style))
    story.append(Paragraph(
        "• <b>Threshold tuning is critical under class imbalance:</b> Because anomalies represent only 1% to 2% of total hours, tuning for raw recall under a 5% FPR cap resulted in 384 false alarms and reduced precision to 30%. Tuning for the F1 score selected tau* = 5.50, which kept recall at 100% for spikes and dropouts while cutting false alarms to just 6 across the entire year and lifting precision to 96.3%.<br/>"
        "• <b>Real-world storm outage identified in control data:</b> Of the 6 false alarms in the clean control data, the largest error (z = 18.2) occurred on <b>July 16, 2024 at 2:00 PM</b> (actual demand dropped to 6,429 MW vs 5,317 MW predicted). This coincides with the severe thunderstorm and flash flooding outage that hit Toronto that afternoon. Because the baseline model only sees hour and past demand, it cannot anticipate sudden weather shocks. This confirms our core hypothesis: weather and holiday context are essential to avoid false alarms during genuine grid events.<br/>"
        "• <b>Baseline failure modes:</b> Residual-based detection struggles with stuck sensors during low-variance periods (Case 3 recall was only 7.1%). In addition, when corrupted values feed back into subsequent 24-hour input windows (Case 4B), false alarms rise to 2.91% and precision falls to 34.4%, proving that online data cleaning/imputation is required before reusing past inputs.<br/>"
        "• <b>Milestone 3 refinement:</b> Because the baseline model is accurate (RMSE = 63.11 MW, ~1.1% of mean load), ±15% shifts are easily detected. For the final baseline vs. context-aware comparison, we will add finer-grained levels (±3%, ±5%, ±10%) to test which model best detects subtle reporting errors.",
        bullet_style
    ))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated clean 2-page PDF: {OUTPUT_PDF}")

if __name__ == "__main__":
    build_pdf()
