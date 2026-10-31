"""
Professional PDF Actuarial Report Generator for InsureAI Platform
Generates official underwriting reports, risk analysis, and counterfactual roadmaps.
"""

import io
from datetime import datetime
from typing import Any, Dict, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_policyholder_pdf_report(
    applicant_data: Dict[str, Any],
    quote_inr: float,
    model_name: str,
    bmi: float,
    cci: float,
    counterfactual_strategies: Optional[list] = None,
    cardholder_name: str = "Policyholder",
    policy_id: str = "PU-INS-2026-98604",
) -> bytes:
    """Generate a clean, high-impact PDF insurance intelligence report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
    )
    section_heading = ParagraphStyle(
        "SectionHead",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    badge_style = ParagraphStyle(
        "Badge",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#15803D"),
    )

    # Header Banner
    header_data = [
        [
            Paragraph("<b>INSUREAI INTELLIGENCE PLATFORM</b><br/><font size=8 color='#64748B'>Panjab University SSGICR Academic Research Framework</font>", title_style),
            Paragraph(f"<b>POLICY ID:</b> {policy_id}<br/><b>DATE:</b> {datetime.now().strftime('%d %b %Y')}<br/><b>STATUS:</b> <font color='#16A34A'>VERIFIED</font>", subtitle_style),
        ]
    ]
    t_header = Table(header_data, colWidths=[3.5 * inch, 3.5 * inch])
    t_header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceBefore=5, spaceAfter=15))

    # Actuarial Valuation Box
    val_data = [
        [
            Paragraph(f"<b>ACTUARIAL VALUATION FOR {cardholder_name.upper()}</b>", ParagraphStyle("H1", parent=body_style, fontName="Helvetica-Bold", textColor=colors.HexColor("#1E3A8A"))),
            Paragraph(f"<font size=16 color='#1E3A8A'><b>₹{quote_inr:,.2f}</b></font> / year", ParagraphStyle("H2", parent=body_style, alignment=2)),
        ],
        [
            Paragraph(f"Champion Inference Engine: <b>{model_name}</b> | Test Confidence: <b>86.95% R²</b> | Empirical Panjab Univ. Benchmark", body_style),
            Paragraph(f"Monthly Installment: <b>₹{quote_inr / 12:,.2f}</b> / mo", ParagraphStyle("H3", parent=body_style, alignment=2)),
        ]
    ]
    t_val = Table(val_data, colWidths=[4.8 * inch, 2.2 * inch])
    t_val.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#BFDBFE")),
        ("PADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_val)
    story.append(Spacer(1, 12))

    # Applicant Clinical Health Profile
    story.append(Paragraph("1. Physiological Vitals & Comorbidity Scoring", section_heading))
    bmi_category = "Normal (18.5 - 24.9)" if bmi < 25 else ("Overweight (25.0 - 29.9)" if bmi < 30 else "Obese (≥ 30.0)")
    vitals_table_data = [
        ["Attribute", "Measured Value", "Actuarial Weight", "Clinical Status"],
        ["Age", f"{applicant_data.get('Age', 0)} years", "Primary Cost Driver ($3,700 SHAP)", "Standard Risk"],
        ["Height & Weight", f"{applicant_data.get('Height', 0)} cm / {applicant_data.get('Weight', 0)} kg", "Physiological Baseline", "Logged"],
        ["Body Mass Index (BMI)", f"{bmi:.1f} kg/m²", "Second Cost Driver ($3,100 SHAP)", bmi_category],
        ["Charlson Comorbidity Index (CCI)", f"{cci:.1f}", "Multi-morbidity Proxy (Table I)", "Low Risk" if cci == 0 else "Elevated Risk"],
    ]
    t_vitals = Table(vitals_table_data, colWidths=[2.2 * inch, 1.8 * inch, 1.8 * inch, 1.4 * inch])
    t_vitals.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_vitals)
    story.append(Spacer(1, 12))

    # Medical Conditions & Underwriting Flags
    story.append(Paragraph("2. Underwriting Risk Checklist", section_heading))
    bp_str = "Yes (Elevated)" if applicant_data.get("BloodPressureProblems") == 1 else "No (Normal)"
    diab_str = "Yes (Type II / Diagnosed)" if applicant_data.get("Diabetes") == 1 else "No"
    trans_str = "Yes (Post-Transplant Care)" if applicant_data.get("AnyTransplants") == 1 else "No"
    chron_str = "Yes" if applicant_data.get("AnyChronicDiseases") == 1 else "No"
    canc_str = "Yes (Family History)" if applicant_data.get("HistoryOfCancerInFamily") == 1 else "No"
    surg_str = f"{applicant_data.get('NumberOfMajorSurgeries', 0)} Major Surgeries"

    med_data = [
        ["Hypertension (BP):", bp_str, "Organ Transplants:", trans_str],
        ["Diabetes Mellitus:", diab_str, "Chronic Diseases:", chron_str],
        ["Family Cancer History:", canc_str, "Past Surgical History:", surg_str],
    ]
    t_med = Table(med_data, colWidths=[2.0 * inch, 1.6 * inch, 2.0 * inch, 1.6 * inch])
    t_med.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_med)
    story.append(Spacer(1, 12))

    # Counterfactual Recourse Strategies (Table VII)
    if counterfactual_strategies:
        story.append(Paragraph("3. Actionable Counterfactual Recourse Roadmap (Table VII)", section_heading))
        rec_data = [["Recommended Feasible Strategy", "Target Modification", "Annual Rupee Savings (₹)"]]
        for strat in counterfactual_strategies:
            if isinstance(strat, dict):
                s_saving = strat.get("annual_saving", 0)
                s_name = strat.get("strategy_name", "Optimization")
                s_action = strat.get("action", "Adjust vitals")
                s_pct = strat.get("saving_percent", 0.0)
                if s_saving > 0:
                    rec_data.append([
                        s_name,
                        s_action,
                        f"₹{s_saving:,.2f} / yr ({s_pct:.1f}%)",
                    ])
            elif isinstance(strat, (list, tuple)) and len(strat) >= 4:
                # e.g. (name, action, notes, quote, saving)
                s_name = str(strat[0])
                s_action = str(strat[1])
                s_saving = float(strat[4]) if len(strat) > 4 else float(strat[3])
                if s_saving > 0:
                    rec_data.append([
                        s_name,
                        s_action,
                        f"₹{s_saving:,.2f} / yr",
                    ])
        if len(rec_data) > 1:
            t_rec = Table(rec_data, colWidths=[2.6 * inch, 3.0 * inch, 1.6 * inch])
            t_rec.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCFCE7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#15803D")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBF7D0")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t_rec)
            story.append(Spacer(1, 12))

    # Research Paper Citation & Academic Verification
    story.append(Paragraph("4. Academic Attribution & Legal Disclaimer", section_heading))
    disclaimer_text = (
        "This evaluation was computed using the research methodology published in <i>'Explainable Machine Learning for "
        "Health Insurance Cost Prediction: A Comparative Study of Tree-Based Ensembles with SHAP, ICE, and Counterfactual Analysis'</i> "
        "(Panjab University SSGICR, Hoshiarpur). Authors: <b>Sanjay Kumar</b> (Roll No: <code>SGM23366</code>) and "
        "<b>Dr. Sukhvinder Singh Bamber</b> (Faculty CSE, <code>ss.bamber@pu.ac.in</code>). "
        "<b>Regulatory Notice:</b> Figures represent empirical regression predictions derived from statistical cohorts and do not constitute a "
        "medical diagnosis or binding commercial underwriting insurance contract."
    )
    story.append(Paragraph(disclaimer_text, ParagraphStyle("Disc", parent=body_style, fontSize=7.5, leading=10, textColor=colors.HexColor("#64748B"))))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
