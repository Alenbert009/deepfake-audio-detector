from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


def generate_detection_report(
    filename,
    prediction,
    confidence,
    probability_fake,
    probability_real,
    threshold,
    risk,
    features,
    waveform_path=None,
    spectrogram_path=None,
    gradcam_path=None,
):
    """
    Generate a PDF detection report.

    Returns:
        BytesIO: PDF file stored in memory.
    """

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        title="Deepfake Audio Detection Report",
        author="Deepfake Audio Detection System",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=10,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "NormalText",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
    )

    story = []
    # TITLE

    story.append(
        Paragraph(
            "Deepfake Audio Detection Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "AI-powered audio authenticity analysis",
            normal_style,
        )
    )

    story.append(Spacer(1, 15))

    # ---------------------------------------------------------
    # REPORT INFORMATION
    # ---------------------------------------------------------

    generated_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    report_info = [
        ["File Name", str(filename)],
        ["Generated On", generated_time],
        ["Detection Model", "CNN Spectrogram Classifier"],
        ["Input Representation", "Mel Spectrogram"],
    ]

    table = Table(
        report_info,
        colWidths=[150, 330],
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eeeeee")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(table)

    # ---------------------------------------------------------
    # DETECTION RESULT
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Detection Result",
            heading_style,
        )
    )

    result_data = [
        ["Prediction", str(prediction)],
        [
            "Confidence",
            f"{float(confidence) * 100:.2f}%",
        ],
        [
            "Fake Probability",
            f"{float(probability_fake) * 100:.2f}%",
        ],
        [
            "Real Probability",
            f"{float(probability_real) * 100:.2f}%",
        ],
        [
            "Decision Threshold",
            f"{float(threshold):.2f}",
        ],
        ["Risk Level", str(risk)],
    ]

    result_table = Table(
        result_data,
        colWidths=[180, 300],
    )

    prediction_upper = str(prediction).upper()

    if prediction_upper == "FAKE":
        prediction_color = colors.HexColor("#b91c1c")
    else:
        prediction_color = colors.HexColor("#15803d")

    result_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                (
                    "TEXTCOLOR",
                    (1, 0),
                    (1, 0),
                    prediction_color,
                ),
                ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
            ]
        )
    )

    story.append(result_table)

    # ---------------------------------------------------------
    # AUDIO FEATURES
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Audio Analysis",
            heading_style,
        )
    )

    feature_data = [
        ["Feature", "Value"],
        [
            "Duration",
            f"{features.get('duration', 'N/A')} seconds",
        ],
        [
            "Sample Rate",
            f"{features.get('sample_rate', 'N/A')} Hz",
        ],
        [
            "RMS Energy",
            str(features.get("rms", "N/A")),
        ],
        [
            "Zero Crossing Rate",
            str(features.get("zcr", "N/A")),
        ],
    ]

    feature_table = Table(
        feature_data,
        colWidths=[240, 240],
    )

    feature_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(feature_table)

    # ---------------------------------------------------------
    # ANALYSIS SUMMARY
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Analysis Summary",
            heading_style,
        )
    )

    if prediction_upper == "FAKE":

        summary = (
            f"The CNN classifier identified the uploaded audio as "
            f"potentially manipulated with a confidence of "
            f"{float(confidence) * 100:.2f}%. "
            f"The estimated probability of the audio being fake is "
            f"{float(probability_fake) * 100:.2f}%. "
            f"The assigned risk level is {risk}."
        )

    else:

        summary = (
            f"The CNN classifier identified the uploaded audio as "
            f"likely authentic with a confidence of "
            f"{float(confidence) * 100:.2f}%. "
            f"The estimated probability of the audio being real is "
            f"{float(probability_real) * 100:.2f}%. "
            f"The assigned risk level is {risk}."
        )

    story.append(
        Paragraph(
            summary,
            normal_style,
        )
    )

    # ---------------------------------------------------------
    # WAVEFORM
    # ---------------------------------------------------------

    if waveform_path:

        story.append(
            Paragraph(
                "Waveform Analysis",
                heading_style,
            )
        )

        story.append(
            Image(
                waveform_path,
                width=6.5 * inch,
                height=2.5 * inch,
            )
        )

    # ---------------------------------------------------------
    # SPECTROGRAM
    # ---------------------------------------------------------

    if spectrogram_path:

        story.append(
            Paragraph(
                "Mel Spectrogram",
                heading_style,
            )
        )

        story.append(
            Image(
                spectrogram_path,
                width=6.5 * inch,
                height=3.0 * inch,
            )
        )

    # ---------------------------------------------------------
    # GRAD-CAM
    # ---------------------------------------------------------

    if gradcam_path:

        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "Explainable AI — Grad-CAM",
                heading_style,
            )
        )

        story.append(
            Paragraph(
                "The Grad-CAM visualization highlights regions "
                "of the spectrogram that contributed most strongly "
                "to the CNN prediction.",
                normal_style,
            )
        )

        story.append(Spacer(1, 10))

        story.append(
            Image(
                gradcam_path,
                width=6.5 * inch,
                height=3.0 * inch,
            )
        )

    # ---------------------------------------------------------
    # DISCLAIMER
    # ---------------------------------------------------------

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This report represents the output "
            "of an AI-based classification model and should be "
            "treated as an automated assessment rather than "
            "absolute proof of audio authenticity.",
            normal_style,
        )
    )

    # ---------------------------------------------------------
    # BUILD PDF
    # ---------------------------------------------------------

    document.build(story)

    pdf_buffer.seek(0)

    return pdf_buffer
def generate_batch_detection_report(
    results,
    threshold
):
    """
    Generate a PDF report for batch deepfake audio detection.

    Parameters
    ----------
    results : list
        List of dictionaries containing:
            filename
            prediction
            confidence
            fake_probability
            real_probability
            risk

    threshold : float
        Detection threshold used for classification.

    Returns
    -------
    BytesIO
        PDF report stored in memory.
    """

    # ---------------------------------------------------------
    # PDF BUFFER
    # ---------------------------------------------------------

    pdf_buffer = BytesIO()

    # ---------------------------------------------------------
    # DOCUMENT
    # ---------------------------------------------------------

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        title="Deepfake Audio Batch Detection Report",
        author="Deepfake Audio Detection System",
    )

    # ---------------------------------------------------------
    # STYLES
    # ---------------------------------------------------------

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "BatchReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "BatchReportSubtitle",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        spaceAfter=15,
    )

    heading_style = ParagraphStyle(
        "BatchReportHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=14,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "BatchReportNormal",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
    )

    small_style = ParagraphStyle(
        "BatchReportSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
    )

    table_header_style = ParagraphStyle(
        "BatchTableHeader",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "BatchTableCell",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
    )

    # ---------------------------------------------------------
    # STORY
    # ---------------------------------------------------------

    story = []

    # ---------------------------------------------------------
    # TITLE
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Deepfake Audio Detection",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Batch Detection Report",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            "AI-powered audio authenticity analysis using "
            "a CNN-based spectrogram classifier.",
            subtitle_style,
        )
    )

    story.append(
        Spacer(1, 10)
    )

    # ---------------------------------------------------------
    # REPORT INFORMATION
    # ---------------------------------------------------------

    generated_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    total_files = len(results)

    fake_count = sum(
        1
        for result in results
        if str(result.get("prediction", "")).upper() == "FAKE"
    )

    real_count = sum(
        1
        for result in results
        if str(result.get("prediction", "")).upper() == "REAL"
    )

    high_risk_count = sum(
        1
        for result in results
        if str(result.get("risk", "")).upper() == "HIGH"
    )

    medium_risk_count = sum(
        1
        for result in results
        if str(result.get("risk", "")).upper() == "MEDIUM"
    )

    low_risk_count = sum(
        1
        for result in results
        if str(result.get("risk", "")).upper() == "LOW"
    )

    report_info = [
        ["Generated On", generated_time],
        ["Detection Model", "CNN Spectrogram Classifier"],
        ["Input Representation", "Mel Spectrogram"],
        ["Detection Threshold", f"{float(threshold):.2f}"],
        ["Total Files Analyzed", str(total_files)],
    ]

    info_table = Table(
        report_info,
        colWidths=[180, 300],
    )

    info_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#eeeeee"),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (1, 0),
                    (1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(
        info_table
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Batch Analysis Summary",
            heading_style,
        )
    )

    summary_data = [
        ["Metric", "Count"],
        ["Total Files", str(total_files)],
        ["Fake Audio", str(fake_count)],
        ["Real Audio", str(real_count)],
        ["High Risk", str(high_risk_count)],
        ["Medium Risk", str(medium_risk_count)],
        ["Low Risk", str(low_risk_count)],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[300, 180],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#374151"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (1, -1),
                    "CENTER",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(
        summary_table
    )

    # ---------------------------------------------------------
    # BATCH RESULTS
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Detection Results",
            heading_style,
        )
    )

    # Table header
    results_data = [
        [
            Paragraph("File", table_header_style),
            Paragraph("Prediction", table_header_style),
            Paragraph("Confidence", table_header_style),
            Paragraph("Fake Probability", table_header_style),
            Paragraph("Risk", table_header_style),
        ]
    ]

    # ---------------------------------------------------------
    # ADD EACH RESULT
    # ---------------------------------------------------------

    for result in results:

        filename = str(
            result.get(
                "filename",
                "Unknown"
            )
        )

        prediction = str(
            result.get(
                "prediction",
                "UNKNOWN"
            )
        ).upper()

        confidence = float(
            result.get(
                "confidence",
                0
            )
        )

        fake_probability = float(
            result.get(
                "fake_probability",
                0
            )
        )

        risk = str(
            result.get(
                "risk",
                "UNKNOWN"
            )
        ).upper()

        # ---------------------------------------------
        # Prediction color
        # ---------------------------------------------

        if prediction == "FAKE":

            prediction_color = colors.HexColor(
                "#b91c1c"
            )

        elif prediction == "REAL":

            prediction_color = colors.HexColor(
                "#15803d"
            )

        else:

            prediction_color = colors.black

        # ---------------------------------------------
        # Risk color
        # ---------------------------------------------

        if risk == "HIGH":

            risk_color = colors.HexColor(
                "#b91c1c"
            )

        elif risk == "MEDIUM":

            risk_color = colors.HexColor(
                "#b45309"
            )

        elif risk == "LOW":

            risk_color = colors.HexColor(
                "#15803d"
            )

        else:

            risk_color = colors.black

        # ---------------------------------------------
        # Create table row
        # ---------------------------------------------

        prediction_style = ParagraphStyle(
            "PredictionCell",
            parent=table_cell_style,
            textColor=prediction_color,
            fontName="Helvetica-Bold",
        )

        risk_style = ParagraphStyle(
            "RiskCell",
            parent=table_cell_style,
            textColor=risk_color,
            fontName="Helvetica-Bold",
        )

        results_data.append(
            [
                Paragraph(
                    filename,
                    table_cell_style,
                ),

                Paragraph(
                    prediction,
                    prediction_style,
                ),

                Paragraph(
                    f"{confidence * 100:.2f}%",
                    table_cell_style,
                ),

                Paragraph(
                    f"{fake_probability * 100:.2f}%",
                    table_cell_style,
                ),

                Paragraph(
                    risk,
                    risk_style,
                ),
            ]
        )

    # ---------------------------------------------------------
    # RESULTS TABLE
    # ---------------------------------------------------------

    results_table = Table(
        results_data,
        colWidths=[
            155,
            75,
            85,
            105,
            60,
        ],
        repeatRows=1,
    )

    results_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#374151"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (-1, -1),
                    "CENTER",
                ),
            ]
        )
    )

    story.append(
        results_table
    )

    # ---------------------------------------------------------
    # INTERPRETATION
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "Batch Interpretation",
            heading_style,
        )
    )

    if total_files > 0:

        fake_percentage = (
            fake_count / total_files
        ) * 100

        real_percentage = (
            real_count / total_files
        ) * 100

        interpretation = (
            f"A total of <b>{total_files}</b> audio file(s) "
            f"were analyzed. "
            f"<b>{fake_count}</b> file(s) were classified as "
            f"FAKE and <b>{real_count}</b> file(s) were classified "
            f"as REAL. "
            f"This corresponds to approximately "
            f"<b>{fake_percentage:.2f}%</b> FAKE and "
            f"<b>{real_percentage:.2f}%</b> REAL classifications."
        )

    else:

        interpretation = (
            "No audio files were successfully analyzed."
        )

    story.append(
        Paragraph(
            interpretation,
            normal_style,
        )
    )

    # ---------------------------------------------------------
    # RISK INTERPRETATION
    # ---------------------------------------------------------

    risk_summary = (
        f"Risk distribution: "
        f"{high_risk_count} HIGH, "
        f"{medium_risk_count} MEDIUM, and "
        f"{low_risk_count} LOW."
    )

    story.append(
        Spacer(1, 8)
    )

    story.append(
        Paragraph(
            risk_summary,
            normal_style,
        )
    )

    # ---------------------------------------------------------
    # DISCLAIMER
    # ---------------------------------------------------------

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This report represents the output "
            "of an AI-based classification model and should be "
            "treated as an automated assessment rather than "
            "absolute proof of audio authenticity. Model predictions "
            "are probabilistic and may be affected by audio quality, "
            "recording conditions, compression, and other factors.",
            small_style,
        )
    )

    # ---------------------------------------------------------
    # BUILD PDF
    # ---------------------------------------------------------

    document.build(
        story
    )

    # ---------------------------------------------------------
    # RESET BUFFER POSITION
    # ---------------------------------------------------------

    pdf_buffer.seek(0)

    return pdf_buffer