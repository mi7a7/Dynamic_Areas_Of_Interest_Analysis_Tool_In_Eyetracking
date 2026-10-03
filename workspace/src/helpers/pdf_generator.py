import os
import pandas as pd
from workspace import config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from sklearn.linear_model import LinearRegression

from workspace.src.helpers.user_interface import UserInterface
from workspace.src.helpers.compute_module import aggregate_object_stats_all_users, compute_age_attention_correlation_per_user, compute_age_correlation_stats, compute_age_regression, generate_user_stats, summarize_logo_attention_all_users

userInterface = UserInterface()

def summarize_logo_attention(object_stats_df: pd.DataFrame, objects_data_df: pd.DataFrame):

    video_duration_s = objects_data_df["Frame_End_Time"].max() - objects_data_df["Frame_Start_Time"].min()
    
    total_objects = len(object_stats_df)
    seen_objects = object_stats_df["consciously_seen"].sum()
    percent_seen = 100 * seen_objects / total_objects if total_objects > 0 else 0

    seen_objects_df = object_stats_df[object_stats_df["consciously_seen"] == True]

    mean_gaze = seen_objects_df["gaze_time_s"].mean() if not seen_objects_df.empty else 0
    mean_ratio = seen_objects_df["view_time_ratio"].mean() if not seen_objects_df.empty else 0

    stats_summary = {
        "Video duration [s]": round(video_duration_s, 3),
        "Total objects": total_objects,
        "Number of objects consciously seen": seen_objects,
        "Percentage of objects seen (%)": round(percent_seen, 2),
        "Mean gaze time [s] (seen only)": round(mean_gaze, 3),
        "Mean view-time ratio (seen only)": round(mean_ratio, 3)
    }

    return stats_summary

def fit_image(path, max_w_cm=17, max_h_cm=12):
    ir = ImageReader(path)
    iw, ih = ir.getSize()
    aspect = ih / iw
    max_w = max_w_cm * cm
    max_h = max_h_cm * cm
    if max_w * aspect <= max_h:
        w = max_w
        h = max_w * aspect
    else:
        h = max_h
        w = max_h / aspect
    return Image(path, width=w, height=h)


def add_image(elements, img_path, style_text, max_w=17, max_h=11):
    if img_path and os.path.exists(img_path):
        elements.append(fit_image(img_path, max_w_cm=max_w, max_h_cm=max_h))
    else:
        elements.append(Paragraph(f"(Missing chart file: {img_path})", style_text))
    elements.append(Spacer(1, 12))


def generate_pdf_report_per_user(object_stats_df,
                                objects_data_df: pd.DataFrame,
                                users_info: pd.DataFrame,
                                user_id,
                                thumbnails_of_objects=None,    
                                thumbs_per_row=2):
    
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    output_path = os.path.join(config.RESULT_DIR, f"logo_analysis_report_user_{user_id}.pdf")
    
    summary_stats = summarize_logo_attention(object_stats_df, objects_data_df)

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    style_title  = ParagraphStyle(name="Title",  parent=styles["Heading1"], alignment=1, fontSize=16, spaceAfter=20)
    style_header = ParagraphStyle(name="Header", parent=styles["Heading2"], fontSize=13, spaceAfter=10)
    style_text   = ParagraphStyle(name="Text",   parent=styles["Normal"],  fontSize=11, spaceAfter=6, leading=14)
    style_cap    = ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=10, leading=12, alignment=1, spaceBefore=2)

    elements = []

    # ---- Title ----
    elements.append(Paragraph(f"Logo Attention Analysis Report - User {user_id}", style_title))
    elements.append(Spacer(1, 12))

    # ---- User info section ----
    elements.append(Paragraph("User information", style_header))
    user_info = users_info.loc[users_info["user_id"].astype(str) == user_id].iloc[0]

    row = [
        user_id,
        user_info.first_name,
        user_info.last_name,
        user_info.gender,
        user_info.age,
    ]
    
    header = ["User ID", "First Name", "Last Name", "Gender", "Age"]
    user_table = Table([header, row],
               colWidths=[3*cm, 4*cm, 4*cm, 3*cm, 2*cm])
    
    user_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eacdc2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))

    elements.append(user_table)
    elements.append(Spacer(1, 20))

    # ---- Summary stats ----
    elements.append(Paragraph("Summary statistics", style_header))

    body = [[k, v] for k, v in summary_stats.items()]
    table = Table(body, colWidths=[8*cm, 4*cm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # ---- Thumbnails ----
    if thumbnails_of_objects:
        elements.append(Paragraph("Thumbnails (per Object_ID)", style_header))
        elements.append(Spacer(1, 6))

        row = []
        for i, (obj_id, fpath) in enumerate(sorted(thumbnails_of_objects.items(), key=lambda x: x[0]), 1):
            if not os.path.exists(fpath):
                elements.append(Paragraph(f"(Missing thumbnail: {fpath})", style_text))
                continue

            img_w = 8.2 if thumbs_per_row == 2 else 5.5
            img_h = 6.0 if thumbs_per_row == 2 else 4.2
            img  = fit_image(fpath, max_w_cm=img_w, max_h_cm=img_h)
            cap  = Paragraph(f"Object_ID: {obj_id}", style_cap)
            cell = Table([[img], [cap]], colWidths=[(17*cm)/thumbs_per_row])
            cell.setStyle(TableStyle([
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ]))
            row.append(cell)

            if (i % thumbs_per_row) == 0:
                elements.append(Table([row], colWidths=[(17*cm)/thumbs_per_row]*thumbs_per_row, hAlign="CENTER"))
                elements.append(Spacer(1, 8))
                row = []

        if row:
            elements.append(Table([row], colWidths=[(17*cm)/len(row)]*len(row), hAlign="CENTER"))

        elements.append(Spacer(1, 20))

    # ---- Data table ----
    elements.append(Paragraph("Detailed results", style_header))

    header_style = ParagraphStyle(
        name="TableHeader",
        fontSize=9,
        alignment=1,  # CENTER
        leading=10
    )

    header =  [
        Paragraph("Object<br/>ID", header_style),
        Paragraph("Object<br/>duration [s]", header_style),
        Paragraph("Gaze<br/>time [s]", header_style),
        Paragraph("View time<br/>ratio", header_style),
        Paragraph("Consciously <br/> seen", header_style),
        Paragraph("User<br/>ID", header_style),
    ]

    body   = object_stats_df.round(3).astype(str).values.tolist()
    table  = Table([header] + body, colWidths=[(17*cm)/len(header)]*len(header))
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eacdc2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 25))

    # ---- Charts ----
    add_image(elements, f"{config.RESULT_DIR}/plots/pie_seen.png", style_text)
    add_image(elements, f"{config.RESULT_DIR}/plots/bar_gaze_time.png", style_text)
    add_image(elements, f"{config.RESULT_DIR}/plots/bar_view_time_ratio.png", style_text)
    add_image(elements, f"{config.RESULT_DIR}/plots/scatter_duration_vs_gaze.png", style_text)
    
    # ---- Build PDF ----
    doc.build(elements)

    userInterface.print_info(f"PDF report ready: {output_path}")


def generate_pdf_report_all_users(object_stats_all_users,
                                 objects_data_df: pd.DataFrame,
                                 users_info_df: pd.DataFrame,
                                 user_id,
                                output_dir,
                                thumbnails_of_objects=None,
                                thumbs_per_row=2):
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"logo_analysis_report_user_{user_id}.pdf")

    summary_stats_all_users = summarize_logo_attention_all_users(object_stats_all_users, objects_data_df)
    per_object_agg = aggregate_object_stats_all_users(object_stats_all_users)
    user_stats = generate_user_stats(users_info_df)

    per_user_age = compute_age_attention_correlation_per_user(
        object_stats_all_users
    )
    pearson_r, pearson_p, spearman_rho, spearman_p = compute_age_correlation_stats(per_user_age)
    slope, intercept, r2 = compute_age_regression(per_user_age)

    age_corr_stats = {
        "N users": len(per_user_age),
        "Pearson r (age vs seen_pct)": round(pearson_r, 3),
        "p-value (Pearson)": f"{pearson_p:.5f}",
        "Spearman rho (age vs seen_pct)": round(spearman_rho, 3),
        "p-value (Spearman)": f"{spearman_p:.5f}",
        "Regression slope β (age → seen_pct)": f"{slope:.3f}",
        "Regression intercept": f"{intercept:.3f}",
        "R² (regression)": f"{r2:.3f}",
    }
    

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    style_title  = ParagraphStyle(name="Title",  parent=styles["Heading1"], alignment=1, fontSize=16, spaceAfter=20)
    style_header = ParagraphStyle(name="Header", parent=styles["Heading2"], fontSize=13, spaceAfter=10)
    style_text   = ParagraphStyle(name="Text",   parent=styles["Normal"],  fontSize=11, spaceAfter=6, leading=14)
    style_cap    = ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=10, leading=12, alignment=1, spaceBefore=2)

    elements = []

    # ---- Title ----
    elements.append(Paragraph(f"Logo Attention Analysis Report - All Users", style_title))
    elements.append(Spacer(1, 12))

    # ---- User info section ----
    elements.append(Paragraph("All users information", style_header))

    rows = []
    for _, row in users_info_df.iterrows():
        rows.append([
            row["user_id"],
            row["first_name"],
            row["last_name"],
            row["gender"],
            row["age"],
        ])
        
    header = ["User ID", "First Name", "Last Name", "Gender", "Age"]
    user_table = Table([header] +rows,
               colWidths=[3*cm, 4*cm, 4*cm, 3*cm, 2*cm])
    
    user_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eacdc2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))

    elements.append(user_table)
    elements.append(Spacer(1, 20))

     # --- Users statistics ---
    elements.append(Paragraph("Users statistics", style_header))
    body = user_stats.values.tolist()

    users_stats_table = Table(body, colWidths=[8*cm, 4*cm])
    users_stats_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    elements.append(users_stats_table)
    elements.append(Spacer(1, 20))

    add_image(elements, f"{config.RESULT_DIR}/plots/hist_age_distribution.png", style_text)
    add_image(elements, f"{config.RESULT_DIR}/plots/pie_gender_distribution.png", style_text)

    
    # ---- Summary stats ----
    elements.append(Paragraph("Summary statistics", style_header))

    body = [[k, v] for k, v in summary_stats_all_users.items()]
    table = Table(body, colWidths=[8*cm, 4*cm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))


    # ---- Thumbnails ----
    if thumbnails_of_objects:
        elements.append(Paragraph("Thumbnails (per Object_ID)", style_header))
        elements.append(Spacer(1, 6))

        row = []
        for i, (obj_id, fpath) in enumerate(sorted(thumbnails_of_objects.items(), key=lambda x: x[0]), 1):
            if not os.path.exists(fpath):
                elements.append(Paragraph(f"(Missing thumbnail: {fpath})", style_text))
                continue

            img_w = 8.2 if thumbs_per_row == 2 else 5.5
            img_h = 6.0 if thumbs_per_row == 2 else 4.2
            img  = fit_image(fpath, max_w_cm=img_w, max_h_cm=img_h)
            cap  = Paragraph(f"Object_ID: {obj_id}", style_cap)
            cell = Table([[img], [cap]], colWidths=[(17*cm)/thumbs_per_row])
            cell.setStyle(TableStyle([
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ]))
            row.append(cell)

            if (i % thumbs_per_row) == 0:
                elements.append(Table([row], colWidths=[(17*cm)/thumbs_per_row]*thumbs_per_row, hAlign="CENTER"))
                elements.append(Spacer(1, 8))
                row = []

        if row:
            elements.append(Table([row], colWidths=[(17*cm)/len(row)]*len(row), hAlign="CENTER"))

        elements.append(Spacer(1, 20))


    # ---- Data table ----

    elements.append(Paragraph("Detailed results", style_header))

    header_style = ParagraphStyle(
        name="TableHeader",
        fontSize=9,
        alignment=1,  # CENTER
        leading=10
    )

    header =  [
        Paragraph("Object<br/>ID", header_style),
        Paragraph("Object<br/>duration [s]", header_style),
        Paragraph("Mean<br/>gaze time [s]", header_style),
        Paragraph("Mean view<br/>time ratio", header_style),
        Paragraph("Number<br/>of users", header_style),
        Paragraph("Consciously <br/> seen[%]", header_style),
    ]
    
    body   = per_object_agg.round(3).astype(str).values.tolist()
    table  = Table([header] + body, colWidths=[(17*cm)/len(header)]*len(header))
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eacdc2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 25))

    add_image(elements, f"{config.RESULT_DIR}/plots/pie_seen_all.png", style_text)
    add_image(elements, f"{config.RESULT_DIR}/plots/bar_mean_gaze_time.png", style_text)
    add_image(elements, f"{config.RESULT_DIR}/plots/bar_mean_view_time_ratio.png", style_text)
    add_image(elements, f"{config.RESULT_DIR}/plots/scatter_duration_vs_mean_gaze.png", style_text)
    add_image(elements, f"{config.RESULT_DIR}/plots/bar_seen_pct_by_gender.png", style_text)
    add_image(elements, f"{config.RESULT_DIR}/plots/box_view_ratio_by_gender.png", style_text)
    add_image(elements, f"{config.RESULT_DIR}/plots/bar_seen_pct_by_age_group.png", style_text)

    # --- Age–attention correlation ---
    elements.append(Paragraph("Age–attention correlation", style_header))

    body = [[k, v] for k, v in age_corr_stats.items()]
    age_table = Table(body, colWidths=[8*cm, 4*cm])
    age_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(age_table)
    elements.append(Spacer(1, 20))

    
    add_image(elements, f"{config.RESULT_DIR}/plots/age_vs_attention_regression.png", style_text)

    doc.build(elements)

    userInterface.print_info(f"PDF report ready: {output_path}")
    

