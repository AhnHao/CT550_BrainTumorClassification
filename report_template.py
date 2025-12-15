import io
import base64
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from PIL import Image

TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")

def _get_image_bytes(report_data, key_name):
    """
    Hàm helper để lấy bytes của hình ảnh.
    """
    name = report_data.get(key_name) or report_data.get(key_name.replace('_name', '_b64'), '')
    if name and not name.startswith('data:'):
        path = os.path.join(TMP_DIR, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
    
    candidates = [
        report_data.get('uploaded_img_b64', ''),
        report_data.get('gradcam_img_b64', ''),
        report_data.get(key_name, '')
    ]
    for b64 in candidates:
        if not b64:
            continue
        if b64.startswith('data:image'):
            b64 = b64.split(',', 1)[-1]
        
        padding = (4 - len(b64) % 4) % 4
        if padding:
            b64 += '=' * padding
            
        try:
            return base64.b64decode(b64)
        except Exception:
            continue
            
    ph = Image.new('RGB', (400, 400), (230, 230, 230))
    buf = io.BytesIO()
    ph.save(buf, format='JPEG', quality=70)
    return buf.getvalue()

def generate_report_pdf(report_data):
    buffer = io.BytesIO()
    
    # 1. TỐI ƯU HÓA LỀ TRANG (Giảm lề để có thêm chỗ trống)
    # Lề giảm xuống 10mm (xấp xỉ 0.4 inch)
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=10*mm, bottomMargin=10*mm)
    
    AVAILABLE_WIDTH = 180 * mm # 210 - 15 - 15
    
    styles = getSampleStyleSheet()
    normal = styles['Normal']
    normal.fontSize = 10
    normal.leading = 12 # Giảm khoảng cách dòng

    flow = []

    # --- Header (Tiêu đề) ---
    header_style = ParagraphStyle('header_title', 
                                  parent=styles['Heading1'], 
                                  alignment=TA_CENTER, 
                                  fontSize=16, # Giảm font size một chút
                                  textColor=colors.white)
    
    title_para = Paragraph("BRAIN TUMOR PREDICTION REPORT", header_style)
    
    header_table = Table([[title_para]], colWidths=[AVAILABLE_WIDTH])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#005b96')),
        ('TOPPADDING', (0,0), (-1,-1), 8), # Giảm padding
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    
    flow.append(header_table)
    flow.append(Spacer(1, 8)) # Giảm Spacer

    # --- Patient Info ---
    patient_table_data = [
        ['Patient Name:', report_data.get('patient_name', 'Unknown')],
        ['Age / Gender:', f"{report_data.get('patient_age', '?')} / {report_data.get('patient_gender', '?')}"],
        ['Doctor:', report_data.get('doctor_name', 'Unknown')],
        ['Model / Time:', f"{report_data.get('model', '?')} ({report_data.get('time_elapsed', '')})"],
    ]
    
    # Gộp thông tin để tiết kiệm dòng
    pt = Table(patient_table_data, colWidths=[40*mm, 100*mm], hAlign='CENTER')
    pt.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f2f5f7')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.grey),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    flow.append(pt)
    flow.append(Spacer(1, 8))

    # --- Prediction Result ---
    best = report_data.get('results', [("N/A", 0.0)])[0]
    pred_label = best[0].upper()
    pred_conf = f"{best[1]*100:.2f}%" if isinstance(best[1], (float,int)) else str(best[1])
    
    # Tạo nội dung HTML cho Paragraph
    # Sử dụng inline font tags để gọn gàng hơn
    pred_html = f"""
    <para align="center">
        <b>PREDICTION RESULT</b><br/><br/>
        <font size="14"><b>{pred_label}</b></font><br/>
        Confidence: <b>{pred_conf}</b><br/>
        <font size="8" color="grey"><i>(Clinical confirmation required)</i></font>
    </para>
    """
    
    pred_para = Paragraph(pred_html, styles['Normal'])

    pred_table = Table([[pred_para]], colWidths=[140*mm], hAlign='CENTER')
    pred_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5da')),
        ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    
    flow.append(pred_table)
    flow.append(Spacer(1, 10))

    # --- Images Section (SỬA LỖI KHUNG LỆCH) ---
    uploaded_bytes = _get_image_bytes(report_data, 'uploaded_img_name')
    gradcam_bytes = _get_image_bytes(report_data, 'gradcam_img_name')

    # 1. Giảm kích thước hiển thị ảnh
    IMG_DISPLAY_HEIGHT = 55 * mm 
    IMG_DISPLAY_WIDTH = 60 * mm
    
    def prepare_image(img_bytes):
        try:
            img = RLImage(io.BytesIO(img_bytes))
            # Chế độ preserveAspectRatio=True tự động scale
            img._restrictSize(IMG_DISPLAY_WIDTH, IMG_DISPLAY_HEIGHT)
            return img
        except:
            return Paragraph("No Image", normal)

    img_left = prepare_image(uploaded_bytes)
    img_gc = prepare_image(gradcam_bytes)

    lbl_left = Paragraph("Original Input", ParagraphStyle('lbl', parent=normal, alignment=TA_CENTER, fontSize=9))
    lbl_right = Paragraph("Grad-CAM Analysis", ParagraphStyle('lbl', parent=normal, alignment=TA_CENTER, fontSize=9))

    # 2. Tạo bảng với Row Heights cố định
    # Điều này đảm bảo khung bao quanh ô (cell) luôn bằng nhau bất kể tỷ lệ ảnh bên trong
    image_table = Table([
        [img_left, '', img_gc],
        [lbl_left, '', lbl_right]
    ], 
    colWidths=[IMG_DISPLAY_WIDTH + 4*mm, 8*mm, IMG_DISPLAY_WIDTH + 4*mm], 
    rowHeights=[IMG_DISPLAY_HEIGHT + 4*mm, None], # Cố định chiều cao hàng chứa ảnh
    hAlign='CENTER')
    
    image_table.setStyle(TableStyle([
        # Căn giữa nội dung trong ô
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        
        # Vẽ khung cho ô ảnh trái (0,0)
        ('BOX', (0,0), (0,0), 0.5, colors.grey),
        
        # Vẽ khung cho ô ảnh phải (2,0) - index cột 2 vì cột 1 là spacer
        ('BOX', (2,0), (2,0), 0.5, colors.HexColor('#f66')),
        
        # Padding để ảnh không dính sát khung
        ('TOPPADDING', (0,0), (-1,0), 2),
        ('BOTTOMPADDING', (0,0), (-1,0), 2),
    ]))
    
    flow.append(image_table)
    flow.append(Spacer(1, 10))

    # --- Probability Table ---
    flow.append(Paragraph("Detailed Probabilities", ParagraphStyle('h3', parent=styles['Heading3'], alignment=TA_CENTER, fontSize=11)))
    flow.append(Spacer(1, 4))
    
    probs = report_data.get('results', [])
    prob_table_data = [['Class', 'Probability']]
    for label, prob in probs:
        prob_table_data.append([label.capitalize(), f"{prob*100:.2f}%"])
    
    prob_tbl = Table(prob_table_data, colWidths=[80*mm, 40*mm], hAlign='CENTER')
    prob_tbl.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e9eef4')),
        ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9), # Giảm font size bảng
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    flow.append(prob_tbl)

    # --- Footer (GỘP DÒNG ĐỂ TIẾT KIỆM CHỖ) ---
    flow.append(Spacer(1, 15))
    
    # Tạo bảng footer 2 cột thay vì 2 dòng riêng biệt
    footer_data = [
        ["Prepared by: __________________", "Date: __________________"]
    ]
    footer_table = Table(footer_data, colWidths=[AVAILABLE_WIDTH/2, AVAILABLE_WIDTH/2])
    footer_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (0,0), 'LEFT'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))
    flow.append(footer_table)
    
    flow.append(Spacer(1, 8))
    flow.append(Paragraph("Report generated by AI Brain Tumor Classification System.", 
                          ParagraphStyle('foot', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))

    doc.build(flow)
    buffer.seek(0)
    return buffer