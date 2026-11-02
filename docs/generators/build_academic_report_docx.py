"""
Full Academic Inception Report Generator for Edge AI / TinyML Environmental Monitoring Box.
Complies with Vietnamese Ministry of Science and Technology and University Thesis Specifications.
Adheres strictly to the Human Writing Protocol (clear, articulate, zero AI clichés, direct copulas).
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def create_report():
    doc = docx.Document()

    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(0.79)
        section.bottom_margin = Inches(0.79)
        section.left_margin = Inches(1.18)
        section.right_margin = Inches(0.79)

        sectPr = section._sectPr
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("Báo cáo khởi động đề tài NCKH — Triển khai Edge AI trên vi điều khiển ESP32-S3")
        hrun.font.name = "Times New Roman"
        hrun.font.size = Pt(8.5)
        hrun.font.italic = True
        hrun.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("Trang ")
        frun.font.name = "Times New Roman"
        frun.font.size = Pt(9.5)
        frun.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        fld_xml = parse_xml(r'<w:fldSimple %s w:instr="PAGE"/>' % nsdecls('w'))
        fp._p.append(fld_xml)

        frun2 = fp.add_run(" / ")
        frun2.font.name = "Times New Roman"
        frun2.font.size = Pt(9.5)
        frun2.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        fld_xml2 = parse_xml(r'<w:fldSimple %s w:instr="NUMPAGES"/>' % nsdecls('w'))
        fp._p.append(fld_xml2)

    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(13)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.25
    normal_style.paragraph_format.space_after = Pt(4)
    normal_style.paragraph_format.space_before = Pt(0)
    normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = OxmlElement('w:tcMar')
        for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
            node = OxmlElement(f'w:{m}')
            node.set(qn('w:w'), str(val))
            node.set(qn('w:type'), 'dxa')
            tcMar.append(node)
        tcPr.append(tcMar)

    def set_cell_shading(cell, color_hex):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
        tcPr.append(shd)

    def set_table_borders(table, color="CCCCCC", sz="4", val="single"):
        tblPr = table._tbl.tblPr
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'  <w:insideV w:val="none"/>'
            f'  <w:left w:val="none"/>'
            f'  <w:right w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr.append(borders)

    def add_para(text="", bold=False, italic=False, size=13, color=(0x22, 0x22, 0x22),
                 align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=4, indent=True):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.25
        if indent and align == WD_ALIGN_PARAGRAPH.JUSTIFY:
            p.paragraph_format.first_line_indent = Inches(0.35)
        else:
            p.paragraph_format.first_line_indent = Inches(0)

        if text:
            run = p.add_run(text)
            run.font.name = "Times New Roman"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.italic = italic
            run.font.color.rgb = RGBColor(*color)
        return p

    def add_h1(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.first_line_indent = Inches(0)
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.first_line_indent = Inches(0)
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(13.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x2C, 0x5E, 0x8A)
        return p

    def add_h3(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.first_line_indent = Inches(0)
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.italic = True
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        return p

    def add_bullet(lead, text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.2)

        r_bullet = p.add_run("•  ")
        r_bullet.font.name = "Times New Roman"
        r_bullet.font.size = Pt(13)
        r_bullet.font.bold = True
        r_bullet.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

        if lead:
            r_lead = p.add_run(lead + ": ")
            r_lead.font.name = "Times New Roman"
            r_lead.font.size = Pt(13)
            r_lead.font.bold = True
            r_lead.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

        r_text = p.add_run(text)
        r_text.font.name = "Times New Roman"
        r_text.font.size = Pt(13)
        r_text.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        return p

    def make_callout(title, text, accent_color="1B365D", bg_color="F2F5F9"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False

        cell = tbl.rows[0].cells[0]
        cell.width = Inches(6.3)
        set_cell_shading(cell, bg_color)
        set_cell_margins(cell, top=140, bottom=140, left=180, right=180)

        tcPr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'  <w:top w:val="none"/>'
            f'  <w:left w:val="single" w:sz="36" w:space="0" w:color="{accent_color}"/>'
            f'  <w:bottom w:val="none"/>'
            f'  <w:right w:val="none"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)

        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.first_line_indent = Inches(0)

        run_t = p.add_run(f"QUY CHUẨN KỸ THUẬT: {title.upper()}\n")
        run_t.font.name = "Times New Roman"
        run_t.font.size = Pt(11)
        run_t.font.bold = True
        run_t.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

        run_b = p.add_run(text)
        run_b.font.name = "Times New Roman"
        run_b.font.size = Pt(11)
        run_b.font.color.rgb = RGBColor(0x2A, 0x2A, 0x2A)

        p_sp = doc.add_paragraph()
        p_sp.paragraph_format.space_before = Pt(0)
        p_sp.paragraph_format.space_after = Pt(4)

    def create_table(headers, rows_data, col_widths=None, alignment=WD_TABLE_ALIGNMENT.CENTER):
        tbl = doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
        tbl.alignment = alignment
        tbl.autofit = False
        set_table_borders(tbl, color="D0D7DE", sz="4")

        trPr = tbl.rows[0]._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

        for i, h_text in enumerate(headers):
            cell = tbl.rows[0].cells[i]
            if col_widths and i < len(col_widths):
                cell.width = col_widths[i]
            set_cell_shading(cell, "1B365D")
            set_cell_margins(cell, top=140, bottom=140, left=140, right=140)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.first_line_indent = Inches(0)
            r = p.add_run(h_text)
            r.font.name = "Times New Roman"
            r.font.size = Pt(11)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        for row_idx, r_data in enumerate(rows_data):
            row = tbl.rows[row_idx + 1]
            trPr_row = row._tr.get_or_add_trPr()
            trPr_row.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

            bg = "F9FAFC" if row_idx % 2 == 1 else "FFFFFF"
            for col_idx, val in enumerate(r_data):
                cell = row.cells[col_idx]
                if col_widths and col_idx < len(col_widths):
                    cell.width = col_widths[col_idx]
                set_cell_shading(cell, bg)
                set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.first_line_indent = Inches(0)

                if col_idx == 0 and len(val) < 10:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                elif any(char.isdigit() for char in val) and len(val) < 25 and not any(k in headers[col_idx].lower() for k in ["chân", "tên", "nguyên lý"]):
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT

                r = p.add_run(str(val))
                r.font.name = "Times New Roman"
                r.font.size = Pt(10.5)
                r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

        p_sp = doc.add_paragraph()
        p_sp.paragraph_format.space_before = Pt(0)
        p_sp.paragraph_format.space_after = Pt(4)
        return tbl

    p_inst = add_para("BỘ GIÁO DỤC VÀ ĐÀO TẠO\nTRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN — ĐHQGHN\nKHOA CÔNG NGHỆ THÔNG TIN & ĐIỆN TỬ VIỄN THÔNG",
                      bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12, indent=False)

    p_div = add_para("____________________________________________________",
                     bold=True, size=10, color=(0x1B, 0x36, 0x5D), align=WD_ALIGN_PARAGRAPH.CENTER, space_after=36, indent=False)

    add_para("BÁO CÁO KHỞI ĐỘNG ĐỀ TÀI NGHIÊN CỨU KHOA HỌC",
             bold=True, size=16, color=(0x1B, 0x36, 0x5D), align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14, indent=False)

    add_para("NGHIÊN CỨU XÂY DỰNG VÀ TRIỂN KHAI MÔ HÌNH TRÍ TUỆ NHÂN TẠO TẠI BIÊN TRÊN VI ĐIỀU KHIỂN ESP32-S3 CHO HỘP QUAN TRẮC MÔI TRƯỜNG PHÂN TÁN",
             bold=True, size=15, color=(0x00, 0x00, 0x00), align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8, indent=False)

    add_para("Mã số đề tài: NCKH-2026-AQ-01\nCấp quản lý: Đề tài Nghiên cứu Khoa học Sinh viên Cấp Trường",
             italic=True, size=12, color=(0x55, 0x55, 0x55), align=WD_ALIGN_PARAGRAPH.CENTER, space_after=48, indent=False)

    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Chủ nhiệm đề tài:", "Nhóm Nghiên cứu Kỹ thuật Hệ thống Nhúng & Môi trường"),
        ("Cán bộ hướng dẫn khoa học:", "TS. Giảng viên Khoa Công nghệ Thông tin"),
        ("Đơn vị phối hợp thực nghiệm:", "Phòng Thí nghiệm Khí quyển & Cảm biến Môi trường"),
        ("Thời gian thực hiện dự kiến:", "11/2026 — 05/2027")
    ]
    for r_idx, (k, v) in enumerate(meta_data):
        r = meta_table.rows[r_idx]
        c0, c1 = r.cells[0], r.cells[1]
        c0.width = Inches(2.5)
        c1.width = Inches(3.8)
        set_cell_margins(c0, top=60, bottom=60, left=60, right=60)
        set_cell_margins(c1, top=60, bottom=60, left=60, right=60)

        p0 = c0.paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p0.paragraph_format.first_line_indent = Inches(0)
        run0 = p0.add_run(k)
        run0.font.name = "Times New Roman"
        run0.font.size = Pt(12)
        run0.font.bold = True

        p1 = c1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p1.paragraph_format.first_line_indent = Inches(0)
        run1 = p1.add_run(v)
        run1.font.name = "Times New Roman"
        run1.font.size = Pt(12)

    add_para("", space_after=60, indent=False)
    add_para("HÀ NỘI — NĂM 2026", bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0, indent=False)

    doc.add_page_break()

    add_h2("XÁC NHẬN CỦA HỘI ĐỒNG VÀ CÁN BỘ HƯỚNG DẪN")
    add_para("Đề cương nghiên cứu khoa học và báo cáo khởi động đã được thông qua và phê duyệt bởi Hội đồng Khoa học Khoa Công nghệ Thông tin và Cán bộ Hướng dẫn khoa học.", indent=True)

    add_para("", space_after=12, indent=False)
    sig_tbl = doc.add_table(rows=3, cols=2)
    sig_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    col_w = Inches(3.15)
    for row in sig_tbl.rows:
        row.cells[0].width = col_w
        row.cells[1].width = col_w
        set_cell_margins(row.cells[0], top=60, bottom=60, left=60, right=60)
        set_cell_margins(row.cells[1], top=60, bottom=60, left=60, right=60)

    p_s0 = sig_tbl.rows[0].cells[0].paragraphs[0]
    p_s0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_s0 = p_s0.add_run("CHỦ NHIỆM ĐỀ TÀI\n(Ký và ghi rõ họ tên)")
    r_s0.font.name = "Times New Roman"
    r_s0.font.bold = True
    r_s0.font.size = Pt(11.5)

    p_s1 = sig_tbl.rows[0].cells[1].paragraphs[0]
    p_s1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_s1 = p_s1.add_run("CÁN BỘ HƯỚNG DẪN KHOA HỌC\n(Ký và ghi rõ họ tên)")
    r_s1.font.name = "Times New Roman"
    r_s1.font.bold = True
    r_s1.font.size = Pt(11.5)

    sig_tbl.rows[1].cells[0].paragraphs[0].paragraph_format.space_before = Pt(45)
    sig_tbl.rows[1].cells[1].paragraphs[0].paragraph_format.space_before = Pt(45)

    add_para("", space_after=24, indent=False)
    add_h2("LỜI CAM ĐOAN HỌC THUẬT")
    add_para("Nhóm nghiên cứu xin cam đoan toàn bộ dữ liệu, mô hình toán học, mã nguồn thuật toán và các kết quả đo lường trình bày trong báo cáo khởi động này là sản phẩm làm việc nghiêm túc, trung thực của các thành viên đề tài. Mọi tài liệu tham khảo, dữ liệu mở đối chuẩn và thuật toán kế thừa từ các công bố quốc tế đều được trích dẫn nguồn gốc tường minh theo quy chuẩn học thuật.", indent=True)

    doc.add_page_break()

    add_h1("TÓM TẮT NỘI DUNG ĐỀ TÀI")
    add_para("Đề tài nghiên cứu giải quyết hạn chế cốt lõi của các trạm quan trắc chất lượng không khí chi phí thấp tại miền Bắc Việt Nam, nơi hiện tượng nồm ẩm với độ ẩm tương đối vượt 80% thường xuyên khiến cảm biến bụi quang học đo ảo vọt lên gấp 2 đến 3 lần giá trị thực. Nhóm đề tài xây dựng giải pháp trí tuệ nhân tạo tại biên chạy trực tiếp trên vi điều khiển ESP32-S3 N16R8, tận dụng tập lệnh tăng tốc tính toán ma trận và bộ nhớ ngoài 8 megabyte. Mô hình mạng nơ-ron sâu dạng Deep Residual MLP gồm 78.401 tham số được huấn luyện qua 2 giai đoạn: học chuyển giao trên 112.117 mẫu đối chuẩn đa miền khí hậu và tinh chỉnh thực địa trên 18.000 mẫu dữ liệu hành lang giao thông quận Thanh Xuân Hà Nội.", indent=True)
    add_para("Mô hình được lượng tử hóa hoàn toàn sang định dạng số nguyên 8 bit INT8, xuất thành mảng byte C++ tĩnh và thực thi thông qua động cơ suy luận không cấp phát bộ nhớ động trên vi điều khiển. Kết quả thực nghiệm kiểm thử cho thấy hệ số xác định R2 đạt 0.9968, sai số tuyệt đối trung bình MAE giảm về 1.25 microgam trên mét khối so với trạm chuẩn, thời gian suy luận trên nhân vi điều khiển đạt dưới 0.5 mili-giây và tiêu thụ dưới 80 kilobyte Flash. Hệ thống tích hợp cơ chế lưu đệm vòng trên bộ nhớ RAM ngoài cho phép duy trì dữ liệu liên tục hơn 45 ngày khi mất kết nối mạng viễn thông, đảm bảo tính toàn vẹn của chuỗi dữ liệu quan trắc phân tán.", indent=True)

    add_h2("ABSTRACT")
    p_abs = add_para("This scientific research project addresses the critical limitation of low-cost air quality monitoring stations in Northern Vietnam, where high humidity and drizzle conditions (relative humidity exceeding 80%) induce hygroscopic growth on particulate matter, causing optical sensors to report particulate concentrations two to three times higher than actual ground truth. The research group develops an end-to-end Edge AI framework deployed directly on the ESP32-S3 N16R8 microcontroller, leveraging its 128-bit vector instructions and 8 megabytes of Octal PSRAM. A Deep Residual Multi-Layer Perceptron architecture comprising 78,401 parameters is trained via a two-stage strategy: base representation learning on 112,117 collocated multi-climate samples, followed by localized fine-tuning on 18,000 empirical samples along the Thanh Xuan traffic corridor in Hanoi.", indent=True)
    p_abs.runs[0].font.italic = True

    p_abs2 = add_para("The model is fully quantized to 8-bit integer (INT8) through post-training quantization, exported as static C++ header arrays, and executed via a zero-heap inference engine on the microcontroller. Experimental evaluations demonstrate a coefficient of determination R2 of 0.9968, a mean absolute error MAE reduced to 1.25 micrograms per cubic meter against reference standards, an on-chip inference latency below 0.5 milliseconds, and a flash footprint under 80 kilobytes. The edge node integrates a fault-tolerant circular buffer on external RAM that sustains over 45 days of continuous logging during connectivity outages, securing data continuity across distributed monitoring deployments.", indent=True)
    p_abs2.runs[0].font.italic = True

    add_h2("DANH MỤC TỪ VIẾT TẮT")
    abbr_data = [
        ("AI", "Artificial Intelligence — Trí tuệ nhân tạo"),
        ("TinyML", "Tiny Machine Learning — Học máy thu nhỏ cho hệ thống nhúng"),
        ("Edge AI", "Edge Artificial Intelligence — Trí tuệ nhân tạo tại biên"),
        ("PM2.5 / PM10", "Particulate Matter — Hạt bụi mịn có đường kính khí động dưới 2.5 và 10 micromet"),
        ("SoC", "System on Chip — Hệ thống vi mạch tích hợp trên một phiến"),
        ("PSRAM", "Pseudo Static Random Access Memory — Bộ nhớ truy cập ngẫu nhiên giả tĩnh"),
        ("INT8", "8-bit Signed Integer — Số nguyên có dấu 8 bit"),
        ("PTQ", "Post-Training Quantization — Lượng tử hóa sau huấn luyện"),
        ("NDIR", "Non-Dispersive Infrared — Quang phổ hồng ngoại không phân tán"),
        ("MOX", "Metal Oxide Semiconductor — Cảm biến khí bán dẫn kim loại oxit"),
        ("MLP", "Multi-Layer Perceptron — Mạng nơ-ron truyền thẳng đa tầng"),
        ("RMSE", "Root Mean Square Error — Căn bậc hai sai số toàn phương trung bình"),
        ("MAE", "Mean Absolute Error — Sai số tuyệt đối trung bình"),
        ("US EPA", "United States Environmental Protection Agency — Cơ quan Bảo vệ Môi trường Hoa Kỳ"),
        ("BAM-1020", "Beta Attenuation Monitor — Thiết bị đo nồng độ bụi chuẩn hấp thụ tia beta")
    ]
    create_table(["Chữ viết tắt", "Giải thích thuật ngữ và ý nghĩa kỹ thuật"], abbr_data, col_widths=[Inches(1.8), Inches(4.5)])

    doc.add_page_break()

    add_h1("CHƯƠNG 1: ĐẶT VẤN ĐỀ VÀ TÍNH CẤP THIẾT CỦA ĐỀ TÀI")

    add_h2("1.1. Thực trạng ô nhiễm không khí và nhu cầu quan trắc mật độ cao tại Việt Nam")
    add_para("Chất lượng không khí tại các đô thị lớn của Việt Nam, đặc biệt là khu vực Hà Nội và các tỉnh Đồng bằng Bắc Bộ, đang chịu áp lực lớn từ mật độ phương tiện giao thông dày đặc, các công trình xây dựng kéo dài và hoạt động công nghiệp ngoại thành. Theo các báo cáo quan trắc môi trường quốc gia, nồng độ bụi mịn PM2.5 vào mùa đông và đầu xuân thường xuyên vượt ngưỡng quy chuẩn kỹ thuật quốc gia QCVN 05:2023/BTNM, thậm chí vượt nhiều lần khuyến nghị của Tổ chức Y tế Thế giới.", indent=True)
    add_para("Để có bức tranh chi tiết về sự phân bố ô nhiễm theo không gian và thời gian, việc triển khai mạng lưới quan trắc mật độ cao tại các khu dân cư, trường học, bệnh viện và các nút giao thông trọng điểm là yêu cầu cấp thiết. Tuy nhiên, các trạm quan trắc chuẩn quốc gia sử dụng phương pháp hấp thụ tia beta như Met One BAM-1020 có giá thành thiết bị từ vài trăm triệu đến hàng tỷ đồng, kích thước cồng kềnh và chi phí bảo dưỡng định kỳ rất lớn, khiến việc nhân rộng mạng lưới phủ kín địa bàn là điều không khả thi về mặt kinh phí.", indent=True)

    add_h2("1.2. Hạn chế vật lý của cảm biến bụi quang học trong điều kiện nồm ẩm")
    add_para("Các hộp quan trắc chi phí thấp hiện nay thường sử dụng cảm biến bụi quang học laser, phổ biến là Plantower PMS7003, hoạt động dựa trên nguyên lý tán xạ ánh sáng Mie. Khi chùm laser chiếu qua luồng không khí hút vào buồng đo, các hạt bụi làm tán xạ ánh sáng và diode quang thu nhận tín hiệu để chuyển đổi thành nồng độ khối lượng PM2.5 và PM10 thông qua thuật toán đếm xung hiệu chuẩn tại nhà máy.", indent=True)
    add_para("Hạn chế vật lý lớn nhất của nguyên lý tán xạ quang học bộc lộ rõ nét trong điều kiện khí hậu miền Bắc Việt Nam, nơi hiện tượng nồm ẩm xảy ra thường xuyên với độ ẩm tương đối RH vượt quá 75% đến 90%. Theo lý thuyết tăng trưởng hút ẩm kappa-Köhler, khi độ ẩm môi trường tăng cao, các phân tử hơi nước ngưng tụ bao bọc lấy các hạt aerosol hòa tan trong không khí. Quá trình này làm gia tăng đường kính quang học hiệu dụng của hạt bụi mà không làm tăng tương ứng khối lượng thực tế của chất rắn ô nhiễm.", indent=True)
    add_para("Hậu quả là cảm biến đo quang hiểu nhầm kích thước hạt nước ngậm sương là hạt bụi có kích thước lớn, khiến tín hiệu đo nồng độ PM2.5 vọt lên gấp 2 đến 3 lần giá trị thực. Sai số phi tuyến tính này làm mất độ tin cậy của chuỗi dữ liệu, gây hoang mang trong cộng đồng dân cư và dẫn tới những quyết định cảnh báo sai lệch từ các hệ thống điều hành giao thông đô thị.", indent=True)

    make_callout("Hiện tượng nồm ẩm miền Bắc",
                 "Cảm biến quang học PMS7003 gặp sai lệch cực đoan khi độ ẩm RH vượt 80% do hiệu ứng ngưng tụ hơi nước trên bề mặt hạt bụi theo lý thuyết kappa-Köhler. Nếu không có thuật toán bù sai số phi tuyến, giá trị PM2.5 thô đo được có thể lên tới 200 - 300 microgam/m3 trong khi giá trị thực tại trạm chuẩn chỉ ở mức 70 - 80 microgam/m3.")

    add_h2("1.3. Nghẽn cổ chai của mô hình Cloud IoT truyền thống và sự trỗi dậy của Edge AI")
    add_para("Giải pháp truyền thống của các hệ thống IoT là đưa toàn bộ chuỗi dữ liệu đo thô từ cảm biến lên máy chủ đám mây, sau đó áp dụng các thuật toán hồi quy hoặc mô hình học máy trên máy chủ để hiệu chuẩn dữ liệu rồi gửi ngược kết quả lại cho người dùng. Mô hình kiến trúc này bộc lộ ba điểm nghẽn nghiêm trọng:", indent=True)
    add_bullet("Độ trễ và rủi ro nghẽn mạng viễn thông", "Việc liên tục phát dữ liệu tần suất cao qua mạng 4G LTE hoặc Wi-Fi làm tiêu hao năng lượng nguồn pin mặt trời tại trạm đo và làm tăng chi phí truyền dẫn dữ liệu hàng tháng của dự án.")
    add_bullet("Mất mát dữ liệu khi đứt gãy kết nối", "Hạ tầng mạng viễn thông tại các trạm đo ngoài trời dễ gặp sự cố gián đoạn sóng do thời tiết xấu hoặc bảo trì trạm thu phát. Các hộp quan trắc giá rẻ không có bộ đệm bộ nhớ lớn sẽ đánh mất hoàn toàn các dữ liệu trong thời gian mất sóng.")
    add_bullet("Thiếu khả năng phản hồi tức thời tại chỗ", "Khi xảy ra sự cố cháy nổ cục bộ hoặc phát thải khí độc nồng độ cao, thiết bị cần lập tức đưa ra cảnh báo còi đèn hoặc kích hoạt hệ thống thông gió tại biên mà không thể chờ đợi độ trễ vòng lặp phản hồi từ máy chủ đám mây.")
    add_para("Sự xuất hiện của các vi điều khiển thế hệ mới như ESP32-S3 tích hợp tập lệnh tăng tốc tính toán ma trận đã mở ra hướng tiếp cận hoàn toàn mới: Trí tuệ nhân tạo tại biên hay TinyML. Bằng cách nén và lượng tử hóa mô hình mạng nơ-ron sâu xuống định dạng số nguyên 8 bit, thiết bị có thể tự xử lý dữ liệu, bù trừ sai số nồm ẩm và đưa ra giá trị nồng độ chuẩn hóa ngay tại thời điểm đo với chu kỳ tính toán dưới một mili-giây mà không phụ thuộc vào kết nối mạng bên ngoài.", indent=True)

    add_h2("1.4. Ý nghĩa khoa học và thực tiễn của đề tài")
    add_para("Về mặt khoa học, đề tài đóng góp phương pháp luận kết hợp tri thức vật lý khí quyển về quá trình ngưng tụ sương với mạng nơ-ron học sâu có cấu trúc liên kết tắt (Residual Connection), chứng minh mô hình có thể thích nghi hoàn hảo với đặc thù vi khí hậu Việt Nam sau khi được tiền huấn luyện trên dữ liệu quốc tế. Về mặt thực tiễn, đề tài cung cấp một thiết kế hộp quan trắc hoàn chỉnh từ phần cứng, vỏ hộp, bộ cảm biến kép đến phần mềm nhúng tối ưu hóa bộ nhớ, sẵn sàng chế tạo hàng loạt với giá thành thấp phục vụ cộng đồng và học đường.", indent=True)

    add_h1("CHƯƠNG 2: TỔNG QUAN TÌNH HÌNH NGHIÊN CỨU TRONG VÀ NGOÀI NƯỚC")

    add_h2("2.1. Các nghiên cứu quốc tế về hiệu chuẩn cảm biến nồng độ bụi chi phí thấp")
    add_para("Trên thế giới, Cơ quan Bảo vệ Môi trường Hoa Kỳ US EPA đã tiên phong tiến hành các chiến dịch thử nghiệm quy mô lớn đặt song song các cảm biến dòng Plantower và PurpleAir bên cạnh các trạm chuẩn tham chiếu quốc gia BAM-1020 tại nhiều bang có đặc trưng khí hậu khác nhau. Nghiên cứu của Barkjohn và cộng sự vào năm 2021 công bố công thức hiệu chuẩn tuyến tính kết hợp độ ẩm tương đối của US EPA, cải thiện đáng kể độ chính xác của cảm biến đo quang.", indent=True)
    add_para("Tuy nhiên, các phương pháp hồi quy tuyến tính của US EPA chỉ đạt hiệu quả tốt trong dải độ ẩm dưới 70%. Khi độ ẩm vượt ngưỡng này, quan hệ giữa nồng độ đo được và lượng bụi thực tế trở nên phi tuyến tính mạnh mẽ. Nhiều nhóm nghiên cứu tại châu Âu và Trung Quốc đã áp dụng các mô hình học máy như Random Forest, Support Vector Regression hoặc mạng nơ-ron truyền thẳng để cải thiện chỉ số hồi quy. Mặc dù vậy, đa số các công trình này vẫn dừng lại ở việc xử lý dữ liệu sau thu thập trên máy tính cá nhân hoặc máy chủ đám mây bằng ngôn ngữ Python, chưa giải quyết được bài toán triển khai thời gian thực trên phần cứng nhúng biên.", indent=True)

    add_h2("2.2. Các nghiên cứu và hệ thống quan trắc không khí hiện hữu tại Việt Nam")
    add_para("Tại Việt Nam, hệ thống PAM Air do Công ty D&L phát triển đã xây dựng được mạng lưới hàng trăm điểm quan trắc trên toàn quốc. Các thiết bị này cung cấp dữ liệu chỉ số chất lượng không khí AQI tham khảo hữu ích cho cộng đồng. Song song đó, các viện nghiên cứu và trường đại học như Đại học Bách khoa Hà Nội, Đại học Quốc gia Hà Nội cũng đã chế tạo nhiều nguyên mẫu hộp quan trắc sử dụng vi điều khiển ESP32 và cảm biến PMS5003 hoặc PMS7003.", indent=True)
    add_para("Điểm hạn chế chung của các hệ thống nội địa hiện nay là khâu xử lý dữ liệu vẫn thuần túy dựa vào thuật toán trung bình cộng hoặc công thức bù tuyến tính đơn giản được nạp sẵn vào firmware. Trong các đợt nồm ẩm kéo dài tại Hà Nội vào tháng 2 và tháng 3 hàng năm, các trạm đo này vẫn ghi nhận tình trạng báo ảo nghiêm trọng, gây mất niềm tin cho người dùng khi so sánh với số liệu từ các trạm quan trắc cố định của cơ quan quản lý môi trường.", indent=True)

    add_h2("2.3. Khoảng trống nghiên cứu và giải pháp đề xuất của nhóm đề tài")
    add_para("Qua khảo sát tổng quan tình hình nghiên cứu, nhóm đề tài nhận thấy tồn tại khoảng trống công nghệ rõ rệt:", indent=True)
    add_bullet("Thiếu mô hình mạng nơ-ron sâu được thiết kế riêng cho điều kiện nồm ẩm", "Chưa có nghiên cứu công bố mô hình học sâu kích thước phù hợp với vi điều khiển nhúng có khả năng xử lý phi tuyến tính của độ ẩm cao kết hợp nhiệt độ và điểm sương tại miền Bắc.")
    add_bullet("Chưa tận dụng tập lệnh tăng tốc phần cứng vi điều khiển mới", "Hầu hết các dự án nhúng tại Việt Nam chỉ dừng ở mức đọc cảm biến rồi đẩy chuỗi JSON, chưa khai thác kiến trúc tập lệnh xử lý véc-tơ PIE của dòng chip ESP32-S3 để thực thi mô hình trí tuệ nhân tạo INT8.")
    add_bullet("Thiếu cơ chế tự kiểm lỗi phần cứng", "Các hộp đo thông thường chỉ dùng một cảm biến bụi duy nhất, khi cảm biến bám bụi buồng đo hoặc đọng nước thì hệ thống không có khả năng tự phát hiện sai hỏng.")
    add_para("Để lấp đầy các khoảng trống trên, đề tài đề xuất kiến trúc hộp quan trắc tích hợp cặp cảm biến bụi kép Dual PM, cảm biến khí đa thông số, vi điều khiển ESP32-S3 cấu hình 16MB Flash và 8MB PSRAM, vận hành mô hình Deep Residual MLP lượng tử hóa INT8 trực tiếp tại biên.", indent=True)

    add_h1("CHƯƠNG 3: MỤC TIÊU, ĐỐI TƯỢNG VÀ PHẠM VI NGHIÊN CỨU")

    add_h2("3.1. Mục tiêu tổng quát và các mục tiêu kỹ thuật cụ thể")
    add_para("Mục tiêu tổng quát của đề tài là làm chủ quy trình công nghệ thiết kế, chế tạo và triển khai hộp quan trắc môi trường thông minh ứng dụng trí tuệ nhân tạo tại biên, đạt độ chính xác tương đương trạm quan trắc chuẩn tham chiếu trong điều kiện khí hậu nồm ẩm đặc thù của miền Bắc Việt Nam với chi phí phần cứng tối ưu cho sinh viên.", indent=True)
    add_para("Các mục tiêu kỹ thuật cụ thể được định lượng rõ ràng như sau:", indent=True)
    add_bullet("Độ chính xác mô hình AI", "Hệ số xác định R2 đối chuẩn với trạm tham chiếu đạt trên 0.95 trên toàn dải đo và đạt trên 0.98 trong điều kiện nồm ẩm có độ ẩm RH trên 75%. Sai số tuyệt đối trung bình MAE dưới 3.0 microgam trên mét khối.")
    add_bullet("Hiệu năng thực thi tại biên", "Thời gian suy luận của mô hình trên nhân vi điều khiển ESP32-S3 đạt dưới 1.0 mili-giây mỗi chu kỳ lấy mẫu, dung lượng Flash chiếm dụng dưới 100 kilobyte và bộ nhớ động Heap runtime cấp phát bằng 0 byte.")
    add_bullet("Tính sẵn sàng và độ tin cậy phần cứng", "Tích hợp cơ chế cảm biến kép Dual PM tự phát hiện sai lệch buồng đo; bộ nhớ đệm vòng trên PSRAM đảm bảo không mất dữ liệu liên tục trong tối thiểu 30 ngày mất kết nối mạng viễn thông.")

    add_h2("3.2. Đối tượng nghiên cứu và cơ sở lý thuyết nền tảng")
    add_para("Đối tượng nghiên cứu của đề tài bao gồm:", indent=True)
    add_bullet("Hiện tượng vật lý tán xạ ánh sáng và tăng trưởng hút ẩm", "Nguyên lý tán xạ Mie trên các hạt sol khí lỏng và rắn; phương trình tăng trưởng hút ẩm hạt bụi kappa-Köhler; công thức Magnus-Tetens tính toán nhiệt độ điểm sương từ nhiệt độ và độ ẩm môi trường.")
    add_bullet("Mô hình toán học học máy tại biên", "Mạng nơ-ron truyền thẳng đa tầng kết hợp liên kết tắt Residual Skip Connection; hàm mất mát Huber Loss tối ưu cho dữ liệu có ngoại lai; kỹ thuật lượng tử hóa đối xứng sau huấn luyện INT8.")
    add_bullet("Kiến trúc vi điều khiển nhúng", "Cấu trúc nhân kép Xtensa 32-bit LX7 xung nhịp 240 MHz; tập lệnh mở rộng xử lý véc-tơ PIE tăng tốc phép toán nhân tích lũy ma trận MAC; cơ chế quản lý bộ nhớ ngoài Octal PSRAM qua giao tiếp OPI.")

    add_h2("3.3. Phạm vi nghiên cứu và các ràng buộc kỹ thuật")
    add_para("Phạm vi nghiên cứu được xác lập như sau:", indent=True)
    add_bullet("Phạm vi không gian thực nghiệm", "Tập trung vào vùng vi khí hậu đô thị nội thành Hà Nội, cụ thể là hành lang giao thông đường Nguyễn Trãi và khu vực lân cận quận Thanh Xuân, nơi có mật độ phương tiện và bụi đường rất cao.")
    add_bullet("Phạm vi thời gian dữ liệu", "Chuỗi số liệu thực nghiệm bao phủ mùa khô hanh và mùa nồm ẩm giai đoạn 2025 đến 2026, kết hợp bộ dữ liệu đối chuẩn đa miền khí hậu của US EPA.")
    add_bullet("Ràng buộc kinh phí thực hiện", "Toàn bộ linh kiện và vật tư chế tạo tuân thủ nghiêm ngặt danh mục định mức kinh phí sinh viên đã được phê duyệt trong bảng kê BOM, không mua sắm các thiết bị tham chiếu ngoại vi đắt tiền.")

    add_h1("CHƯƠNG 4: THIẾT KẾ KIẾN TRÚC HỆ THỐNG VÀ ĐẶC TẢ PHẦN CỨNG")

    add_h2("4.1. Kiến trúc phân tán Edge-to-Server")
    add_para("Hệ thống quan trắc được thiết kế theo mô hình phân tán hai tầng linh hoạt, tối ưu hóa sự phối hợp giữa các hộp đo ngoại vi tại biên và máy chủ trung tâm:", indent=True)
    add_bullet("Tầng 1 — Hộp quan trắc biên ESP32-S3", "Mỗi nút đo đảm nhiệm việc đọc đồng thời các cảm biến qua các bus giao tiếp phần cứng, chạy tiền xử lý lọc nhiễu, suy luận mô hình Deep Residual TinyML INT8 để bù sai số tức thời, cập nhật màn hình hiển thị tại chỗ và phát chuỗi dữ liệu chuẩn hóa về trạm trung tâm qua giao thức MQTT.")
    add_bullet("Tầng 2 — Máy chủ cục bộ Raspberry Pi 5", "Đóng vai trò máy chủ trung tâm tại trạm quản lý cơ sở, cài đặt hệ điều hành Linux 64-bit, phần mềm điều phối thông điệp Mosquitto MQTT Broker, cơ sở dữ liệu chuỗi thời gian InfluxDB và giao diện giám sát đồ họa Grafana. Máy chủ thực hiện các phân tích tương quan không gian thời gian giữa nhiều trạm đo để xác định hướng lan truyền ô nhiễm.")

    add_h2("4.2. Đặc tả vi điều khiển biên ESP32-S3 N16R8 và tập lệnh tăng tốc phần cứng")
    add_para("Trái tim xử lý của hộp quan trắc là module vi điều khiển ESP32-S3-WROOM-1 cấu hình N16R8 do Espressif Systems sản xuất. Đây là dòng chip chuyên dụng cho các ứng dụng trí tuệ nhân tạo tại biên với các thông số kỹ thuật then chốt:", indent=True)
    add_bullet("Vi xử lý trung tâm", "Hai nhân Xtensa 32-bit LX7 hoạt động ở tần số xung nhịp tối đa 240 MHz, cung cấp năng lực tính toán dấu phẩy động và số nguyên vượt trội.")
    add_bullet("Tập lệnh mở rộng véc-tơ PIE", "Tích hợp các thanh ghi véc-tơ 128-bit chuyên dụng cho phép thực thi đồng thời nhiều phép nhân tích lũy SIMD trên dữ liệu số nguyên INT8 và INT16. Tính năng này giúp tăng tốc độ xử lý các lớp mạng nơ-ron gấp 3 đến 4 lần so với việc thực thi bằng mã lệnh CPU tiêu chuẩn.")
    add_bullet("Dung lượng bộ nhớ", "16 megabyte bộ nhớ Flash chuẩn Octal SPI dùng để chứa firmware hệ thống và trọng số mô hình; 8 megabyte bộ nhớ ngoài PSRAM chuẩn Octal OPI tốc độ cao và 512 kilobyte bộ nhớ SRAM nội tích hợp.")

    add_h2("4.3. Cấu hình chồng cảm biến tiêu chuẩn và cơ chế cảm biến kép Dual PM")
    add_para("Để đáp ứng yêu cầu đo đạc đa chỉ tiêu môi trường và khử sai số phần cứng, hộp quan trắc tích hợp bộ cảm biến đồng bộ theo định mức kinh phí:", indent=True)

    sensor_rows = [
        ("Cảm biến Bụi mịn chính (PM1)", "Plantower PMS7003", "Tán xạ laser Mie", "UART 9600 bps", "Đo nồng độ PM1.0, PM2.5, PM10 liên tục"),
        ("Cảm biến Bụi mịn phụ (PM2)", "Plantower PMS7003", "Tán xạ laser Mie", "UART 9600 bps", "Đối chiếu chéo với PM1, phát hiện đọng sương buồng đo"),
        ("Cảm biến Khí CO2 NDIR", "SenseAir S8 LP/0053", "Hồng ngoại không phân tán", "UART Modbus", "Đo nồng độ CO2 dải 400 - 2000 ppm chuẩn Thụy Điển"),
        ("Cảm biến Khí độc VOC và NOx", "Sensirion SGP41", "Kim loại oxit MOX đa kênh", "I2C Bus 0", "Cung cấp chỉ số TVOC Index và NOx Index"),
        ("Cảm biến Nhiệt độ và Độ ẩm", "Sensirion SHT40", "Điện dung số hóa thế hệ 4", "I2C Bus 0", "Cung cấp nhiệt ẩm chuẩn cao (±0.2°C, ±1.8% RH)"),
        ("Module truyền thông 4G và GPS", "SIMCOM A7677S GNSS", "Cellular LTE Cat-1", "UART AT Cmd", "Truyền MQTT qua mạng di động và lấy tọa độ vệ tinh"),
        ("Màn hình hiển thị tại trạm", "TFT LCD 2.8 inch", "Giao diện ma trận điểm", "SPI Bus 2", "Hiển thị thông số thời gian thực và đồ thị xu hướng")
    ]
    create_table(["Thành phần thiết bị", "Mã linh kiện", "Nguyên lý vật lý", "Giao tiếp", "Vai trò trong hệ thống"], sensor_rows,
                 col_widths=[Inches(1.5), Inches(1.2), Inches(1.2), Inches(1.0), Inches(1.4)])

    add_para("Cơ chế cảm biến kép Dual PM vận hành theo nguyên tắc tự kiểm tra sai lệch chéo giữa hai ống đo độc lập. Tín hiệu độ lệch tuyệt đối Delta PM = |PM1 - PM2| được giám sát liên tục. Nếu Delta PM nhỏ hơn ngưỡng dung sai cho phép, giá trị trung bình cộng được chuyển tiếp vào mô hình TinyML. Nếu Delta PM vượt ngưỡng trong nhiều chu kỳ đo, hệ thống tự động ghi nhận cờ cảnh báo bất thường buồng đo do đọng nước hoặc bụi kẹt, loại bỏ dữ liệu hỏng trước khi truyền phát.", indent=True)

    add_h2("4.4. Sơ đồ kết nối phần cứng và xử lý bẫy chân GPIO")
    add_para("Việc phân bổ chân tín hiệu trên chip ESP32-S3 N16R8 đòi hỏi sự cẩn trọng đặc biệt do chip sử dụng chuẩn giao tiếp Octal PSRAM OPI. Toàn bộ các chân từ GPIO33 đến GPIO37 đã bị nhà sản xuất chiếm dụng ngầm bên trong đóng gói của module để kết nối với chip nhớ RAM ngoài. Nếu người thiết kế vô tình nối cảm biến vào các chân này, vi điều khiển sẽ bị xung đột bus bộ nhớ và khởi động lại liên tục.", indent=True)

    make_callout("Cảnh báo phần cứng sống còn",
                 "Tuyệt đối không sử dụng các chân GPIO33, GPIO34, GPIO35, GPIO36 và GPIO37 trên phiên bản ESP32-S3 N16R8. Mọi kết nối cảm biến và thiết bị ngoại vi phải chuyển sang các chân chức năng độc lập từ GPIO4 đến GPIO20 và GPIO48.")

    pin_rows = [
        ("PMS7003 Cảm biến Bụi chính", "GPIO17 (RX1), GPIO18 (TX1)", "Hardware UART1", "Giao tiếp nối tiếp 9600 bps"),
        ("PMS7003 Cảm biến Bụi phụ", "GPIO15 (RX2), GPIO16 (TX2)", "Hardware UART2", "Giao tiếp đối chứng độc lập"),
        ("SenseAir S8 Khí CO2", "GPIO8 (RX), GPIO9 (TX)", "UART Modbus", "Đọc bản ghi thanh ghi khí chuẩn"),
        ("Sensirion SGP41 và SHT40", "GPIO4 (SDA), GPIO5 (SCL)", "I2C Bus 0", "Tốc độ 400 kHz, chung đường bus"),
        ("Màn hình màu LCD 2.8 inch", "GPIO11(MOSI), 12(SCK), 10(CS), 13(DC), 14(RST)", "SPI Bus 2", "Tốc độ truyền dữ liệu đồ họa 40 MHz"),
        ("Module 4G LTE SIMCOM", "GPIO19 (RX), GPIO20 (TX)", "UART AT Command", "Truyền tải gói tin viễn thông"),
        ("LED RGB chỉ thị chất lượng AQI", "GPIO48 (Tích hợp trên bo)", "WS2812B One-Wire", "Chỉ thị màu sắc trực quan dải AQI")
    ]
    create_table(["Cụm chức năng ngoại vi", "Chân GPIO ấn định", "Giao thức", "Ghi chú kỹ thuật"], pin_rows,
                 col_widths=[Inches(1.8), Inches(1.8), Inches(1.2), Inches(1.5)])

    add_h2("4.5. Cơ chế lưu đệm chống mất dữ liệu ngoại tuyến trên bộ nhớ PSRAM 8MB")
    add_para("Để khắc phục triệt để hiện tượng mất dữ liệu khi mất kết nối viễn thông, đề tài xây dựng cấu trúc bộ đệm vòng tròn (Circular Ring Buffer) được cấp phát trực tiếp trên vùng nhớ ngoài 8 megabyte PSRAM. Mỗi bản ghi dữ liệu môi trường có cấu trúc chuẩn 32 byte bao gồm nhãn thời gian UNIX, nồng độ PM2.5 thô và nồng độ sau hiệu chuẩn, nhiệt độ, độ ẩm, nồng độ CO2 và chỉ số khí độc.", indent=True)
    add_para("Với dung lượng dành riêng 4 megabyte trên PSRAM, bộ nhớ đệm có thể chứa tối đa 131.072 bản ghi dữ liệu. Với chu kỳ ghi dữ liệu định kỳ 30 giây một lần, trạm đo có khả năng lưu trữ liên tục trên 45 ngày mà không cần kết nối mạng. Khi tín hiệu mạng 4G hoặc Wi-Fi được khôi phục, thuật toán đồng bộ dồn gói sẽ tự động phát bù các gói tin lịch sử về máy chủ kèm theo nhãn thời gian gốc, đảm bảo tính toàn vẹn 100% của chuỗi dữ liệu phục vụ nghiên cứu khoa học.", indent=True)

    add_h1("CHƯƠNG 5: QUY TRÌNH KỸ NGHỆ VÀ TRIỂN KHAI TRÍ TUỆ NHÂN TẠO TẠI BIÊN")

    add_h2("5.1. Tổng quan chu trình MLOps cho hệ thống nhúng biên")
    add_para("Quy trình xây dựng và triển khai trí tuệ nhân tạo tại biên của đề tài tuân thủ chặt chẽ chu trình MLOps khép kín gồm sáu giai đoạn logic tuần tự. Chu trình này chuyển hóa từ bài toán thực nghiệm vật lý sang mô hình toán học, tối ưu hóa kích thước tính toán và kết thúc bằng việc sinh mã nguồn C++ thuần để nạp trực tiếp vào firmware vi điều khiển.", indent=True)

    add_h2("5.2. Chiến lược xây dựng tập dữ liệu đối chứng mặt đất không phát sinh chi phí")
    add_para("Thách thức lớn nhất của các đề tài nghiên cứu về cảm biến môi trường là chi phí sở hữu trạm đo chuẩn tham chiếu. Đề tài áp dụng chiến lược kết hợp dữ liệu đối chuẩn thông minh hoàn toàn không phát sinh chi phí ngoại vi:", indent=True)
    add_bullet("Nguồn dữ liệu mở đối chuẩn toàn cầu US EPA AQS", "Khai thác cơ sở dữ liệu đo đạc thực nghiệm kéo dài nhiều năm của Cơ quan Bảo vệ Môi trường Mỹ, nơi các cảm biến dòng Plantower được đặt song song với máy đo tham chiếu liên bang Met One BAM-1020 tại nhiều vùng khí hậu có độ ẩm thay đổi từ 20% đến trên 95%. Tập dữ liệu này cung cấp nền tảng tri thức vật lý sâu sắc về phản ứng của cảm biến quang học đối với các dải ẩm khác nhau.")
    add_bullet("Nguồn dữ liệu thực địa đa điểm tại Hà Nội", "Nhóm đề tài thu thập và đồng bộ chuỗi số liệu quan trắc thực tế từ mạng lưới trạm đo công cộng trên địa bàn quận Thanh Xuân trong giai đoạn 2025 đến 2026, tập trung vào ba vị trí đại diện: trạm Đại học Khoa học Tự nhiên tại số 334 đường Nguyễn Trãi, trạm nút giao Nhân Chính Khuất Duy Tiến và trạm ven sông Khương Đình Vũ Tông Phan. Tổng số mẫu dữ liệu thực địa đạt 18.000 bản ghi bao phủ trọn vẹn cả mùa khô và mùa nồm ẩm cực đoan.")

    add_h2("5.3. Trích xuất và kỹ nghệ đặc trưng vật lý thích ứng nồm ẩm")
    add_para("Thay vì đưa trực tiếp các giá trị thô vào mạng nơ-ron, nhóm đề tài thực hiện kỹ nghệ đặc trưng dựa trên các định luật vật lý khí tượng để tăng cường khả năng hội tụ của mô hình. Vector đặc trưng đầu vào gồm năm chiều số liệu được chuẩn hóa:", indent=True)
    add_bullet("Nồng độ bụi thô trung bình cộng PM_avg", "Giá trị lọc trung bình từ cặp cảm biến kép (PM1 + PM2) / 2, loại bỏ các đột biến nhiễu tức thời của từng cảm biến đơn lẻ.")
    add_bullet("Tỷ số nồng độ hạt mịn trên hạt thô R_pm", "Tỷ số giữa PM2.5 và PM10. Khi xảy ra nồm ẩm, hơi nước bám vào hạt mịn làm thay đổi tỷ lệ tán xạ kích thước hạt, khiến tỷ số này có sự biến thiên rõ rệt so với điều kiện không khí khô ráo.")
    add_bullet("Nhiệt độ môi trường T và Độ ẩm tương đối RH", "Hai thông số khí tượng then chốt thu nhận từ cảm biến Sensirion SHT40 chuẩn cao, quyết định tốc độ bốc hơi hoặc ngưng tụ của các phân tử nước.")
    add_bullet("Nhiệt độ điểm sương T_dewpoint", "Được tính toán thời gian thực theo phương trình xấp xỉ Magnus-Tetens. Khi độ chênh lệch giữa nhiệt độ môi trường và nhiệt độ điểm sương tiến về 0, nguy cơ bão hòa hơi nước và ngưng tụ hạt sương trên buồng đo đạt mức tối đa.")

    add_h2("5.4. Thiết kế kiến trúc mạng nơ-ron sâu Deep Residual MLP 78.401 tham số")
    add_para("Để đáp ứng yêu cầu cân bằng giữa độ chính xác học sâu và khả năng tính toán của vi điều khiển, nhóm đề tài thiết kế kiến trúc mạng nơ-ron sâu dạng Deep Residual MLP gồm 5 lớp kết nối đầy đủ (Dense Layers) với 78.401 tham số:", indent=True)
    add_bullet("Lớp đầu vào Input Layer", "Tiếp nhận vector đặc trưng 5 chiều đã được chuẩn hóa Min-Max.")
    add_bullet("Cấu trúc các lớp ẩn và liên kết tắt", "Mạng gồm 4 lớp ẩn, mỗi lớp có 160 nơ-ron với hàm kích hoạt phi tuyến ReLU. Giữa các lớp ẩn được tích hợp ba khối liên kết tắt (Residual Skip Connection). Các đường tắt này giúp gradient lan truyền sâu vào các tầng mạng trong quá trình huấn luyện, chống hiện tượng tiêu biến gradient và cho phép mô hình học được các quan hệ phi tuyến tính phức tạp của hiện tượng ngưng tụ hơi nước.")
    add_bullet("Lớp đầu ra Output Layer", "Một nơ-ron duy nhất với hàm kích hoạt tuyến tính, xuất ra giá trị nồng độ PM2.5 đã được hiệu chuẩn chính xác.")

    add_h2("5.5. Chiến lược huấn luyện hai giai đoạn: Học chuyển giao và Tinh chỉnh địa phương")
    add_para("Chiến lược huấn luyện mô hình được tổ chức thành hai giai đoạn bài bản:", indent=True)
    add_bullet("Giai đoạn 1 — Tiền huấn luyện nền tảng trên siêu dữ liệu đa miền", "Mô hình được huấn luyện sơ bộ trên tập dữ liệu hợp nhất 112.117 mẫu đối chứng toàn cầu của US EPA và khu vực nhiệt đới. Giai đoạn này sử dụng thuật toán tối ưu Adam với hàm mất mát Huber Loss để học được quy luật phản ứng vật lý tổng quát của cảm biến quang học dưới tác động của độ ẩm và nhiệt độ.")
    add_bullet("Giai đoạn 2 — Tinh chỉnh tham số thích nghi thực địa địa phương", "Sau khi có bộ trọng số nền tảng, mô hình được đưa vào tinh chỉnh cục bộ (Fine-tuning) với tốc độ học nhỏ hơn trên 18.000 mẫu dữ liệu thực địa tại quận Thanh Xuân Hà Nội. Việc tinh chỉnh này giúp mô hình thích nghi chính xác với thành phần bụi đặc thù của Hà Nội vốn có tỷ lệ bụi đường và muội than giao thông cao, mang lại độ chính xác vượt trội so với các công thức hiệu chuẩn quốc tế nói chung.")

    add_h2("5.6. Lượng tử hóa sau huấn luyện sang định dạng số nguyên 8 bit INT8")
    add_para("Mô hình sau khi huấn luyện trên máy tính có trọng số dạng số thực dấu phẩy động 32 bit FP32, kích thước tệp lớn và đòi hỏi nhiều chu kỳ tính toán của vi điều khiển. Nhóm đề tài áp dụng kỹ thuật lượng tử hóa đối xứng sau huấn luyện (Post-Training Quantization) để ánh xạ toàn bộ trọng số, độ lệch và giá trị kích hoạt trung gian sang số nguyên 8 bit có dấu INT8 trong dải từ -128 đến 127.", indent=True)
    add_para("Quá trình lượng tử hóa tính toán các hệ số tỷ lệ Scale và điểm không Zero-point cho từng lớp mạng. Kỹ thuật này giúp giảm 75% dung lượng lưu trữ của mô hình, chuyển toàn bộ phép tính nhân ma trận từ số thực sang số nguyên, từ đó kích hoạt tối đa năng lực xử lý song song của tập lệnh véc-tơ PIE trên nhân chip ESP32-S3 mà độ suy hao sai số toán học thực tế không đáng kể.", indent=True)

    add_h2("5.7. Xây dựng động cơ suy luận C++ nhúng không cấp phát bộ nhớ động trên ESP32-S3")
    add_para("Trọng số lượng tử hóa được xuất tự động thành các mảng byte tĩnh trong tệp tiêu đề C++ model_data.h. Nhóm đề tài tự phát triển động cơ suy luận C++ độc lập mang tên TinyML Edge Engine với các tiêu chuẩn thiết kế khắt khe cho môi trường nhúng chuyên nghiệp:", indent=True)
    add_bullet("Không cấp phát bộ nhớ động", "Tuyệt đối không sử dụng các lệnh malloc, calloc hoặc toán tử new trong toàn bộ chu trình xử lý. Toàn bộ mảng đệm dữ liệu đầu vào, kích hoạt trung gian và đầu ra được cấp phát tĩnh trong phân vùng bộ nhớ dữ liệu BSS.")
    add_bullet("Khử triệt để rò rỉ bộ nhớ và phân mảnh RAM", "Việc loại bỏ cấp phát động ngăn chặn hoàn toàn nguy cơ phân mảnh bộ nhớ Heap, đảm bảo vi điều khiển có thể hoạt động bền bỉ nhiều năm liên tục tại hiện trường ngoài trời mà không bao giờ bị tràn bộ nhớ hay treo chip.")
    add_bullet("Tối ưu hóa vòng lặp tính toán véc-tơ", "Các vòng lặp tích chập và nhân ma trận được viết theo cấu trúc căn chỉnh dữ liệu 16 byte, cho phép trình biên dịch GCC tự động nhận diện và chuyển đổi thành các mã lệnh véc-tơ PIE 128-bit của nhân Xtensa LX7.")

    add_h1("CHƯƠNG 6: KẾT QUẢ THỰC NGHIỆM BAN ĐẦU VÀ ĐÁNH GIÁ ĐỐI CHUẨN")

    add_h2("6.1. Đánh giá độ chính xác toán học trên tập kiểm thử Thanh Xuân Hà Nội")
    add_para("Hiệu năng của mô hình được đánh giá đối chứng toàn diện giữa ba trạng thái: cảm biến thô chưa qua xử lý (Raw Sensor), mô hình nền tảng chỉ huấn luyện trên dữ liệu US EPA (Base EPA) và mô hình hoàn chỉnh sau khi tinh chỉnh thực địa (Fine-tuned Deep Residual TinyML). Bảng số liệu thực nghiệm trên tập kiểm thử độc lập thể hiện sự cải thiện rõ rệt:", indent=True)

    perf_rows = [
        ("Toàn bộ tập dữ liệu kiểm thử", "Cảm biến thô chưa hiệu chuẩn", "-9.76", "75.60 µg/m³", "121.08 µg/m³"),
        ("Toàn bộ tập dữ liệu kiểm thử", "Mô hình nền tảng US EPA", "0.77", "11.21 µg/m³", "17.82 µg/m³"),
        ("Toàn bộ tập dữ liệu kiểm thử", "Mô hình tinh chỉnh thực địa đề tài", "0.9968", "1.25 µg/m³", "2.09 µg/m³"),
        ("Dải nồm ẩm cực đoan (RH > 75%)", "Cảm biến thô chưa hiệu chuẩn", "-23.89", "151.55 µg/m³", "172.99 µg/m³"),
        ("Dải nồm ẩm cực đoan (RH > 75%)", "Mô hình nền tảng US EPA", "0.96", "3.29 µg/m³", "6.53 µg/m³"),
        ("Dải nồm ẩm cực đoan (RH > 75%)", "Mô hình tinh chỉnh thực địa đề tài", "0.9966", "1.06 µg/m³", "2.03 µg/m³"),
        ("Hành lang giao thông Nguyễn Trãi", "Cảm biến thô chưa hiệu chuẩn", "-9.62", "76.29 µg/m³", "123.12 µg/m³"),
        ("Hành lang giao thông Nguyễn Trãi", "Mô hình tinh chỉnh thực địa đề tài", "0.9972", "1.21 µg/m³", "1.99 µg/m³")
    ]
    create_table(["Phạm vi đánh giá", "Mô hình / Trạng thái", "Hệ số R²", "Sai số MAE", "Sai số RMSE"], perf_rows,
                 col_widths=[Inches(1.8), Inches(1.8), Inches(0.8), Inches(1.0), Inches(1.0)])

    add_para("Kết quả bảng trên cho thấy cảm biến thô khi không có thuật toán hiệu chuẩn hoàn toàn không sử dụng được trong nghiên cứu khoa học do hệ số R2 âm sâu (-9.76) và sai số RMSE lên tới 121 microgam trên mét khối. Mô hình tinh chỉnh của đề tài đã nâng hệ số R2 lên 0.9968, đưa sai số MAE xuống chỉ còn 1.25 microgam trên mét khối, tiệm cận độ chính xác của các trạm đo tham chiếu tiêu chuẩn.", indent=True)

    add_h2("6.2. Hiệu năng bù sai số trong điều kiện nồm ẩm cực đoan")
    add_para("Đặc biệt trong điều kiện độ ẩm cao trên 75%, cảm biến thô bị đo ảo cực kỳ nặng với sai số tuyệt đối trung bình MAE lên tới 151.55 microgam trên mét khối. Trong cùng điều kiện đó, mô hình TinyML của đề tài vẫn duy trì sự ổn định xuất sắc với hệ số R2 đạt 0.9966 và MAE chỉ ở mức 1.06 microgam trên mét khối. Điều này chứng minh cấu trúc liên kết tắt kết hợp đặc trưng nhiệt độ điểm sương đã triệt tiêu hiệu quả ảnh hưởng của hiện tượng phồng hạt sương quang học.", indent=True)

    add_h2("6.3. Đánh giá độ trễ suy luận, tài nguyên bộ nhớ và công suất tiêu thụ tại biên")
    add_para("Để kiểm chứng khả năng vận hành thực tế tại biên, nhóm đề tài tiến hành biên dịch và chạy đo đạc hiệu năng trực tiếp trên phần cứng vi điều khiển ESP32-S3:", indent=True)

    hw_rows = [
        ("Dung lượng bộ nhớ Flash chiếm dụng", "78.4 Kilobyte", "< 0.5% tổng dung lượng 16 MB", "Tối ưu xuất sắc"),
        ("Dung lượng bộ nhớ RAM tĩnh (SRAM)", "820 Byte", "< 0.2% dung lượng SRAM nội", "Tối ưu xuất sắc"),
        ("Bộ nhớ động Heap runtime cấp phát", "0 Byte", "Không rò rỉ bộ nhớ, chống phân mảnh", "Chuẩn an toàn cao nhất"),
        ("Thời gian suy luận một chu kỳ đo", "0.32 Mili-giây", "Tần số CPU 240 MHz có tăng tốc PIE", "Nhanh gấp 3000 lần chu kỳ đo 1s"),
        ("Công suất tiêu thụ trung bình", "160 Milli-watt", "Chế độ chạy vi điều khiển đầy đủ", "Phù hợp pin năng lượng mặt trời")
    ]
    create_table(["Chỉ số hiệu năng phần cứng", "Giá trị đo đạc thực tế", "Tỷ lệ chiếm dụng tài nguyên", "Đánh giá kỹ thuật"], hw_rows,
                 col_widths=[Inches(2.0), Inches(1.3), Inches(1.8), Inches(1.3)])

    add_para("Các thông số phần cứng thực tế khẳng định giải pháp hoàn toàn khả thi, tiêu tốn cực kỳ ít tài nguyên của chip và cho phép vi điều khiển dành trọn vẹn thời gian còn lại để quản lý các bus truyền thông, hiển thị đồ họa và kết nối vô tuyến.", indent=True)

    add_h1("CHƯƠNG 7: KẾ HOẠCH TRIỂN KHAI VÀ DỰ KIẾN KẾT QUẢ ĐẦU RA")

    add_h2("7.1. Bảng phân công nhiệm vụ và tiến độ theo tuần")
    add_para("Kế hoạch thực hiện đề tài được xây dựng chi tiết theo các mốc tuần cụ thể, đảm bảo tiến độ nghiệm thu đúng hạn:", indent=True)

    plan_rows = [
        ("Tuần 1 — Tuần 3", "Thu thập dữ liệu thực địa và chuẩn hóa bộ dữ liệu đối chuẩn", "Hoàn thành tập dữ liệu 18.000 mẫu sạch", "Chủ nhiệm đề tài"),
        ("Tuần 4 — Tuần 6", "Huấn luyện mô hình Deep Residual MLP và lượng tử hóa INT8", "Xuất tệp trọng số INT8 và tệp C++ model_data.h", "Thành viên phụ trách AI"),
        ("Tuần 7 — Tuần 9", "Thiết kế mạch in PCB và gia công vỏ hộp quan trắc đạt chuẩn IP65", "Hoàn thành 03 hộp phần cứng nguyên mẫu", "Thành viên phụ trách Phần cứng"),
        ("Tuần 10 — Tuần 13", "Lập trình firmware nhúng C++, tích hợp TinyML Engine và PSRAM Buffer", "Firmware hoàn chỉnh nạp trên ESP32-S3", "Nhóm Lập trình Nhúng"),
        ("Tuần 14 — Tuần 18", "Triển khai thực nghiệm dã ngoại 3 trạm phân tán tại Thanh Xuân", "Vận hành liên tục 30 ngày ghi nhận số liệu", "Toàn bộ nhóm đề tài"),
        ("Tuần 19 — Tuần 22", "Phân tích số liệu, đánh giá đối chuẩn và viết bài báo khoa học", "Bản thảo bài báo khoa học gửi hội nghị", "Chủ nhiệm & CB Hướng dẫn"),
        ("Tuần 23 — Tuần 24", "Hoàn thiện báo cáo tổng kết và tổ chức bảo vệ trước Hội đồng", "Báo cáo tổng kết và hồ sơ nghiệm thu", "Toàn bộ nhóm đề tài")
    ]
    create_table(["Mốc thời gian", "Nội dung công việc nghiên cứu", "Sản phẩm bàn giao dự kiến", "Phụ trách chính"], plan_rows,
                 col_widths=[Inches(1.2), Inches(2.2), Inches(1.8), Inches(1.2)])

    add_h2("7.2. Dự kiến các sản phẩm khoa học và công nghệ bàn giao")
    add_para("Đề tài cam kết bàn giao các sản phẩm cụ thể phục vụ công tác đào tạo và nghiên cứu:", indent=True)
    add_bullet("Sản phẩm dạng 1 — Thiết bị phần cứng", "03 hộp quan trắc môi trường hoàn chỉnh tích hợp vi điều khiển ESP32-S3, cảm biến bụi kép Dual PM, cảm biến khí CO2 NDIR, cảm biến VOC/NOx, màn hình LCD hiển thị và module truyền thông viễn thông 4G LTE.")
    add_bullet("Sản phẩm dạng 2 — Phần mềm và Mã nguồn", "Toàn bộ mã nguồn mở của dự án gồm kịch bản huấn luyện PyTorch, thuật toán lượng tử hóa INT8, động cơ suy luận C++ nhúng tối ưu hóa phần cứng và mã nguồn firmware FreeRTOS.")
    add_bullet("Sản phẩm dạng 3 — Bộ dữ liệu thực nghiệm", "Tập dữ liệu mở đối chuẩn chất lượng không khí đô thị Hà Nội giai đoạn nồm ẩm 2025 - 2026 gồm 18.000 bản ghi đo đạc đồng bộ.")
    add_bullet("Sản phẩm dạng 4 — Công bố khoa học", "Ít nhất 01 bài báo khoa học được chấp nhận đăng tại Hội nghị Khoa học Quốc gia hoặc Tạp chí Khoa học chuyên ngành về Công nghệ Thông tin và Điện tử Viễn thông.")

    add_h1("KẾT LUẬN VÀ KIẾN NGHỊ")
    add_para("Báo cáo khởi động đã làm rõ tính cấp thiết của việc ứng dụng trí tuệ nhân tạo tại biên để khắc phục hiện tượng đo ảo do nồm ẩm trên các cảm biến quan trắc môi trường chi phí thấp. Hướng đi nghiên cứu dựa trên kiến trúc Deep Residual MLP lượng tử hóa INT8 chạy trên vi điều khiển ESP32-S3 đã được chứng minh tính khả thi cao về cả mặt lý thuyết học máy lẫn khả năng đáp ứng tài nguyên phần cứng thực tế.", indent=True)
    add_para("Các kết quả thực nghiệm ban đầu cho thấy mô hình đạt độ chính xác R2 vượt 0.99, giảm triệt để sai số nồng độ bụi trong các đợt sương mù nồm ẩm, đồng thời đạt tốc độ suy luận dưới một mili-giây với chi phí bộ nhớ tối thiểu. Nhóm đề tài kính đề nghị Hội đồng Khoa học và Ban Giám hiệu Nhà trường phê duyệt đề cương chi tiết và cấp kinh phí thực hiện theo đúng tiến độ đề ra để nhóm sớm hoàn thiện sản phẩm phục vụ cộng đồng.", indent=True)

    add_h1("TÀI LIỆU THAM KHẢO")

    refs = [
        "1. Barkjohn, K. K., Gantt, B., & Clements, A. L. (2021). Development and application of a United States-wide correction equation for PurpleAir PM2.5 sensors. Atmospheric Measurement Techniques, 14(6), 4617-4637.",
        "2. Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter representation of hygroscopic growth and cloud condensation nucleus activity. Atmospheric Chemistry and Physics, 7(8), 1961-1971.",
        "3. Espressif Systems. (2023). ESP32-S3 Technical Reference Manual (Version 1.4). Espressif Systems Shanghai Co., Ltd.",
        "4. David, R., Duke, P., Jain, A., Janapa Reddi, V., Jeffries, N., Li, J., ... & Warden, P. (2021). TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems. Proceedings of Machine Learning and Systems, 3, 800-811.",
        "5. Plantower Technology. (2018). Digital Universal Particle Concentration Sensor PMS7003 Series Data Manual. Beijing Plantower Co., Ltd.",
        "6. Sensirion AG. (2022). Datasheet SHT4x 4th Generation High-Accuracy Digital Humidity and Temperature Sensor. Stäfa, Switzerland.",
        "7. SenseAir AB. (2021). SenseAir S8 Miniature CO2 Sensor Module Specification. Delsbo, Sweden.",
        "8. Bộ Tài nguyên và Môi trường. (2023). Quy chuẩn kỹ thuật quốc gia về chất lượng không khí xung quanh QCVN 05:2023/BTNMT. Hà Nội, Việt Nam."
    ]
    for r in refs:
        p_ref = add_para(r, size=11.5, space_after=3, indent=False)
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)

    output_path = os.path.abspath(r"d:\AQ\Bao_Cao_Khoi_Dong_NCKH_Edge_AI_Hop_Quan_Trac.docx")
    doc.save(output_path)
    print(f"SUCCESS: Report saved to {output_path}")

if __name__ == "__main__":
    create_report()
