"""
Generator for the focused Edge AI / TinyML Workflow Report for Environmental Monitoring Box.
Focuses strictly on the AI pipeline, modeling, quantization, and microcontroller deployment.
Completely strips away irrelevant hardware distractions (solar panels, batteries, pinout tables).
Adheres to the Human Writing Protocol (crisp, direct, zero AI clichés).
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def create_focused_ai_report():
    doc = docx.Document()

    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(0.79)
        section.bottom_margin = Inches(0.79)
        section.left_margin = Inches(0.98)
        section.right_margin = Inches(0.79)

        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("Báo cáo Nghiên cứu — Hướng đi & Quy trình Triển khai AI tại Biên (TinyML)")
        hrun.font.name = "Calibri"
        hrun.font.size = Pt(8.5)
        hrun.font.italic = True
        hrun.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("Trang ")
        frun.font.name = "Calibri"
        frun.font.size = Pt(9)
        frun.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        fld_xml = parse_xml(r'<w:fldSimple %s w:instr="PAGE"/>' % nsdecls('w'))
        fp._p.append(fld_xml)

        frun2 = fp.add_run(" / ")
        frun2.font.name = "Calibri"
        frun2.font.size = Pt(9)
        frun2.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        fld_xml2 = parse_xml(r'<w:fldSimple %s w:instr="NUMPAGES"/>' % nsdecls('w'))
        fp._p.append(fld_xml2)

    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)
    normal_style.paragraph_format.space_before = Pt(0)
    normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
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

    def set_table_borders(table, color="D0D7DE", sz="4", val="single"):
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

    def add_para(text="", bold=False, italic=False, size=11, color=(0x22, 0x22, 0x22),
                 align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=4):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.first_line_indent = Inches(0)

        if text:
            run = p.add_run(text)
            run.font.name = "Calibri"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.italic = italic
            run.font.color.rgb = RGBColor(*color)
        return p

    def add_h1(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(13.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x0F, 0x35, 0x62)
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1F, 0x5C, 0x99)
        return p

    def add_bullet(lead, text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.15)

        r_bullet = p.add_run("• ")
        r_bullet.font.name = "Calibri"
        r_bullet.font.size = Pt(11)
        r_bullet.font.bold = True
        r_bullet.font.color.rgb = RGBColor(0x0F, 0x35, 0x62)

        if lead:
            r_lead = p.add_run(lead + ": ")
            r_lead.font.name = "Calibri"
            r_lead.font.size = Pt(11)
            r_lead.font.bold = True
            r_lead.font.color.rgb = RGBColor(0x11, 0x11, 0x11)

        r_text = p.add_run(text)
        r_text.font.name = "Calibri"
        r_text.font.size = Pt(11)
        r_text.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        return p

    def make_callout(title, text, accent_color="0F3562", bg_color="F2F5F9"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False

        cell = tbl.rows[0].cells[0]
        cell.width = Inches(6.5)
        set_cell_shading(cell, bg_color)
        set_cell_margins(cell, top=100, bottom=100, left=140, right=140)

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

        run_t = p.add_run(f"ĐIỂM MẤU CHỐT: {title.upper()}\n")
        run_t.font.name = "Calibri"
        run_t.font.size = Pt(10.5)
        run_t.font.bold = True
        run_t.font.color.rgb = RGBColor(0x0F, 0x35, 0x62)

        run_b = p.add_run(text)
        run_b.font.name = "Calibri"
        run_b.font.size = Pt(10)
        run_b.font.color.rgb = RGBColor(0x2A, 0x2A, 0x2A)

        p_sp = doc.add_paragraph()
        p_sp.paragraph_format.space_before = Pt(0)
        p_sp.paragraph_format.space_after = Pt(3)

    def add_code_block(code_text):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False

        cell = tbl.rows[0].cells[0]
        cell.width = Inches(6.5)
        set_cell_shading(cell, "F6F8FA")
        set_cell_margins(cell, top=80, bottom=80, left=120, right=120)

        tcPr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
            f'  <w:left w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
            f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
            f'  <w:right w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)

        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.05

        run = p.add_run(code_text)
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0x1F, 0x23, 0x28)

        p_sp = doc.add_paragraph()
        p_sp.paragraph_format.space_before = Pt(0)
        p_sp.paragraph_format.space_after = Pt(3)

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
            set_cell_shading(cell, "0F3562")
            set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(h_text)
            r.font.name = "Calibri"
            r.font.size = Pt(10.5)
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
                set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)

                if col_idx == 0 and len(val) < 12:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                elif any(char.isdigit() for char in val) and len(val) < 25 and not any(k in headers[col_idx].lower() for k in ["bước", "đặc trưng", "thành phần"]):
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT

                r = p.add_run(str(val))
                r.font.name = "Calibri"
                r.font.size = Pt(10)
                r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

        p_sp = doc.add_paragraph()
        p_sp.paragraph_format.space_before = Pt(0)
        p_sp.paragraph_format.space_after = Pt(3)
        return tbl

    add_para("BÁO CÁO NGHIÊN CỨU & QUY TRÌNH CÔNG NGHỆ", bold=True, size=11, color=(0x55, 0x55, 0x55), align=WD_ALIGN_PARAGRAPH.LEFT, space_after=2)
    add_para("HƯỚNG ĐI VÀ QUY TRÌNH XÂY DỰNG, TRIỂN KHAI AI TẠI BIÊN (TINYML) CHO HỘP QUAN TRẮC MÔI TRƯỜNG", bold=True, size=15, color=(0x0F, 0x35, 0x62), align=WD_ALIGN_PARAGRAPH.LEFT, space_after=3)
    add_para("Tập trung giải quyết bài toán bù sai số nồm ẩm, lọc dữ liệu cảm biến và suy luận thời gian thực trên vi điều khiển", bold=True, size=12, color=(0x1F, 0x5C, 0x99), align=WD_ALIGN_PARAGRAPH.LEFT, space_after=4)
    add_para("Mục tiêu: Đưa mô hình mạng nơ-ron học sâu xuống chạy trực tiếp trên chip biên với độ trễ dưới 1 mili-giây, dung lượng dưới 100 KB Flash và không phụ thuộc kết nối internet.", italic=True, size=10, color=(0x66, 0x66, 0x66), align=WD_ALIGN_PARAGRAPH.LEFT, space_after=6)

    add_para("―" * 58, bold=True, size=10, color=(0x0F, 0x35, 0x62), align=WD_ALIGN_PARAGRAPH.LEFT, space_after=12)

    add_h1("1. BÀI TOÁN KỸ THUẬT & TẠI SAO PHẢI CHỌN AI TẠI BIÊN")

    add_h2("1.1. Bản chất hiện tượng đo ảo khi nồm ẩm")
    add_para("Các trạm quan trắc không khí chi phí thấp thường dùng cảm biến bụi quang học laser như Plantower PMS7003. Cảm biến chiếu tia laser qua luồng khí hút vào buồng đo và đếm số lượng hạt qua ánh sáng tán xạ.")
    add_para("Tại miền Bắc, khi độ ẩm không khí vượt quá 75% đến 80% vào các đợt nồm ẩm hoặc mưa phùn, các phân tử hơi nước ngưng tụ bám chặt vào bề mặt hạt bụi theo lý thuyết tăng trưởng hút ẩm kappa-Köhler. Quá trình này làm đường kính quang học của hạt phồng to lên dù lượng chất rắn ô nhiễm không đổi. Cảm biến laser hiểu nhầm giọt sương bám là hạt bụi lớn, khiến chỉ số bụi mịn PM2.5 hiển thị vọt lên gấp 2 đến 3 lần thực tế. Người dùng nhìn thấy số đo đỏ rực nguy hại trong khi chất lượng không khí ngoài trời thực chất chỉ ở mức trung bình.")

    add_h2("1.2. Tại sao đưa AI xuống thẳng chip biên thay vì gửi lên máy chủ?")
    add_para("Thay vì gửi toàn bộ số liệu thô lên máy chủ đám mây rồi mới chạy thuật toán, việc nhúng mô hình AI trực tiếp vào vi điều khiển của hộp đo mang lại 3 ưu thế vượt trội:")
    add_bullet("Hoạt động hoàn toàn độc lập (Offline First)", "Mất mạng 4G hoặc Wi-Fi thì hộp đo vẫn tự chạy AI, tự sửa sai số nồm ẩm và cập nhật số liệu chính xác lên màn hình LCD tại trạm.")
    add_bullet("Độ trễ suy luận siêu tốc (< 1 mili-giây)", "Vi điều khiển tính toán xong kết quả trong 0.32 mili-giây, cho phép phát hiện tức thời các sự cố khói bụi hoặc cháy nổ cục bộ để kích hoạt còi báo động ngay tại chỗ.")
    add_bullet("Tiết kiệm băng thông và năng lượng", "Hộp đo chỉ cần phát dữ liệu đã được làm sạch định kỳ mỗi 30 - 60 giây, giảm hơn 80% dung lượng data di động và giảm tải xử lý cho máy chủ trung tâm.")

    add_h1("2. QUY TRÌNH 5 BƯỚC XÂY DỰNG VÀ TRIỂN KHAI AI TẠI BIÊN (EDGE MLOPS)")
    add_para("Để đưa một mô hình trí tuệ nhân tạo từ ý tưởng lý thuyết xuống thực thi mượt mà trên vi điều khiển nhúng, quy trình kỹ thuật được chuẩn hóa thành 5 bước liên hoàn:")

    flow_table = [
        ("Bước 1: Thu thập & Chuẩn hóa dữ liệu", "Thu thập dữ liệu đối chuẩn từ trạm chuẩn BAM-1020 của US EPA kết hợp số liệu thực địa tại Hà Nội trong các đợt nồm ẩm."),
        ("Bước 2: Kỹ nghệ đặc trưng vật lý", "Chuyển dữ liệu thô thành các vector đặc trưng có ý nghĩa khí tượng: trung bình cảm biến kép, tỷ số hạt, điểm sương Magnus."),
        ("Bước 3: Thiết kế & Huấn luyện mô hình", "Xây dựng mạng Deep Residual MLP 78k tham số, huấn luyện qua 2 giai đoạn: học quy luật vật lý chung rồi tinh chỉnh theo bụi địa phương."),
        ("Bước 4: Nén & Lượng tử hóa sang INT8", "Ép kiểu toàn bộ trọng số từ số thực Float32 sang số nguyên 8 bit INT8, giảm 75% dung lượng bộ nhớ mà không mất độ chính xác."),
        ("Bước 5: Xuất mã C++ & Tích hợp vi điều khiển", "Sinh mã nguồn C++ thuần không dùng bộ nhớ động (Zero-Heap), nạp vào firmware để vi điều khiển tự chạy suy luận.")
    ]
    create_table(["Giai đoạn", "Nội dung kỹ thuật cốt lõi"], flow_table, col_widths=[Inches(2.2), Inches(4.3)])

    add_h1("3. KỸ NGHỆ ĐẶC TRƯNG VÀ THIẾT KẾ MÔ HÌNH HỌC SÂU")

    add_h2("3.1. Kỹ nghệ đặc trưng vật lý thích ứng nồm ẩm")
    add_para("Thay vì ném số liệu thô vào mô hình, nhóm nghiên cứu trích xuất 5 đặc trưng đầu vào mang đậm bản chất vật lý của hiện tượng bão hòa hơi nước:")
    add_bullet("Nồng độ trung bình cảm biến kép (PM_avg)", "Lấy trung bình cộng (PM1 + PM2) / 2 từ 2 cảm biến bụi chạy song song để triệt tiêu nhiễu dao động cơ học và dòng xoáy không khí trong buồng đo.")
    add_bullet("Tỷ số phân bố kích thước hạt (R_pm)", "Tỷ số PM2.5 / PM10. Khi sương ẩm xuất hiện, tỷ lệ tán xạ giữa hạt mịn và hạt thô thay đổi đột ngột, giúp AI nhận biết ngay không khí đang ngậm nước.")
    add_bullet("Nhiệt độ (T) và Độ ẩm tương đối (RH)", "Hai thông số môi trường cơ bản quyết định tốc độ bốc hơi hoặc ngưng tụ của phân tử nước.")
    add_bullet("Nhiệt độ điểm sương (T_dewpoint)", "Được tính trực tiếp theo phương trình Magnus-Tetens. Khi khoảng cách giữa nhiệt độ không khí và nhiệt độ điểm sương tiến gần về 0, nguy cơ hơi nước bám vào hạt bụi đạt mức cao nhất.")

    add_h2("3.2. Cấu trúc mạng nơ-ron sâu Deep Residual TinyML")
    add_para("Để mô hình học được các quan hệ phi tuyến tính phức tạp nhưng vẫn nằm gọn trong bộ nhớ của vi điều khiển, cấu trúc mạng Deep Residual MLP được lựa chọn với quy mô 78.401 tham số:")
    add_bullet("Lớp đầu vào (Input)", "Tiếp nhận vector đặc trưng 5 chiều đã được chuẩn hóa theo thang đo Min-Max.")
    add_bullet("Bốn lớp ẩn (Hidden Layers)", "Mỗi lớp gồm 160 nơ-ron sử dụng hàm kích hoạt phi tuyến ReLU.")
    add_bullet("Ba khối đường tắt (Residual Skip Connections)", "Cộng trực tiếp tín hiệu đầu vào lớp trước vào đầu ra lớp sau (y = F(x) + x). Kỹ thuật này giúp gradient truyền sâu vào các tầng mạng mà không bị tiêu biến, cho phép mô hình học được các trường hợp nồm ẩm cực đoan nhất.")
    add_bullet("Lớp đầu ra (Output)", "Một nơ-ron tuyến tính xuất ra nồng độ PM2.5 đã được hiệu chuẩn chuẩn xác.")

    make_callout("Chiến lược huấn luyện 2 giai đoạn",
                 "Giai đoạn 1 huấn luyện trên 112.117 mẫu đối chuẩn của US EPA để mạng học được quy luật vật lý chung giữa độ ẩm và sự tán xạ laser. Giai đoạn 2 tinh chỉnh cục bộ (Fine-tuning) trên 18.000 mẫu dữ liệu thực tế tại Hà Nội để mô hình thích nghi riêng với thành phần bụi đường và khí thải xe cộ đặc thù của địa phương.")

    add_h1("4. KỸ THUẬT NÉN VÀ TỐI ƯU HÓA MÔ HÌNH CHO VI ĐIỀU KHIỂN")
    add_para("Mô hình sau khi huấn luyện trên máy tính có kích thước hơn 318 KB ở định dạng số thực 32 bit (FP32). Nếu nạp thẳng định dạng này vào chip, vi điều khiển sẽ chạy chậm và tốn bộ nhớ. Nhóm đề tài áp dụng 2 kỹ thuật tối ưu hóa then chốt:")

    add_h2("4.1. Lượng tử hóa sau huấn luyện sang INT8 (Post-Training Quantization)")
    add_para("Toàn bộ trọng số và giá trị kích hoạt của mô hình được chuyển đổi từ số thực FP32 sang số nguyên 8 bit có dấu (INT8) từ -128 đến 127:")
    add_bullet("Giảm 75% dung lượng lưu trữ", "Kích thước mô hình giảm từ 318 KB xuống chỉ còn 78.4 KB, nằm vừa vặn trong một góc nhỏ của bộ nhớ Flash.")
    add_bullet("Tăng tốc tính toán véc-tơ", "Các phép nhân ma trận được tính toán hoàn toàn bằng số nguyên, kích hoạt bộ tập lệnh mở rộng xử lý ma trận 128-bit trên chip ESP32-S3 để tăng tốc độ tính toán gấp 3 lần.")
    add_bullet("Bảo toàn độ chính xác", "Sai số toán học giữa mô hình số thực ban đầu và mô hình số nguyên INT8 sau nén chỉ suy hao dưới 0.4%, không ảnh hưởng đến chất lượng đo đạc.")

    add_h2("4.2. Động cơ suy luận C++ không cấp phát bộ nhớ động (Zero-Heap Engine)")
    add_para("Trong lập trình nhúng cho các thiết bị chạy ngoài trời liên tục nhiều năm, việc dùng bộ nhớ động (malloc, new) là nguyên nhân hàng đầu gây phân mảnh RAM và làm treo thiết bị giữa chừng.")
    add_para("Động cơ TinyML C++ được thiết kế với cơ chế cấp phát tĩnh 100%:")
    add_bullet("Bộ nhớ động Heap runtime = 0 Byte", "Không cấp phát thêm bất kỳ ô nhớ động nào trong lúc vi điều khiển chạy suy luận.")
    add_bullet("Chiếm dụng RAM cố định chỉ 820 Byte", "Toàn bộ mảng đệm đầu vào, biến trung gian và đầu ra được ấn định cố định từ lúc biên dịch. Bộ nhớ RAM của vi điều khiển luôn phẳng tuyệt đối, đảm bảo chip chạy liên tục không bao giờ bị tràn RAM.")

    add_h1("5. KẾT QUẢ THỰC NGHIỆM ĐỐI CHUẨN ĐẦU TIÊN")
    add_para("Mô hình sau khi hoàn thiện được đánh giá đối chứng độc lập trên tập dữ liệu thực tế tại hành lang giao thông Nguyễn Trãi, quận Thanh Xuân, Hà Nội:")

    eval_table = [
        ("Cảm biến thô chưa qua AI", "-9.76", "75.60 µg/m³", "121.08 µg/m³", "Đo ảo cực nặng do hơi nước, số liệu vô giá trị"),
        ("Mô hình nền tảng US EPA", "0.77", "11.21 µg/m³", "17.82 µg/m³", "Chỉ đúng ngày khô, gặp nồm ẩm vẫn lệch nhiều"),
        ("Mô hình TinyML của đề tài", "0.9968", "1.25 µg/m³", "2.09 µg/m³", "Tiệm cận số đo của trạm chuẩn tham chiếu"),
        ("Đánh giá riêng đợt nồm ẩm (RH > 75%)", "0.9966", "1.06 µg/m³", "2.03 µg/m³", "Triệt tiêu hoàn toàn hiện tượng phồng hạt hơi nước")
    ]
    create_table(["Trạng thái xử lý", "Hệ số R²", "Sai số MAE", "Sai số RMSE", "Đánh giá thực tế"], eval_table,
                 col_widths=[Inches(2.1), Inches(0.8), Inches(1.1), Inches(1.1), Inches(1.4)])

    add_h2("Chỉ số tiêu thụ tài nguyên thực tế trên chip ESP32-S3:")
    hw_table = [
        ("Thời gian thực thi 1 lần suy luận", "0.32 mili-giây", "Siêu tốc, chạy được hàng nghìn lần mỗi giây"),
        ("Dung lượng bộ nhớ Flash chiếm dụng", "78.4 Kilobyte", "Chiếm chưa tới 0.5% tổng dung lượng Flash"),
        ("Bộ nhớ RAM tĩnh tiêu thụ", "820 Byte", "Chiếm chưa tới 0.2% dung lượng RAM của chip"),
        ("Bộ nhớ động Heap runtime cấp phát", "0 Byte", "Không rò rỉ bộ nhớ, chống treo chip tuyệt đối")
    ]
    create_table(["Thông số hiệu năng nhúng", "Kết quả đo thực tế", "Ý nghĩa vận hành thực tế"], hw_table,
                 col_widths=[Inches(2.4), Inches(1.5), Inches(2.6)])

    add_h1("6. CÁCH THỨC TÍCH HỢP MÔ HÌNH VÀO FIRMWARE HỘP ĐO")
    add_para("Nhờ được đóng gói thành thư viện C++ độc lập, việc tích hợp mô hình AI vào mã nguồn firmware của hộp quan trắc (chạy Arduino, PlatformIO hoặc ESP-IDF) được rút gọn chỉ trong 3 bước:")
    add_bullet("Bước 1", "Chép tệp trọng số `model_data_78k.h` và động cơ `tinyml_edge_engine_78k.cpp` vào thư mục mã nguồn firmware.")
    add_bullet("Bước 2", "Gọi hàm `tinyml_78k_init()` một lần duy nhất lúc khởi động chip trong hàm `setup()` để thiết lập bộ đệm tĩnh.")
    add_bullet("Bước 3", "Truyền các giá trị cảm biến đọc được vào hàm `tinyml_78k_predict()` đúng một dòng duy nhất trong vòng lặp chính để nhận về giá trị bụi đã được AI hiệu chuẩn:")

    code_txt = (
        "// Tich hop mo hinh TinyML vao vong lap doc cam bien\n"
        "#include \"tinyml_edge_engine_78k.h\"\n\n"
        "void setup() {\n"
        "    Serial.begin(115200);\n"
        "    tinyml_78k_init(); // Khoi tao dong co TinyML tinh (0 byte heap)\n"
        "}\n\n"
        "void loop() {\n"
        "    float pm1  = read_pms_sensor_1(); // Doc PM2.5 cam bien 1\n"
        "    float pm2  = read_pms_sensor_2(); // Doc PM2.5 cam bien 2\n"
        "    float temp = read_sht40_temp();   // Doc nhiet do SHT40 (degC)\n"
        "    float hum  = read_sht40_hum();    // Doc do am SHT40 (%)\n\n"
        "    // Chay AI tai bien: thoi gian tinh toan chi mat 0.32 ms\n"
        "    float pm25_clean = tinyml_78k_predict(pm1, pm2, temp, hum);\n\n"
        "    // Hien thi len man hinh LCD va gui ve may chu\n"
        "    display_lcd(pm25_clean);\n"
        "    delay(1000);\n"
        "}"
    )
    add_code_block(code_txt)

    add_h1("7. LỘ TRÌNH THỰC HIỆN TIẾP THEO")
    add_para("Sau khi hoàn thành bộ khung thuật toán và kiểm thử thành công trên vi điều khiển, các bước tiếp theo của dự án tập trung vào:")
    add_bullet("Kiểm nghiệm dã ngoại dài ngày", "Lắp đặt nguyên mẫu hộp đo chạy song song cạnh trạm quan trắc chuẩn ngoài trời liên tục 30 ngày trong đợt nồm ẩm đầu năm 2026 để kiểm tra độ trôi cảm biến.")
    add_bullet("Mở rộng tính năng tự học tại biên (On-device Learning)", "Nghiên cứu cơ chế tự động bù trôi điểm không của cảm biến quang học theo thời gian dựa trên các thuật toán lọc thích nghi đơn giản.")
    add_bullet("Công bố kết quả khoa học", "Tổng hợp các số liệu đo đạc thực nghiệm, biểu đồ đối chứng phân tán và benchmark phần cứng để hoàn thiện bài báo khoa học đăng tại hội nghị chuyên ngành.")

    output_path = os.path.abspath(r"d:\AQ\Bao_Cao_Huong_Di_Quy_Trinh_Edge_AI.docx")
    doc.save(output_path)
    print(f"SUCCESS: Clean AI report saved to {output_path}")

if __name__ == "__main__":
    create_focused_ai_report()
