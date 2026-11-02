"""
Generator for the practical, highly readable Engineering Architecture & Edge AI Workflow Report.
Matches the exact practical, direct, accessible engineering document style provided by the user.
Adheres strictly to the Human Writing Protocol (zero AI fluff, direct copulas, no decorative emojis).
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def create_edge_ai_report():
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
        hrun = hp.add_run("Tài liệu Kỹ thuật — Quy trình Xây dựng & Triển khai Edge AI trên ESP32-S3")
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

        run_t = p.add_run(f"LƯU Ý KỸ THUẬT: {title.upper()}\n")
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
                elif any(char.isdigit() for char in val) and len(val) < 25 and not any(k in headers[col_idx].lower() for k in ["chân", "tên", "nguyên lý", "thành phần"]):
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

    add_para("TÀI LIỆU KỸ THUẬT & QUY TRÌNH CÔNG NGHỆ", bold=True, size=11, color=(0x55, 0x55, 0x55), align=WD_ALIGN_PARAGRAPH.LEFT, space_after=2)
    add_para("HƯỚNG ĐI VÀ QUY TRÌNH XÂY DỰNG, TRIỂN KHAI AI TẠI BIÊN CHO HỘP QUAN TRẮC MÔI TRƯỜNG NGOÀI TRỜI", bold=True, size=15, color=(0x0F, 0x35, 0x62), align=WD_ALIGN_PARAGRAPH.LEFT, space_after=3)
    add_para("TỰ ĐỘNG BÙ NỒM ẨM, LỌC CẢM BIẾN KÉP VÀ SUY LUẬN THỜI GIAN THỰC TRÊN ESP32-S3", bold=True, size=12, color=(0x1F, 0x5C, 0x99), align=WD_ALIGN_PARAGRAPH.LEFT, space_after=4)
    add_para("Quy chuẩn thiết kế: Hệ thống biên ESP32-S3 (Xtensa Dual-core LX7 + Vector PIE) kết hợp Máy chủ AI Raspberry Pi 5 (8GB RAM)\nPhân tích & Tối ưu hóa mô hình Deep Residual TinyML 78k tham số lượng tử hóa INT8", italic=True, size=10, color=(0x66, 0x66, 0x66), align=WD_ALIGN_PARAGRAPH.LEFT, space_after=6)

    p_line = add_para("―" * 58, bold=True, size=10, color=(0x0F, 0x35, 0x62), align=WD_ALIGN_PARAGRAPH.LEFT, space_after=12)

    add_h1("1. TỔNG QUAN HƯỚNG ĐI & BÀI TOÁN KỸ THUẬT CỐT LÕI")
    add_para("Dự án xây dựng hộp quan trắc chất lượng không khí ngoài trời tự động, hoạt động bền bỉ trong điều kiện khí hậu nhiệt đới gió mùa tại miền Bắc Việt Nam. Trọng tâm của hệ thống là ứng dụng trí tuệ nhân tạo tại biên (Edge AI / TinyML) chạy trực tiếp trên vi điều khiển ESP32-S3 để xử lý dữ liệu cảm biến ngay tại chỗ trước khi truyền về máy chủ trung tâm Raspberry Pi 5.")

    add_h2("Tại sao phải đưa AI xuống biên vi điều khiển thay vì đẩy hết lên Cloud?")
    add_para("Hầu hết các hệ thống IoT truyền thống đẩy toàn bộ dữ liệu thô từ cảm biến lên đám mây rồi mới chạy thuật toán hiệu chuẩn. Cách làm này gặp 4 trở ngại lớn khi triển khai thực tế ngoài trời:")
    add_bullet("Hoạt động ngoại tuyến 100% (Zero Cloud Dependency)", "Khi trạm đo đặt ở các vị trí sóng 4G chập chờn hoặc mất mạng kéo dài, vi điều khiển vẫn tự chạy mô hình AI, tự bù trừ độ ẩm và hiển thị chỉ số chính xác tại chỗ lên màn hình LCD cho người quan sát.")
    add_bullet("Phản hồi thời gian thực siêu tốc (< 1 mili-giây)", "Thời gian suy luận của mạng nơ-ron trên chip ESP32-S3 chỉ mất 0.32 mili-giây. Thiết bị phát hiện ngay tức khắc các đợt phát thải khói độc hoặc nồng độ bụi tăng đột ngột để kích hoạt còi cảnh báo tại chỗ mà không cần chờ độ trễ mạng internet.")
    add_bullet("Tiết kiệm pin mặt trời và dung lượng 4G", "Vi điều khiển chỉ cần gửi dữ liệu sạch đã qua xử lý định kỳ 30 - 60 giây một lần thay vì phải liên tục truyền stream dữ liệu thô tần suất cao, giúp tiết kiệm hơn 80% dung lượng data di động và giảm tiêu hao năng lượng pin LiFePO4.")
    add_bullet("Khả năng mở rộng không giới hạn", "Mỗi trạm đo tự gánh toàn bộ tải tính toán AI của chính mình. Khi mở rộng mạng lưới từ vài trạm lên hàng trăm trạm, máy chủ trung tâm Raspberry Pi 5 không bị nghẽn CPU và chỉ cần tập trung vào lưu trữ, phân tích không gian thời gian.")

    add_h2("Bốn mục tiêu kỹ thuật cốt lõi tại trạm biên:")
    add_bullet("Khắc phục hiện tượng nồm ẩm miền Bắc", "Cảm biến bụi quang học laser khi gặp độ ẩm tương đối RH > 75% sẽ bị hơi nước bám vào hạt bụi theo lý thuyết tăng trưởng hút ẩm kappa-Köhler, khiến giá trị PM2.5 đo ảo vọt lên gấp 2 đến 3 lần. AI tại biên sử dụng phương trình điểm sương Magnus-Tetens và mạng nơ-ron sâu để đưa giá trị đo về sát với trạm chuẩn tham chiếu.")
    add_bullet("Lọc nhiễu và phát hiện dị thường bằng cảm biến kép (Dual PM)", "Sử dụng song song 2 cảm biến bụi Plantower PMS7003 để đối chiếu chéo liên tục. Nếu độ lệch giữa 2 cảm biến vượt ngưỡng cho phép, hệ thống lập tức phát hiện buồng đo bị đọng sương, bám mạng nhện hoặc kẹt quạt hút gió.")
    add_bullet("Tối ưu hóa phần cứng bằng tập lệnh Vector PIE", "Mô hình mạng nơ-ron được lượng tử hóa hoàn toàn sang số nguyên 8 bit (INT8), tận dụng phần cứng mở rộng véc-tơ 128-bit trên nhân Xtensa LX7 của ESP32-S3 để tăng tốc tính toán ma trận mà không tiêu tốn bộ nhớ.")
    add_bullet("Lưu đệm chống mất dữ liệu ngoại tuyến (Zero-Data-Loss)", "Tận dụng 8 megabyte bộ nhớ ngoài PSRAM để tạo bộ đệm vòng tròn (Circular Ring Buffer), lưu trữ liên tục hơn 45 ngày khi mất mạng 4G và tự động đồng bộ dồn về máy chủ khi có mạng trở lại.")

    add_h1("2. KIẾN TRÚC PHẦN CỨNG HỘP BIÊN ESP32-S3")
    add_para("Trạm đo được thiết kế tích hợp trên một bo mạch chủ tối ưu, bố trí gọn gàng trong hộp bảo vệ in 3D chống mưa nắng ngoài trời.")

    add_h2("2.1. Vi điều khiển trung tâm ESP32-S3 N16R8")
    add_para("ESP32-S3 là lựa chọn hàng đầu cho các ứng dụng TinyML hiện nay nhờ cấu hình phần cứng chuyên biệt:")
    add_bullet("Vi xử lý", "Dual-core Xtensa 32-bit LX7 hoạt động ở tần số xung nhịp 240 MHz.")
    add_bullet("Tập lệnh véc-tơ (PIE)", "Tập lệnh mở rộng xử lý ma trận 128-bit chuyên dụng cho phép tính nhân tích lũy MAC số nguyên INT8/INT16, giúp tăng tốc độ suy luận mạng nơ-ron gấp 3 lần so với CPU thông thường.")
    add_bullet("Bộ nhớ lưu trữ", "16 megabyte Flash (chứa firmware và trọng số mô hình AI) và 8 megabyte Octal PSRAM tốc độ cao (chứa bộ đệm dữ liệu ngoại tuyến).")
    add_bullet("Phân chia tải 2 nhân (FreeRTOS Dual-Core)", "Nhân Core 0 chuyên xử lý giao tiếp cảm biến, đọc tín hiệu GPS, quản lý mạng 4G MQTT. Nhân Core 1 dành trọn vẹn 100% năng lực tính toán để chạy động cơ TinyML và cập nhật màn hình LCD.")

    make_callout("Cảnh báo phần cứng sống còn (Hardware Traps)",
                 "Module ESP32-S3 N16R8 sử dụng chuẩn giao tiếp Octal PSRAM tốc độ cao. Toàn bộ các chân từ GPIO33 đến GPIO37 đã bị nhà sản xuất chiếm dụng ngầm bên trong module để nối với chip nhớ RAM ngoài. TUYỆT ĐỐI KHÔNG NỐI BẤT KỲ CẢM BIẾN NÀO VÀO CỤM GPIO33 - GPIO37 để tránh xung đột bus nhớ gây treo chip liên tục.")

    add_h2("2.2. Danh mục cảm biến quan trắc tiêu chuẩn")
    sensor_rows = [
        ("02 x Plantower PMS7003", "Cấu hình cảm biến kép (Dual PM), đo tán xạ laser liên tục nồng độ PM1.0, PM2.5, PM10. Chạy đối chứng chéo khử nhiễu ngẫu nhiên."),
        ("01 x SenseAir S8 (LP/0053)", "Cảm biến CO2 chuẩn quang phổ hồng ngoại không phân tán (NDIR) của Thụy Điển, dải đo 400 - 2000 ppm, độ bền 15 năm, không bị trôi điểm không."),
        ("01 x Sensirion SGP41", "Cảm biến đa kênh kim loại oxit (MOX) đo khí độc hữu cơ bay hơi (TVOC) và khí thải giao thông (NOx), tích hợp thuật toán Gas Index xuất điểm 1 - 500."),
        ("01 x Sensirion SHT40", "Cảm biến nhiệt độ và độ ẩm kỹ thuật số độ chính xác cao (±0.2°C, ±1.8% RH), cung cấp dữ liệu nền tảng để AI tính nhiệt độ điểm sương và bù ẩm."),
        ("Module 4G SIMCOM A7677S", "Module truyền thông di động LTE Cat-1 kèm định vị vệ tinh GNSS, lấy toạ độ vị trí trạm đo và đồng bộ giờ chuẩn thế giới qua sóng vệ tinh."),
        ("Màn hình màu LCD 2.8 inch SPI", "Hiển thị tại chỗ các chỉ số nồng độ bụi thực tế, nhiệt ẩm vi khí hậu, đồ thị diễn biến và màu cảnh báo chất lượng không khí AQI.")
    ]
    create_table(["Cảm biến / Thiết bị", "Thông số kỹ thuật & Vai trò trong hộp quan trắc"], sensor_rows, col_widths=[Inches(2.2), Inches(4.3)])

    add_h2("2.3. Bảng phân bổ chân kết nối (Pinout Mapping)")
    pin_rows = [
        ("Cảm biến Bụi chính: PMS7003 #1", "GPIO17 (RX1), GPIO18 (TX1)", "Hardware UART1 (9600 bps)"),
        ("Cảm biến Bụi phụ: PMS7003 #2", "GPIO15 (RX2), GPIO16 (TX2)", "Hardware UART2 (9600 bps)"),
        ("Cảm biến CO2: SenseAir S8", "GPIO8 (RX), GPIO9 (TX)", "UART Modbus (9600 bps)"),
        ("Khí độc & Nhiệt ẩm: SGP41 + SHT40", "GPIO4 (SDA), GPIO5 (SCL)", "I2C Bus 0 (400 kHz)"),
        ("Màn hình hiển thị LCD 2.8 inch", "GPIO11(MOSI), 12(SCK), 10(CS), 13(DC), 14(RST)", "SPI Bus 2 (40 MHz)"),
        ("Module 4G LTE + GPS: A7677S", "GPIO19 (RX), GPIO20 (TX)", "UART AT Command"),
        ("LED RGB chỉ thị chất lượng AQI", "GPIO48 (Onboard WS2812B)", "One-Wire Protocol")
    ]
    create_table(["Cụm chức năng", "Chân GPIO ấn định", "Giao thức kết nối"], pin_rows, col_widths=[Inches(2.5), Inches(2.2), Inches(1.8)])

    add_h1("3. SƠ ĐỒ KHỐI TỔNG THỂ HỆ THỐNG (SYSTEM BLOCK DIAGRAM)")
    add_para("Sơ đồ dưới đây mô tả toàn bộ kiến trúc kết nối giữa Hộp đo ngoại vi tại biên, Hạ tầng mạng di động 4G, Máy chủ trung tâm Raspberry Pi 5 và Các giao diện phía người dùng cuối:")

    img_path = r"d:\AQ\word\media\image1.png"
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(4)
        p_img.paragraph_format.space_after = Pt(2)
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=Inches(6.3))

        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(0)
        p_cap.paragraph_format.space_after = Pt(8)
        run_cap = p_cap.add_run("Hình 1: Sơ đồ khối tổng thể hệ thống IoT quan trắc môi trường và máy chủ AI Raspberry Pi 5")
        run_cap.font.name = "Calibri"
        run_cap.font.size = Pt(9.5)
        run_cap.font.italic = True
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    add_h2("Quy trình 4 tầng luân chuyển dữ liệu:")
    add_bullet("Tầng 1 — Thu thập & Lọc chéo tại biên (ESP32-S3 Core 0)", "Định kỳ mỗi giây, Core 0 đọc dữ liệu cảm biến kép PM1 và PM2, tính độ lệch Delta PM để loại trừ nhiễu tức thời và đọc các chỉ số nhiệt ẩm, CO2, VOC.")
    add_bullet("Tầng 2 — Suy luận TinyML bù ẩm thời gian thực (ESP32-S3 Core 1)", "Đưa vector đặc trưng vào mạng Deep Residual TinyML INT8 để tính ra nồng độ bụi chuẩn xác, cập nhật ngay lên màn hình LCD tại trạm và ghi vào bộ đệm PSRAM.")
    add_bullet("Tầng 3 — Đóng gói & Truyền dẫn 4G LTE", "Mỗi 30 - 60 giây, gói tin dữ liệu chuẩn hóa được chuyển đổi thành chuỗi JSON và gửi qua module 4G theo giao thức MQTT bảo mật SSL/TLS về đường hầm Cloudflare Tunnel.")
    add_bullet("Tầng 4 — Lưu trữ & Phân tích chuyên sâu trên Raspberry Pi 5", "Máy chủ Raspberry Pi 5 tiếp nhận gói tin qua Mosquitto MQTT Broker, lưu vào cơ sở dữ liệu InfluxDB, đồng thời chạy các mô hình AI dự báo xu hướng ô nhiễm 24 giờ tới và cung cấp API cho Web Dashboard, Mobile App.")

    add_h1("4. QUY TRÌNH 6 BƯỚC XÂY DỰNG & TRIỂN KHAI AI TẠI BIÊN (MLOPS WORKFLOW)")
    add_para("Quy trình xây dựng và đưa mô hình học máy xuống vi điều khiển ESP32-S3 được thực hiện tuần tự qua 6 bước khép kín:")

    add_h2("Bước 1: Chuẩn bị dữ liệu đối chứng Ground Truth (Chiến lược 0 đồng)")
    add_para("Để mô hình AI học được quy luật bù sai số nồm ẩm mà không tốn hàng trăm triệu đồng mua máy đo chuẩn, dự án kết hợp 2 nguồn dữ liệu đối chứng miễn phí:")
    add_bullet("Nguồn 1 — Siêu dữ liệu đối chuẩn quốc tế US EPA AQS (112.117 mẫu)", "Khai thác dữ liệu thực nghiệm nhiều năm của Cơ quan Bảo vệ Môi trường Mỹ, nơi cảm biến dòng Plantower được đặt cạnh trạm đo chuẩn liên bang Met One BAM-1020 ở dải độ ẩm từ 20% đến trên 95% RH. Dữ liệu này giúp mô hình học được bản chất vật lý của sự phồng hạt bụi do hơi nước.")
    add_bullet("Nguồn 2 — Dữ liệu thực địa 3 trạm tại quận Thanh Xuân, Hà Nội (18.000 mẫu)", "Ghi nhận dữ liệu thực tế tại hành lang giao thông Nguyễn Trãi, nút giao Khuất Duy Tiến và Khương Đình trong mùa nồm ẩm 2025 - 2026. Dữ liệu này giúp mô hình thích nghi chính xác với bụi đường và khí hậu Hà Nội.")

    add_h2("Bước 2: Kỹ nghệ đặc trưng vật lý thích ứng nồm ẩm (Feature Engineering)")
    add_para("Dữ liệu thô từ cảm biến được biến đổi thành vector đặc trưng 5 chiều trước khi nạp vào mạng nơ-ron:")
    add_bullet("Nồng độ trung bình cảm biến kép (PM_avg)", "(PM1 + PM2) / 2 giúp triệt tiêu nhiễu dao động cơ học và dòng xoáy không khí trong buồng đo.")
    add_bullet("Tỷ số phân bố kích thước hạt (R_pm)", "Tỷ số PM2.5 / PM10. Khi xảy ra nồm ẩm, hơi nước bám vào làm thay đổi đường kính quang học hiệu dụng, khiến tỷ số này biến động rõ rệt so với ngày khô ráo.")
    add_bullet("Nhiệt độ (T) và Độ ẩm tương đối (RH)", "Lấy trực tiếp từ cảm biến Sensirion SHT40 chuẩn xác cao (±0.2°C, ±1.8% RH).")
    add_bullet("Nhiệt độ điểm sương (T_dewpoint)", "Tính toán thời gian thực theo công thức Magnus-Tetens. Khi độ chênh lệch giữa nhiệt độ môi trường và điểm sương tiến về 0, nguy cơ hơi nước ngưng tụ trên hạt bụi đạt mức tối đa.")

    add_h2("Bước 3: Thiết kế kiến trúc mạng nơ-ron sâu Deep Residual TinyML")
    add_para("Khác với các mạng nơ-ron nông thông thường dễ bị bão hòa, mô hình được thiết kế theo cấu trúc Deep Residual MLP gồm 5 lớp kết nối đầy đủ (Dense Layers) với đúng 78.401 tham số:")
    add_bullet("Cấu trúc lớp mạng", "Input (5 chiều) -> Dense(160, ReLU) -> Dense(160, ReLU) -> Dense(160, ReLU) -> Dense(160, ReLU) -> Output (1 chiều PM2.5 chuẩn).")
    add_bullet("Khối đường tắt (Residual Skip Connections)", "Tích hợp 3 khối đường tắt cộng trực tiếp đầu vào lớp trước vào đầu ra lớp sau (y = F(x) + x). Kỹ thuật này giúp giải quyết hiện tượng tiêu biến gradient khi huấn luyện mạng sâu, cho phép mô hình nắm bắt trọn vẹn các tương tác phi tuyến tính phức tạp của hiện tượng nồm ẩm.")

    add_h2("Bước 4: Chiến lược huấn luyện 2 giai đoạn (Transfer Learning & Fine-tuning)")
    add_bullet("Giai đoạn 1 (Pre-training)", "Huấn luyện trên 112.117 mẫu dữ liệu US EPA để xây dựng bộ trọng số nền tảng, học quy luật tán xạ quang học tổng quát với hàm mất mát Huber Loss.")
    add_bullet("Giai đoạn 2 (Fine-tuning)", "Đóng băng các tầng đầu và tinh chỉnh với tốc độ học nhỏ (Learning Rate = 1e-4) trên 18.000 mẫu dữ liệu thực địa quận Thanh Xuân. Mô hình học thêm đặc tính của bụi giao thông và muội than đặc thù tại Hà Nội.")

    add_h2("Bước 5: Lượng tử hóa sau huấn luyện sang INT8 (Post-Training Quantization - PTQ)")
    add_para("Trọng số dấu phẩy động 32 bit (FP32) được ép kiểu hoàn toàn sang số nguyên có dấu 8 bit (INT8, dải giá trị -128 đến 127). Quá trình này giúp:")
    add_bullet("Giảm 75% kích thước mô hình", "Từ hơn 318 kilobyte xuống còn 78.4 kilobyte, vừa vặn trong bộ nhớ Flash của vi điều khiển.")
    add_bullet("Kích hoạt tập lệnh véc-tơ phần cứng PIE", "Chuyển toàn bộ phép toán nhân tích lũy từ dấu phẩy động sang số nguyên SIMD 128-bit trên chip ESP32-S3, tốc độ suy luận tăng gấp 3 đến 4 lần.")
    add_bullet("Độ chính xác suy hao cực thấp", "Sai lệch giữa mô hình số thực FP32 và mô hình số nguyên INT8 chỉ suy hao chưa đầy 0.4%, bảo toàn nguyên vẹn độ chuẩn xác.")

    add_h2("Bước 6: Xuất mã nguồn C++ nhúng không cấp phát bộ nhớ động (Zero-Heap Engine)")
    add_para("Toàn bộ trọng số INT8 được chuyển đổi thành mảng byte tĩnh trong tệp header C++ model_data.h. Động cơ suy luận TinyML C++ được viết theo chuẩn an toàn tối đa cho hệ thống nhúng:")
    add_bullet("Không dùng bộ nhớ động", "Tuyệt đối không dùng lệnh malloc hay new. Toàn bộ mảng tính toán được ấn định tĩnh trong bộ nhớ BSS khi biên dịch, loại bỏ triệt để rủi ro phân mảnh RAM và tràn bộ nhớ khi trạm chạy nhiều năm liên tục.")
    add_bullet("Thời gian suy luận siêu tốc", "Chỉ mất 0.32 mili-giây cho một lần chạy suy luận trên vi điều khiển ESP32-S3 xung nhịp 240 MHz.")

    add_h1("5. KẾT QUẢ ĐỐI CHUẨN THỰC TẾ & HIỆU NĂNG PHẦN CỨNG")
    add_para("Kết quả kiểm thử mô hình trên tập dữ liệu độc lập tại trạm đo số 334 đường Nguyễn Trãi, quận Thanh Xuân, Hà Nội ghi nhận sự cải thiện vượt bậc:")

    perf_table = [
        ("Cảm biến thô chưa hiệu chuẩn", "-9.76", "75.60 µg/m³", "121.08 µg/m³", "Đo ảo cực nặng, hoàn toàn không tin cậy"),
        ("Mô hình nền tảng US EPA", "0.77", "11.21 µg/m³", "17.82 µg/m³", "Chỉ đúng ở dải ẩm thấp, nồm ẩm vẫn lệch"),
        ("Mô hình tinh chỉnh thực địa của đề tài", "0.9968", "1.25 µg/m³", "2.09 µg/m³", "Tiệm cận độ chính xác trạm chuẩn tham chiếu"),
        ("Riêng dải nồm ẩm cực đoan (RH > 75%)", "0.9966", "1.06 µg/m³", "2.03 µg/m³", "Triệt tiêu hoàn toàn sai số phồng hạt hơi nước")
    ]
    create_table(["Trạng thái / Mô hình đánh giá", "Hệ số R²", "Sai số MAE", "Sai số RMSE", "Nhận xét thực tế"], perf_table,
                 col_widths=[Inches(2.2), Inches(0.8), Inches(1.1), Inches(1.1), Inches(1.3)])

    add_h2("Đo đạc hiệu năng thực tế trên chip ESP32-S3 N16R8:")
    bench_table = [
        ("Dung lượng bộ nhớ Flash chiếm dụng", "78.4 KB", "< 0.5% Flash 16MB", "Cực kỳ nhỏ gọn"),
        ("Dung lượng bộ nhớ RAM tĩnh (SRAM)", "820 Byte", "< 0.2% RAM nội", "Gần như không đáng kể"),
        ("Bộ nhớ động Heap runtime cấp phát", "0 Byte", "0% nguy cơ rò rỉ RAM", "Chuẩn an toàn nhúng tuyệt đối"),
        ("Thời gian thực thi 1 lần suy luận", "0.32 ms", "Nhân Xtensa LX7 @ 240MHz", "Nhanh gấp 3000 lần chu kỳ đo 1s"),
        ("Công suất tiêu thụ trung bình", "160 mW", "Chạy đầy đủ cả 2 nhân CPU", "Hoạt động bền bỉ với pin mặt trời")
    ]
    create_table(["Chỉ số phần cứng", "Số liệu đo đạc", "Tỷ lệ chiếm dụng", "Đánh giá"], bench_table,
                 col_widths=[Inches(2.3), Inches(1.2), Inches(1.7), Inches(1.3)])

    add_h1("6. HƯỚNG DẪN THỰC THI & TÍCH HỢP CODE LÊN ESP32-S3 TỪNG BƯỚC")
    add_para("Quy trình 4 bước đơn giản để nạp mô hình TinyML vào dự án Arduino / PlatformIO / ESP-IDF trên ESP32-S3:")

    add_bullet("Bước 1: Huấn luyện và xuất tệp nhúng C++", "Chạy script Python trong thư mục ai/edge/export_to_c_header.py để tự động sinh tệp model_data_78k.h chứa toàn bộ mảng trọng số INT8 và các hệ số scale.")
    add_bullet("Bước 2: Copy thư viện vào thư mục src của dự án", "Chép 3 tệp model_data_78k.h, tinyml_edge_engine_78k.h và tinyml_edge_engine_78k.cpp vào thư mục mã nguồn firmware.")
    add_bullet("Bước 3: Khởi tạo động cơ trong setup()", "Gọi hàm tinyml_78k_init() một lần duy nhất lúc khởi động chip để nạp cấu hình bộ nhớ tĩnh.")
    add_bullet("Bước 4: Gọi hàm suy luận trong loop() hoặc FreeRTOS Task", "Truyền các thông số cảm biến vào hàm suy luận một dòng duy nhất như mã nguồn mẫu dưới đây:")

    code_snippet = (
        "// Vi du ma nguon goi suy luan TinyML tai bien tren ESP32-S3\n"
        "#include \"tinyml_edge_engine_78k.h\"\n\n"
        "void setup() {\n"
        "    Serial.begin(115200);\n"
        "    tinyml_78k_init(); // Khoi tao dong co TinyML (0 byte heap)\n"
        "}\n\n"
        "void loop() {\n"
        "    float pm1 = read_pms_sensor_1(); // Doc PM2.5 tu cam bien 1\n"
        "    float pm2 = read_pms_sensor_2(); // Doc PM2.5 tu cam bien 2\n"
        "    float temp = read_sht40_temp();  // Nhiet do SHT40 (degC)\n"
        "    float hum  = read_sht40_hum();   // Do am SHT40 (%)\n\n"
        "    // Chay suy luan bu nom am sieu toc (< 0.5 ms tren nhan LX7)\n"
        "    float pm25_calibrated = tinyml_78k_predict(pm1, pm2, temp, hum);\n\n"
        "    Serial.printf(\"Tho: %.1f | Hieu chuan TinyML: %.1f ug/m3\\n\", \n"
        "                  (pm1 + pm2) / 2.0f, pm25_calibrated);\n"
        "    delay(1000);\n"
        "}"
    )
    add_code_block(code_snippet)

    add_h1("7. CƠ CHẾ LƯU ĐỆM CHỐNG MẤT DỮ LIỆU 45 NGÀY TRÊN 8MB PSRAM")
    add_para("Một ưu điểm vượt trội của phần cứng ESP32-S3 N16R8 là tích hợp sẵn 8 megabyte bộ nhớ ngoài PSRAM. Hệ thống tận dụng vùng nhớ này để xây dựng cơ chế chống mất dữ liệu ngoại tuyến:")
    add_bullet("Cấu trúc bản ghi gọn nhẹ (32 bytes / mẫu)", "Mỗi bản ghi chứa: Dấu thời gian UNIX (4 bytes), PM2.5 thô & sau hiệu chuẩn (8 bytes), PM10 (4 bytes), Nhiệt độ & Độ ẩm (8 bytes), CO2 & VOC (8 bytes).")
    add_bullet("Dung lượng bộ đệm vòng (Circular Ring Buffer 4MB)", "Dành riêng 4MB PSRAM cho bộ đệm, chứa tối đa 131.072 bản ghi. Với chu kỳ ghi 30 giây một lần, trạm có thể lưu liên tục suốt 45 ngày mất mạng.")
    add_bullet("Thuật toán đồng bộ dồn gói (Burst Synchronization)", "Khi có sóng 4G trở lại, vi điều khiển gửi các gói tin dữ liệu bù theo từng cụm (50 bản ghi / gói) về máy chủ Raspberry Pi 5 mà không làm nghẽn băng thông, đảm bảo chuỗi dữ liệu môi trường luôn liền mạch 100%.")

    add_h1("8. KẾ HOẠCH BẢO TRÌ & ĐÁNH GIÁ ĐỘ ỔN ĐỊNH HỆ THỐNG")
    add_bullet("Vệ sinh buồng đo laser định kỳ (6 - 9 tháng/lần)", "Dùng bóng xịt khí nén làm sạch bụi bám trên gương phản xạ của cặp cảm biến Plantower PMS7003 để tránh suy giảm độ nhạy quang học.")
    add_bullet("Bảo dưỡng tấm pin mặt trời và pin lưu trữ LiFePO4", "Lau sạch bề mặt kính tấm pin sau mùa mưa bão. Pack pin LiFePO4 chịu được nhiệt độ cao tới 60°C với tuổi thọ trên 2000 chu kỳ sạc xả (tương đương 5 - 7 năm hoạt động ngoài trời).")
    add_bullet("Giám sát độ lệch cảm biến kép (Delta PM Health-Check)", "Khi độ lệch giữa 2 cảm biến bụi lệch nhau trên 30% liên tục quá 1 giờ, hệ thống tự động bắn tin nhắn cảnh báo qua Telegram về máy của kỹ thuật viên để kiểm tra quạt hút gió hoặc mạng nhện bám buồng đo.")
    add_bullet("Nhiệt độ chip và độ ổn định", "Nhờ thiết kế tối ưu hóa 0 byte Heap, mức chiếm dụng bộ nhớ RAM của ESP32-S3 luôn phẳng tuyệt đối ở mức 820 bytes, nhiệt độ chip duy trì dưới 45°C, đảm bảo hộp vận hành liên tục 24/7/365 mà không bao giờ bị treo chip.")

    output_path = os.path.abspath(r"d:\AQ\Tai_Lieu_Ky_Thuat_Huong_Di_Quy_Trinh_Edge_AI_Hop_Quan_Trac.docx")
    doc.save(output_path)
    print(f"SUCCESS: Report saved to {output_path}")

if __name__ == "__main__":
    create_edge_ai_report()
