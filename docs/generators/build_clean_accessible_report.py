"""
Script tao Bao cao ban dau De tai NCKH ve TinyML tren Hop quan trac
Van phong: Thuc te, mach lac, de hieu, chuan phong cach ky su thuc chien, khong cao sieu han lam.
Thong so ky thuat: Khop 100% codebase train_robust_resmlp_19k.py va tinyml_edge_engine.cpp.
"""

import os
import shutil
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "docs", "figures")

def create_report():
    doc = docx.Document()

    sections = doc.sections
    for section in sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(0.8)

        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("BÁO CÁO BAN ĐẦU ĐỀ TÀI NCKH: TRÍ TUỆ NHÂN TẠO TẠI BIÊN (TINYML)")
        hrun.font.name = "Times New Roman"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(0x71, 0x80, 0x96)

        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        frun = fp.add_run("Trang ")
        frun.font.name = "Times New Roman"
        frun.font.size = Pt(9)
        frun.font.color.rgb = RGBColor(0x71, 0x80, 0x96)

        fld_xml1 = parse_xml(r'<w:fldSimple %s w:instr="PAGE"/>' % nsdecls('w'))
        fp._p.append(fld_xml1)

        frun2 = fp.add_run(" / ")
        frun2.font.name = "Times New Roman"
        frun2.font.size = Pt(9)
        frun2.font.color.rgb = RGBColor(0x71, 0x80, 0x96)

        fld_xml2 = parse_xml(r'<w:fldSimple %s w:instr="NUMPAGES"/>' % nsdecls('w'))
        fp._p.append(fld_xml2)

    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.25
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

    def add_p(text="", bold=False, italic=False, size=12, color=(0x22, 0x22, 0x22),
              align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=4, indent=True):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.25
        if indent and align == WD_ALIGN_PARAGRAPH.JUSTIFY:
            p.paragraph_format.first_line_indent = Inches(0.3)
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
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Times New Roman"
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
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1F, 0x5C, 0x99)
        return p

    def add_bullet(lead, text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.18)

        r_bullet = p.add_run("•  ")
        r_bullet.font.name = "Times New Roman"
        r_bullet.font.size = Pt(12)
        r_bullet.font.bold = True
        r_bullet.font.color.rgb = RGBColor(0x0F, 0x35, 0x62)

        if lead:
            r_lead = p.add_run(lead + ": ")
            r_lead.font.name = "Times New Roman"
            r_lead.font.size = Pt(12)
            r_lead.font.bold = True
            r_lead.font.color.rgb = RGBColor(0x11, 0x11, 0x11)

        r_text = p.add_run(text)
        r_text.font.name = "Times New Roman"
        r_text.font.size = Pt(12)
        r_text.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
        return p

    def make_callout(title, text, accent_color="0F3562", bg_color="F2F5F9"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False

        cell = tbl.rows[0].cells[0]
        cell.width = Inches(6.3)
        set_cell_shading(cell, bg_color)
        set_cell_margins(cell, top=90, bottom=90, left=130, right=130)

        tcPr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'  <w:top w:val="none"/>'
            f'  <w:left w:val="single" w:sz="32" w:space="0" w:color="{accent_color}"/>'
            f'  <w:bottom w:val="none"/>'
            f'  <w:right w:val="none"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)

        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.20

        run_t = p.add_run(f"ĐIỂM THEN CHỐT: {title.upper()}\n")
        run_t.font.name = "Times New Roman"
        run_t.font.size = Pt(11)
        run_t.font.bold = True
        run_t.font.color.rgb = RGBColor(0x0F, 0x35, 0x62)

        run_b = p.add_run(text)
        run_b.font.name = "Times New Roman"
        run_b.font.size = Pt(11)
        run_b.font.color.rgb = RGBColor(0x2A, 0x2A, 0x2A)

        p_sp = doc.add_paragraph()
        p_sp.paragraph_format.space_before = Pt(0)
        p_sp.paragraph_format.space_after = Pt(2)

    def add_image_box(image_path, caption, width=Inches(5.5)):
        if os.path.exists(image_path):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run()
            run.add_picture(image_path, width=width)

            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(6)
            rcap = p_cap.add_run(caption)
            rcap.font.name = "Times New Roman"
            rcap.font.size = Pt(10.5)
            rcap.font.italic = True
            rcap.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    def add_code_block(code_text):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False

        cell = tbl.rows[0].cells[0]
        cell.width = Inches(6.3)
        set_cell_shading(cell, "F8FAFC")
        set_cell_margins(cell, top=80, bottom=80, left=120, right=120)

        tcPr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
            f'  <w:left w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
            f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
            f'  <w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
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
        run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

        p_sp = doc.add_paragraph()
        p_sp.paragraph_format.space_before = Pt(0)
        p_sp.paragraph_format.space_after = Pt(3)

    def create_table(headers, rows_data, col_widths=None):
        tbl = doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        set_table_borders(tbl, color="CBD5E1", sz="4")

        trPr = tbl.rows[0]._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

        for i, h_text in enumerate(headers):
            cell = tbl.rows[0].cells[i]
            if col_widths and i < len(col_widths):
                cell.width = col_widths[i]
            set_cell_shading(cell, "0F3562")
            set_cell_margins(cell, top=90, bottom=90, left=110, right=110)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(h_text)
            r.font.name = "Times New Roman"
            r.font.size = Pt(11)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        for row_idx, r_data in enumerate(rows_data):
            row = tbl.rows[row_idx + 1]
            trPr_row = row._tr.get_or_add_trPr()
            trPr_row.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

            bg = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
            for col_idx, val in enumerate(r_data):
                cell = row.cells[col_idx]
                if col_widths and col_idx < len(col_widths):
                    cell.width = col_widths[col_idx]
                set_cell_shading(cell, bg)
                set_cell_margins(cell, top=70, bottom=70, left=110, right=110)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)

                if col_idx == 0 and len(val) < 20:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT

                r = p.add_run(str(val))
                r.font.name = "Times New Roman"
                r.font.size = Pt(10.5)
                r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

        p_sp = doc.add_paragraph()
        p_sp.paragraph_format.space_before = Pt(0)
        p_sp.paragraph_format.space_after = Pt(3)
        return tbl

    add_p("BÁO CÁO BAN ĐẦU ĐỀ TÀI NGHIÊN CỨU KHOA HỌC", bold=True, size=11, color=(0x55, 0x55, 0x55),
          align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
    add_p("NGHIÊN CỨU XÂY DỰNG VÀ TRIỂN KHAI MÔ HÌNH HỌC MÁY TẠI BIÊN (TINYML)\nHIỆU CHUẨN CẢM BIẾN BỤI MỊN TRÊN HỘP QUAN TRẮC MÔI TRƯỜNG",
          bold=True, size=14, color=(0x0F, 0x35, 0x62), align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=6, indent=False)
    add_p("Hướng tiếp cận: Hiệu chuẩn sai số nồm ẩm phi tuyến tính bằng mạng nơ-ron học sâu RobustResMLP 19k tham số trên vi điều khiển ESP32-S3",
          italic=True, size=11, color=(0x33, 0x33, 0x33), align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8, indent=False)

    add_p("―" * 56, bold=True, size=10, color=(0x0F, 0x35, 0x62), align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10, indent=False)

    add_h1("1. TỔNG QUAN ĐỀ TÀI & MỤC TIÊU PHÁT TRIỂN")
    add_p("Trong các hệ thống quan trắc chất lượng không khí ngoài trời, cảm biến bụi quang học giá rẻ (như dòng Plantower PMS7003 / PMS5003) đóng vai trò then chốt giúp nhân rộng mạng lưới đo đạc với chi phí thấp. Tuy nhiên, nhược điểm chí mạng của loại cảm biến này là hiện tượng 'đo ảo': vào những ngày thời tiết nồm ẩm hoặc sương mù (độ ẩm không khí vượt trên 75%), các hạt bụi ngậm hơi nước phồng to lên khiến cảm biến laser đếm sai kích thước, làm số đo PM2.5 vọt lên gấp 2 đến 3 lần so với thực tế.")
    add_p("Đề tài này tập trung giải quyết triệt để bài toán trên bằng giải pháp Trí tuệ nhân tạo tại biên (TinyML / Edge AI). Thay vì phải lắp đặt các bộ sấy nhiệt cồng kềnh tốn điện hoặc phụ thuộc vào máy chủ đám mây, một mô hình mạng nơ-ron sâu siêu nhẹ (RobustResMLP 19k tham số) sẽ được nạp trực tiếp vào vi điều khiển ESP32-S3 ngay trong hộp đo. Chip sẽ tự động đọc dữ liệu bụi, nhiệt độ, độ ẩm và tự sửa sai số ngay tại chỗ trước khi truyền số liệu đi.")

    add_p("Mục tiêu kỹ thuật cốt lõi của đề tài:", bold=True, indent=False)
    add_bullet("Triệt tiêu sai số nồm ẩm tại biên", "Mô hình AI tự động nhận diện và hạ nồng độ bụi đo ảo xuống sát giá trị thực tế của trạm chuẩn, giảm thiểu ít nhất 65% sai số do hơi ẩm gây ra.")
    add_bullet("Hoạt động hoàn toàn độc lập (In-situ Edge AI)", "Toàn bộ phép tính suy luận AI diễn ra trực tiếp trên chip ESP32-S3 trong vòng chưa đến 0.5 mili-giây, không cần internet, không tốn chi phí thuê máy chủ.")
    add_bullet("Tối ưu hóa tài nguyên phần cứng nhúng", "Mô hình được lượng tử hóa INT8 chỉ chiếm khoảng 19 KB bộ nhớ Flash (chưa đầy 0.12% dung lượng chip), không sử dụng bộ nhớ động Heap (0 Byte Heap) để bảo đảm vi điều khiển chạy liên tục nhiều năm không bao giờ bị tràn bộ nhớ hay treo máy.")
    add_bullet("Giám sát và bảo vệ cảm biến kép (Dual PM)", "Thuật toán liên tục so sánh chéo giữa hai buồng đo bụi chạy song song, tự động phát hiện khi có một cảm biến bị đọng nước hoặc côn trùng chui vào để kích hoạt cảnh báo bảo trì.")

    add_h1("2. BỐI CẢNH VÀ TÍNH CẤP THIẾT CỦA BÀI TOÁN")

    add_h2("2.1. Thực trạng cảm biến bụi quang học bị đo ảo do thời tiết nồm ẩm")
    add_p("Cảm biến bụi quang học hoạt động theo nguyên lý tán xạ laser: không khí được quạt hút vào một buồng tối, tia laser chiếu qua dòng khí và cảm biến quang sẽ ghi nhận các xung ánh sáng lóe lên khi gặp hạt bụi để tính ra nồng độ PM2.5. Thuật toán mặc định trong cảm biến của nhà sản xuất được hiệu chuẩn trong điều kiện phòng thí nghiệm khô ráo (độ ẩm khoảng 40% đến 50%).")
    add_p("Tuy nhiên, điều kiện khí hậu nhiệt đới gió mùa tại Việt Nam, đặc biệt là mùa nồm ẩm miền Bắc vào mùa xuân, độ ẩm không khí thường xuyên duy trì ở mức 80% đến xấp xỉ 100%. Khi đó xuất hiện hai hiện tượng vật lý gây sai số nghiêm trọng:")
    add_bullet("Hạt bụi hút ẩm phồng to", "Các hạt bụi mịn có tính háo nước sẽ hút hơi nước xung quanh và tăng kích thước hình học lên gấp nhiều lần. Cảm biến laser chỉ nhìn thấy tiết diện tán xạ ánh sáng lớn nên lầm tưởng đó là hạt bụi khổng lồ.")
    add_bullet("Hạt sương mù li ti bay vào buồng đo", "Các hạt sương nước siêu nhỏ có kích thước tương đương hạt bụi lọt vào buồng quang học, khiến cảm biến đếm nhầm sương thành bụi mịn.")
    add_p("Hậu quả là vào những ngày trời nồm nhiều sương, cảm biến giá rẻ báo nồng độ PM2.5 lên tới 150 - 200 µg/m³ (mức tím nguy hại), trong khi trạm đo chuẩn của trạm khí tượng quốc gia chỉ ghi nhận khoảng 40 - 50 µg/m³. Sai số này khiến dữ liệu quan trắc mất hoàn toàn độ tin cậy.")

    add_h2("2.2. So sánh các hướng giải quyết và lý do lựa chọn TinyML tại biên")
    add_p("Để khắc phục sai số độ ẩm, các nhóm nghiên cứu trong nước và quốc tế đã thử nghiệm nhiều phương án khác nhau. Dưới đây là bảng so sánh phân tích ưu và nhược điểm của từng hướng tiếp cận:")

    comp_rows = [
        ("Sấy nhiệt phần cứng (Heated Inlet)",
         "Dùng điện trở nhiệt đun nóng dòng khí nạp lên 40-50°C để sấy khô sương trước khi vào buồng đo.",
         "Làm bay hơi sương rất tốt nhưng tiêu thụ công suất rất lớn (5W - 15W), không thể chạy bằng pin mặt trời; làm thiết bị cồng kềnh, dễ cháy hỏng ngoài trời."),
        ("Công thức kinh nghiệm hồi quy tuyến tính",
         "Sử dụng công thức toán tuyến tính đơn giản kết hợp độ ẩm để trừ bớt giá trị đo.",
         "Tính toán nhẹ nhưng độ chính xác rất kém vì hiện tượng ngậm nước của bụi là phi tuyến tính mạnh; khi độ ẩm trên 80% công thức tuyến tính bị sai lệch hoàn toàn."),
        ("Xử lý AI trên máy chủ đám mây (Cloud AI)",
         "Hộp đo gửi dữ liệu thô qua 4G về server để chạy các mô hình học máy phức tạp sửa sai số.",
         "Xử lý được thuật toán phức tạp nhưng phụ thuộc 100% vào đường truyền mạng; nếu mất sóng hoặc nghẽn mạng thì mất dữ liệu; tốn chi phí duy trì máy chủ hàng tháng."),
        ("Trí tuệ nhân tạo tại biên (TinyML trên ESP32-S3) — Hướng đi của đề tài",
         "Nén mô hình mạng nơ-ron sâu học từ trạm chuẩn thành mã nhúng C++ chạy trực tiếp trên vi điều khiển.",
         "Khắc phục triệt để sai số phi tuyến; chạy độc lập tại chỗ không cần mạng; tiêu thụ điện cực thấp (< 0.5W); dữ liệu xuất ra tức thì đạt chuẩn ngay tại chân cảm biến.")
    ]
    create_table(["Hướng tiếp cận", "Nguyên lý kỹ thuật", "Đánh giá thực tế"], comp_rows,
                 col_widths=[Inches(1.8), Inches(2.2), Inches(2.3)])

    make_callout("Tại sao phải dùng TinyML trực tiếp trên ESP32-S3?",
                 "Xử lý tại biên (Edge AI) giải quyết tận gốc tính toàn vẹn của dữ liệu. Thiết bị quan trắc có thể lắp đặt ở bất kỳ đâu — từ cột đèn, trang trại đến vùng xa xôi không có sóng di động — nhưng số liệu đo xuất ra màn hình tại chỗ hay lưu vào thẻ nhớ vẫn luôn là số liệu đã được hiệu chuẩn chính xác, không còn nguy cơ bị 'ảo' do thời tiết nồm ẩm.")

    add_h1("3. KIẾN TRÚC VÀ NGUYÊN LÝ MÔ HÌNH AI (ROBUST RESMLP 19K)")

    add_h2("3.1. Nguồn dữ liệu huấn luyện đối chứng chuẩn quốc tế (Ground Truth)")
    add_p("Để huấn luyện một mô hình AI có độ tin cậy cao, việc sở hữu dữ liệu chuẩn đối chứng (Ground Truth) là yếu tố quyết định sống còn. Đề tài sử dụng bộ dữ liệu thực nghiệm quy mô lớn gồm 112.117 mẫu đo đối chứng được thu thập đồng bộ giữa cảm biến quang học giá rẻ và thiết bị đo chuẩn quốc tế Met One BAM-1020 (dòng máy đo khối lượng bụi theo nguyên lý suy giảm tia Beta đạt chuẩn EPA Hoa Kỳ).")
    add_p("Bộ dữ liệu bao phủ đầy đủ các điều kiện thời tiết khắc nghiệt tại các trạm đo Đông Nam Á và mạng lưới US EPA, với dải nhiệt độ từ 5°C đến 42°C và dải độ ẩm tương đối từ 20% đến 99%. Nhờ đó, mô hình học được bản chất quy luật sai lệch của cảm biến quang học ở mọi kịch bản thời tiết thực tế.")

    add_h2("3.2. Năm thông số đầu vào có ý nghĩa thực tế")
    add_p("Mô hình tiếp nhận một vector đầu vào gồm đúng 5 thông số vi khí hậu được tính toán tức thời tại mỗi chu kỳ đo:")
    add_bullet("pm_avg", "Nồng độ bụi trung bình từ hai cảm biến PMS7003 chạy song song (đơn vị: µg/m³). Việc lấy trung bình giúp giảm nhiễu ngẫu nhiên của từng buồng đo riêng lẻ.")
    add_bullet("diff_pm", "Độ chênh lệch tuyệt đối giữa hai cảm biến (|PMS1 - PMS2|). Đây là thông số quan trọng giúp AI nhận biết xem có một buồng đo nào đang bị đọng nước bất thường hay không.")
    add_bullet("temp", "Nhiệt độ môi trường (°C) đo từ cảm biến chuyên dụng Sensirion SHT40 độ chính xác cao.")
    add_bullet("rh", "Độ ẩm tương đối của không khí (%) đo từ cảm biến Sensirion SHT40.")
    add_bullet("dew_point", "Nhiệt độ điểm sương (°C), được tính toán theo công thức Magnus từ nhiệt độ và độ ẩm. Khoảng cách giữa nhiệt độ không khí và nhiệt độ điểm sương cho biết không khí đang gần mức bão hòa hơi nước đến mức nào.")

    add_h2("3.3. Cấu trúc mạng nơ-ron học sâu Robust ResMLP 19.297 tham số")
    add_p("Mô hình được thiết kế theo cấu trúc mạng nơ-ron truyền thẳng sâu có liên kết tắt (Residual Multilayer Perceptron - ResMLP) với tổng cộng đúng 19.297 tham số. Đây là kích thước 'điểm vàng' (Sweet Spot) được tính toán tối ưu riêng cho kiến trúc vi điều khiển ESP32-S3:")
    add_bullet("Tầng đầu vào (fc_in)", "Tiếp nhận 5 đặc trưng đầu vào, chuẩn hóa theo phân phối chuẩn Z-score và mở rộng lên 96 nơ-ron qua hàm kích hoạt ReLU.")
    add_bullet("Hai khối liên kết tắt liên tiếp (Residual Blocks)", "Mỗi khối gồm 96 nơ-ron. Điểm đặc biệt của khối Residual là đường dẫn tắt (Skip Connection) cộng trực tiếp tín hiệu đầu vào khối với tín hiệu sau biến đổi: h = ReLU(W * x + b + x). Cơ chế này giúp mạng nơ-ron học sâu nhưng không bị suy biến gradient, bảo toàn các đặc trưng vật lý quan trọng.")
    add_bullet("Tầng đầu ra (fc_out)", "Thu hẹp từ 96 chiều về 1 giá trị đầu ra duy nhất để tính toán hệ số hiệu chuẩn.")

    add_image_box(os.path.join(FIGURES_DIR, "fig1_robust_resmlp_architecture.png"),
                  "Hình 1: Sơ đồ kiến trúc tổng thể của mô hình học sâu Robust ResMLP 19k tham số")

    add_image_box(os.path.join(FIGURES_DIR, "fig2_residual_block_skip_connection.png"),
                  "Hình 2: Cơ chế liên kết tắt (Skip Connection) trong khối Residual giúp bảo toàn tín hiệu")

    add_h2("3.4. Điểm cốt lõi: Dự đoán hệ số co có chặn vật lý thay vì đoán mò nồng độ bụi")
    add_p("Một sai lầm rất phổ biến trong các nghiên cứu AI cảm biến trước đây là để mô hình hồi quy trực tiếp ra nồng độ bụi dự đoán. Khi huấn luyện mô hình dự đoán trực tiếp nồng độ bụi, nếu gặp dữ liệu thời tiết ngoài vùng huấn luyện, mạng nơ-ron rất dễ dự đoán ra số bụi âm (< 0 µg/m³) hoặc những con số nhảy vọt phi lý, gây nguy hiểm cho hệ thống giám sát.")
    add_p("Để giải quyết triệt để rủi ro này, đề tài áp dụng cơ chế dự đoán Hệ số hiệu chỉnh có chặn vật lý (Physics-Informed Bounded Output):")
    add_bullet("Mô hình chỉ dự đoán hệ số co C", "Mạng nơ-ron không đoán số bụi mà chỉ dự đoán hệ số hiệu chỉnh C thông qua hàm kích hoạt Sigmoid có chặn:")
    add_p("C = Sigmoid(z) * 0.80 + 0.25", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=4, indent=False)
    add_bullet("Khoảng biến thiên an toàn", "Nhờ chặn vật lý, hệ số C luôn luôn nằm chặt chẽ trong khoảng từ 0.25 đến 1.05. Sau đó, nồng độ bụi chuẩn được tính bằng: PM_calib = PM_avg * C.")
    add_bullet("Ý nghĩa vật lý", "Khi trời khô ráo không có sương, C xấp xỉ 0.95 - 1.0 (giữ nguyên số đo gốc của cảm biến). Khi nồm ẩm nặng làm hạt bụi ngậm nước phồng to gấp 2 - 3 lần, C tự động hạ xuống mức 0.35 - 0.45 để kéo số đo về đúng thực tế. Hệ thống bảo đảm 100% không bao giờ sinh ra số bụi âm hoặc biến dạng dữ liệu.")

    add_image_box(os.path.join(FIGURES_DIR, "fig4_bounded_output_activation.png"),
                  "Hình 3: Hàm kích hoạt Sigmoid có chặn giới hạn hệ số C nghiêm ngặt trong khoảng [0.25, 1.05]")

    add_h2("3.5. Cổng hòa trộn vi khí hậu thích ứng (Adaptive Humidity Gating)")
    add_p("Trong điều kiện thời tiết khô ráo bình thường (độ ẩm dưới 55%), các cảm biến bụi quang học của nhà sản xuất vốn đã hoạt động tương đối chính xác. Nếu AI liên tục can thiệp vào giai đoạn này sẽ có nguy cơ làm lệch số liệu chuẩn ban đầu.")
    add_p("Đề tài xây dựng một Cổng hòa trộn thích ứng (Adaptive Humidity Gating) với tâm chuyển tiếp tại mốc 55% độ ẩm tương đối:")
    add_bullet("Khi độ ẩm dưới 55%", "Cổng hòa trộn đóng hoàn toàn, hệ số C được cố định ở mức 0.95 (nhường quyền kiểm soát cho cảm biến gốc, giữ nguyên độ nhạy thiết bị).")
    add_bullet("Khi độ ẩm từ 55% đến trên 75%", "Cổng hòa trộn mở dần một cách mượt mà theo hàm phi tuyến, kích hoạt toàn lực mô hình AI để loại bỏ hiện tượng đo ảo khi bước vào vùng nồm ẩm cao.")

    add_image_box(os.path.join(FIGURES_DIR, "fig3_adaptive_humidity_gating.png"),
                  "Hình 4: Đường cong kích hoạt của Cổng hòa trộn độ ẩm thích ứng với tâm chuyển tiếp 55% RH")

    add_h1("4. THUẬT TOÁN KHỬ NHIỄU VÀ BẢO VỆ CẢM BIẾN KÉP")

    add_h2("4.1. Cơ chế đối chiếu chéo giữa hai cảm biến bụi (Dual PM)")
    add_p("Hộp quan trắc được thiết kế với hai cảm biến bụi PMS7003 hoạt động độc lập đặt cạnh nhau. Việc sử dụng cảm biến kép mang lại lợi thế vượt trội so với các thiết bị đo đơn lẻ thông thường:")
    add_bullet("Bình thường (diff_pm <= 25 µg/m³)", "Hai cảm biến hoạt động đồng thuận. Hệ thống lấy trung bình cộng để triệt tiêu nhiễu lượng tử ngẫu nhiên của chùm laser, nâng cao độ mịn của tín hiệu đo.")
    add_bullet("Bất thường (diff_pm > 25 µg/m³)", "Hai cảm biến có sự lệch pha rõ rệt. Đây là dấu hiệu chắc chắn cho thấy một trong hai cảm biến đang gặp sự cố cục bộ (như bị giọt nước đọng trong buồng đo, bám bụi bẩn lâu ngày hoặc có côn trùng chui vào làm nghẽn quạt hút).")

    add_h2("4.2. Xử lý tự động khi buồng đo bị đọng sương hoặc bám bẩn")
    add_p("Khi phát hiện độ lệch diff_pm vượt ngưỡng an toàn 25 µg/m³, thuật toán xử lý tại biên thực hiện ngay 3 hành động tự động:")
    add_bullet("Bật cờ cảnh báo bất thường", "Biến is_anomaly được gán bằng true, gửi kèm theo gói tin dữ liệu để thông báo cho kỹ thuật viên biết vị trí trạm cần vệ sinh bảo trì.")
    add_bullet("Chuyển sang buồng đo sạch hơn", "Thuật toán tự động bỏ qua cảm biến có giá trị vọt cao bất thường (buồng đo đang bị dính sương đo ảo) và chọn giá trị của cảm biến thấp hơn (buồng đo thông thoáng hơn) làm giá trị cơ sở.")
    add_bullet("Đưa vào mô hình AI hiệu chuẩn", "Giá trị đã lọc tiếp tục được mô hình AI bù sai số độ ẩm. Nhờ đó, trạm đo vẫn xuất ra số liệu nồng độ bụi ổn định, không bị sốc số liệu hoặc gửi dữ liệu rác về hệ thống trung tâm.")

    add_h1("5. QUY TRÌNH NÉN VÀ TRIỂN KHAI C++ TRÊN ESP32-S3")

    add_h2("5.1. Kỹ thuật lượng tử hóa INT8 sau huấn luyện (Post-Training Quantization)")
    add_p("Trên máy tính phát triển, mô hình được huấn luyện bằng PyTorch với định dạng số thực 32-bit (FP32), chiếm khoảng 77 KB bộ nhớ và đòi hỏi nhiều phép tính dấu phẩy động phức tạp. Để đưa mô hình lên vi điều khiển ESP32-S3 một cách tối ưu, đề tài áp dụng kỹ thuật lượng tử hóa đối xứng sau huấn luyện (INT8 PTQ):")
    add_bullet("Chuyển đổi trọng số sang số nguyên", "Toàn bộ trọng số của 19.297 tham số được ánh xạ về dạng số nguyên 8-bit có dấu trong dải [-128, 127] cùng các hệ số tỷ lệ co dãn (Scale Factors).")
    add_bullet("Tiết kiệm bộ nhớ vượt bậc", "Mô hình nén lại chỉ còn đúng 18.84 KB Flash, tương đương chưa đầy 0.12% dung lượng bộ nhớ của chip ESP32-S3.")
    add_bullet("Bảo toàn độ chính xác", "Sai số tính toán giữa mô hình số thực FP32 và mô hình số nguyên INT8 sau lượng tử hóa nhỏ hơn 0.3%, hoàn toàn không ảnh hưởng đến chất lượng hiệu chuẩn.")

    add_h2("5.2. Động cơ suy luận C++ nhúng Zero-Heap (Không cấp phát bộ nhớ động)")
    add_p("Một trong những nguyên nhân hàng đầu khiến các thiết bị IoT bị treo máy ngoài thực địa là hiện tượng phân mảnh bộ nhớ RAM do việc liên tục xin cấp phát và giải phóng bộ nhớ động (lệnh malloc / free hoặc new / delete). Đề tài xây dựng một động cơ suy luận viết bằng C++ thuần độc lập hoàn toàn, tuân thủ kỷ luật phần mềm nhúng khắt khe:")
    add_bullet("Bộ nhớ động Heap runtime = 0 Byte", "Tuyệt đối không sử dụng bất kỳ lệnh cấp phát bộ nhớ động nào trong toàn bộ quá trình chạy suy luận của mô hình AI.")
    add_bullet("Bộ nhớ ngăn xếp Stack < 1 KB", "Toàn bộ các mảng trung gian chứa nơ-ron (96 phần tử) được khai báo tĩnh (static buffer) ngay trong ngăn xếp tạm thời với dung lượng tiêu thụ chưa tới 1 Kilobyte.")
    add_bullet("Độ tin cậy tuyệt đối 24/7", "Vi điều khiển có thể vận hành liên tục nhiều năm liền trong môi trường khắc nghiệt mà không bao giờ gặp nguy cơ tràn bộ nhớ RAM (Memory Leak) hay phân mảnh bộ nhớ.")

    add_h2("5.3. Cách nhúng vào luồng đo của firmware ESP32-S3")
    add_p("Toàn bộ động cơ AI được đóng gói gọn trong hai tệp mã nguồn: model_data.h (chứa bảng trọng số INT8) và tinyml_edge_engine.cpp (chứa hàm tính toán). Việc tích hợp vào mã nguồn chính của thiết bị vô cùng đơn giản:")

    code_txt = (
        "// Tich hop dong co suy luan TinyML vao vong lap firmware ESP32-S3\n"
        "#include \"tinyml_edge_engine.h\"\n\n"
        "void setup() {\n"
        "    Serial.begin(115200);\n"
        "}\n\n"
        "void loop() {\n"
        "    // 1. Doc gia tri tho tu cac cam bien\n"
        "    float pm1_raw  = read_pms_sensor_1(); // So do bui tu PMS7003 so 1\n"
        "    float pm2_raw  = read_pms_sensor_2(); // So do bui tu PMS7003 so 2\n"
        "    float temp_c   = read_sht40_temp();   // Nhiet do tu SHT40 (degC)\n"
        "    float rh_pct   = read_sht40_hum();    // Do am tu SHT40 (%)\n"
        "    bool is_anomaly = false;\n\n"
        "    // 2. Chay TinyML bu sai so nom am va loc cam bien kep (~0.2 ms)\n"
        "    float pm25_calib = tinyml_19k_predict_pm25(pm1_raw, pm2_raw, \n"
        "                                               temp_c, rh_pct, \n"
        "                                               &is_anomaly);\n\n"
        "    // 3. Xuat ket qua da duoc lam sach\n"
        "    if (is_anomaly) {\n"
        "        Serial.println(\"CANH BAO: Phat hien bat thuong buong do cam bien!\");\n"
        "    }\n"
        "    Serial.printf(\"Bui tho: %.1f -> Bui sau AI: %.1f ug/m3\\n\", \n"
        "                  (pm1_raw + pm2_raw) * 0.5f, pm25_calib);\n\n"
        "    delay(1000); // Chu ky do tiep theo\n"
        "}"
    )
    add_code_block(code_txt)

    add_h1("6. KẾ HOẠCH THỬ NGHIỆM VÀ TIÊU CHÍ NGHIỆM THU KỲ VỌNG")
    add_p("Là một đề tài nghiên cứu ban đầu, nhóm nghiên cứu xác lập rõ ràng phương pháp thử nghiệm thực nghiệm và bảng tiêu chí nghiệm thu định lượng để hội đồng khoa học theo dõi và đánh giá kết quả sau khi hoàn thành.")

    add_h2("6.1. Quy trình thử nghiệm thực nghiệm hai bước")
    add_bullet("Bước 1: Kiểm thử trong buồng vi khí hậu nhân tạo",
               "Hộp quan trắc được đưa vào buồng thử nghiệm có thể chủ động điều khiển nhiệt độ (15°C - 35°C) và phun sương tạo độ ẩm biến thiên từ 50% đến 95% RH. Thử nghiệm này nhằm kiểm tra phản ứng của cổng hòa trộn và tính ổn định của hệ số C khi độ ẩm thay đổi đột ngột.")
    add_bullet("Bước 2: Đo đạc đối chứng thực địa dã ngoại (Collocation Field Testing)",
               "Lắp đặt hộp quan trắc chạy song song trong 30 ngày liên tục ngay cạnh trạm quan trắc chuẩn không khí quốc gia. Dữ liệu sau khi qua mô hình TinyML tại biên sẽ được lưu trữ và đối chiếu trực tiếp với số đo của trạm chuẩn để tính toán các chỉ số thống kê.")

    add_h2("6.2. Bảng tiêu chí cam kết nghiệm thu định lượng")
    add_p("Đề tài cam kết đạt được các chỉ tiêu định lượng cụ thể sau khi kết thúc quá trình nghiên cứu và thực nghiệm:")

    metrics_rows = [
        ("Hệ số tương quan (R²)",
         "Đo lường mức độ bám sát của số đo sau AI so với trạm đo chuẩn quốc tế.",
         "R² >= 0.90 (tăng mạnh so với cảm biến thô chỉ đạt R² ~ 0.55 - 0.65)."),
        ("Sai số tuyệt đối trung bình (MAE)",
         "Đo độ lệch nồng độ trung bình giữa cảm biến và máy đo chuẩn.",
         "MAE <= 3.5 µg/m³ (trong dải nồng độ bụi thông thường)."),
        ("Tỷ lệ triệt tiêu sai số nồm ẩm",
         "Mức độ kéo giảm sai số đo ảo trong điều kiện độ ẩm cao (RH > 80%).",
         "Giảm thiểu ít nhất 65% hiện tượng tăng ảo nồng độ do sương ẩm."),
        ("Dung lượng bộ nhớ Flash ESP32-S3",
         "Không gian lưu trữ mô hình và bảng trọng số nhúng trong chip.",
         "Dưới 25 KB Flash (mô hình INT8 thực tế chỉ chiếm 18.84 KB)."),
        ("Bộ nhớ động Heap runtime",
         "Lượng RAM động tiêu tốn trong suốt quá trình chạy mô hình AI.",
         "0 Byte Heap (cấm tuyệt đối cấp phát động, chống phân mảnh RAM)."),
        ("Thời gian thực thi một phép tính",
         "Thời gian chip xử lý xong một chu kỳ đọc và hiệu chuẩn nồng độ bụi.",
         "Dưới 0.5 mili-giây (thực tế chạy trên ESP32-S3 240MHz chỉ mất ~0.2 ms).")
    ]
    create_table(["Chỉ tiêu kỹ thuật", "Ý nghĩa đo lường", "Chỉ số cam kết nghiệm thu"], metrics_rows,
                 col_widths=[Inches(1.8), Inches(2.3), Inches(2.2)])

    add_h1("7. KẾ HOẠCH NGHIÊN CỨU VÀ CÁC MỐC TIẾN ĐỘ DỰ KIẾN")
    add_p("Kế hoạch thực hiện đề tài được phân kỳ rõ ràng qua 5 giai đoạn nối tiếp nhau:")

    roadmap_rows = [
        ("Giai đoạn 1: Chuẩn bị dữ liệu & Kỹ nghệ đặc trưng",
         "Làm sạch 112.000 mẫu đo đối chứng chuẩn EPA, trích xuất điểm sương và chuẩn hóa vector đầu vào.",
         "Bộ dữ liệu sạch và pipeline trích xuất đặc trưng tự động."),
        ("Giai đoạn 2: Thiết kế mạng & Huấn luyện mô hình",
         "Xây dựng kiến trúc RobustResMLP 19k tham số, tích hợp hàm kích hoạt chặn vật lý và cổng hòa trộn độ ẩm thích ứng.",
         "Mô hình học sâu hoàn chỉnh trên PyTorch và tệp trọng số đã huấn luyện."),
        ("Giai đoạn 3: Lượng tử hóa & Đóng gói động cơ C++",
         "Thực hiện lượng tử hóa INT8 PTQ, sinh tệp hằng số model_data.h và viết động cơ suy luận nhúng Zero-Heap.",
         "Thư viện C++ nhúng hoàn chỉnh sẵn sàng nạp lên vi điều khiển ESP32-S3."),
        ("Giai đoạn 4: Thử nghiệm thực địa & Tinh chỉnh",
         "Triển khai hộp quan trắc chạy dã ngoại 30 ngày đối chứng cạnh trạm quan trắc chuẩn, ghi nhận số liệu đánh giá sai số.",
         "Bộ dữ liệu đối chứng thực tế và báo cáo phân tích độ chính xác."),
        ("Giai đoạn 5: Nghiệm thu & Bàn giao sản phẩm",
         "Tổng hợp kết quả nghiên cứu, hoàn thiện báo cáo tổng kết đề tài và đóng gói tài liệu kỹ thuật bàn giao.",
         "Báo cáo tổng kết đề tài NCKH và gói mã nguồn phần mềm nhúng hoàn chỉnh.")
    ]
    create_table(["Giai đoạn thực hiện", "Nội dung công việc nghiên cứu", "Sản phẩm bàn giao dự kiến"], roadmap_rows,
                 col_widths=[Inches(1.8), Inches(2.7), Inches(1.8)])

    add_h1("8. KẾT LUẬN & ĐỀ XUẤT")
    add_p("Đề tài 'Nghiên cứu xây dựng và triển khai mô hình học máy tại biên (TinyML) hiệu chuẩn cảm biến bụi mịn trên hộp quan trắc môi trường' tập trung giải quyết đúng và trúng điểm nghẽn kỹ thuật lớn nhất của các thiết bị quan trắc không khí giá rẻ hiện nay.")
    add_p("Bằng cách ứng dụng mô hình mạng nơ-ron sâu siêu nhẹ (RobustResMLP 19k) kết hợp với các cơ chế vật lý thông minh (dự đoán hệ số co có chặn, cổng hòa trộn độ ẩm thích ứng và lọc cảm biến kép), đề tài mở ra hướng đi đột phá: nâng tầm chất lượng dữ liệu của cảm biến bụi giá rẻ tiệm cận độ chính xác của trạm đo chuẩn quốc tế mà không làm tăng chi phí phần cứng hay tiêu hao năng lượng.")
    add_p("Nhóm nghiên cứu kính đề nghị Hội đồng Khoa học xem xét phê duyệt đề cương và tạo điều kiện hỗ trợ địa điểm thử nghiệm thực địa đối chứng để đề tài sớm được triển khai và đưa vào ứng dụng thực tiễn.")

    path_root = os.path.abspath(os.path.join(PROJECT_ROOT, "docs", "reports", "Bao_Cao_Ban_Dau_NCKH_Edge_AI.docx"))
    path_tailieu = os.path.abspath(os.path.join(PROJECT_ROOT, "tailieu", "Bao_Cao_Ban_Dau_NCKH_Edge_AI.docx"))

    os.makedirs(os.path.dirname(path_root), exist_ok=True)
    doc.save(path_root)
    os.makedirs(os.path.dirname(path_tailieu), exist_ok=True)
    shutil.copy2(path_root, path_tailieu)

    print(f"SUCCESS: Report saved to:\n1. {path_root}\n2. {path_tailieu}")

if __name__ == "__main__":
    create_report()
