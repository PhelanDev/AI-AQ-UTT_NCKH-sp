"""
Script tao tai lieu: DU KIEN XAY DUNG 2 MO HINH AI (BAN BAO VE BI QUYET CONG NGHE - IP PROTECTED)
Luu dung thu muc quy chuan: docs/reports/Du_Kien_Xay_Dung_2_Mo_Hinh_AI.docx
Nhung 3 so do True-Scale 6.4 inch: Chu to, ro rang, khong bi be teo.
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

def generate_doc():
    doc = docx.Document()

    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.8)

        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("THIẾT KẾ KỸ THUẬT: ĐỀ XUẤT 2 MÔ HÌNH AI")
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
    normal_style.font.size = Pt(11.5)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.25
    normal_style.paragraph_format.space_after = Pt(4)
    normal_style.paragraph_format.space_before = Pt(0)
    normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def set_cell_margins(cell, top=70, bottom=70, left=110, right=110):
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

    def set_table_borders(table, color="CBD5E1", sz="4", val="single"):
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

    def add_p(text="", bold=False, italic=False, size=11.5, color=(0x22, 0x22, 0x22),
              align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=4, indent=True):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.25
        if indent and align == WD_ALIGN_PARAGRAPH.JUSTIFY:
            p.paragraph_format.first_line_indent = Inches(0.25)
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
        p.paragraph_format.space_before = Pt(13)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(12.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x0F, 0x35, 0x62)
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1F, 0x5C, 0x99)
        return p

    def add_bullet(lead, text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.20
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.15)

        r_bullet = p.add_run("•  ")
        r_bullet.font.name = "Times New Roman"
        r_bullet.font.size = Pt(11.5)
        r_bullet.font.bold = True
        r_bullet.font.color.rgb = RGBColor(0x0F, 0x35, 0x62)

        if lead:
            r_lead = p.add_run(lead + ": ")
            r_lead.font.name = "Times New Roman"
            r_lead.font.size = Pt(11.5)
            r_lead.font.bold = True
            r_lead.font.color.rgb = RGBColor(0x11, 0x11, 0x11)

        r_text = p.add_run(text)
        r_text.font.name = "Times New Roman"
        r_text.font.size = Pt(11.5)
        r_text.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
        return p

    def make_callout(title, text, accent_color="0F3562", bg_color="F2F5F9"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False

        cell = tbl.rows[0].cells[0]
        cell.width = Inches(6.4)
        set_cell_shading(cell, bg_color)
        set_cell_margins(cell, top=70, bottom=70, left=110, right=110)

        tcPr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'  <w:top w:val="none"/>'
            f'  <w:left w:val="single" w:sz="28" w:space="0" w:color="{accent_color}"/>'
            f'  <w:bottom w:val="none"/>'
            f'  <w:right w:val="none"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)

        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15

        run_t = p.add_run(f"ĐIỂM SÁNG KIẾN CỐT LÕI: {title.upper()}\n")
        run_t.font.name = "Times New Roman"
        run_t.font.size = Pt(10.5)
        run_t.font.bold = True
        run_t.font.color.rgb = RGBColor(0x0F, 0x35, 0x62)

        run_b = p.add_run(text)
        run_b.font.name = "Times New Roman"
        run_b.font.size = Pt(10.5)
        run_b.font.color.rgb = RGBColor(0x2A, 0x2A, 0x2A)

        p_sp = doc.add_paragraph()
        p_sp.paragraph_format.space_before = Pt(0)
        p_sp.paragraph_format.space_after = Pt(2)

    def add_image_box(image_path, caption, width=Inches(6.0)):
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
            rcap.font.size = Pt(10)
            rcap.font.italic = True
            rcap.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

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
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(h_text)
            r.font.name = "Times New Roman"
            r.font.size = Pt(10.5)
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
                set_cell_margins(cell, top=55, bottom=55, left=90, right=90)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)

                if col_idx == 0 and len(val) < 22:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT

                r = p.add_run(str(val))
                r.font.name = "Times New Roman"
                r.font.size = Pt(10)
                r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

        p_sp = doc.add_paragraph()
        p_sp.paragraph_format.space_before = Pt(0)
        p_sp.paragraph_format.space_after = Pt(2)
        return tbl

    add_p("TÀI LIỆU ĐỀ XUẤT GIẢI PHÁP KỸ THUẬT", bold=True, size=11, color=(0x55, 0x55, 0x55),
          align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
    add_p("DỰ KIẾN KIẾN TRÚC VÀ ĐỊNH HƯỚNG XÂY DỰNG 2 MÔ HÌNH AI\nTRONG HỆ THỐNG QUAN TRẮC MÔI TRƯỜNG THÔNG MINH",
          bold=True, size=13.5, color=(0x0F, 0x35, 0x62), align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=4, indent=False)
    add_p("Phân tầng xử lý: AI tại biên (Edge AI) trên vi điều khiển và AI phân tích chuỗi thời gian trên máy chủ cục bộ",
          italic=True, size=10.5, color=(0x33, 0x33, 0x33), align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6, indent=False)

    add_p("―" * 56, bold=True, size=10, color=(0x0F, 0x35, 0x62), align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6, indent=False)

    add_h1("1. TỔNG QUAN PHÂN TẦNG VÀ ĐỊNH VỊ 2 MÔ HÌNH AI")
    add_p("Kiến trúc xử lý thông minh của hệ thống được tổ chức thành 2 tầng AI chuyên biệt, phân định ranh giới rõ ràng về nhiệm vụ, vị trí thực thi và dữ liệu xử lý:")

    arch_rows = [
        ("Mô hình 1: Hiệu chuẩn & Lọc biên (Edge AI / TinyML)",
         "Vi điều khiển nhúng tại trạm đo (ESP32-S3)",
         "Xử lý làm sạch tín hiệu tại nguồn, bù trừ phi tuyến sai số nồm ẩm và kiểm định chéo độ tin cậy của cảm biến.",
         "Tính toán tức thời theo thời gian thực (In-situ), hoàn toàn ngoại tuyến không phụ thuộc kết nối mạng."),
        ("Mô hình 2: Phân tích & Dự báo chuỗi thời gian (Temporal AI)",
         "Máy chủ biên cục bộ (Raspberry Pi 5 / Edge Server)",
         "Khai phá quy luật thời gian, dự báo xu hướng biến thiên nồng độ ô nhiễm và phát hiện các biến động môi trường bất thường.",
         "Vận hành định kỳ theo chu kỳ lấy mẫu, xử lý đa biến chuỗi thời gian lịch sử nhiều ngày.")
    ]
    create_table(["Phân tầng AI", "Môi trường thực thi", "Nhiệm vụ cốt lõi", "Cơ chế vận hành"], arch_rows,
                 col_widths=[Inches(1.6), Inches(1.5), Inches(2.0), Inches(1.4)])

    make_callout("Nguyên lý bảo toàn chất lượng dữ liệu",
                 "Mô hình 1 giữ vai trò chốt chặn chất lượng tại điểm đo: chỉ khi dữ liệu cảm biến được triệt tiêu hoàn toàn các sai số quang học do sương ẩm, số liệu gửi về trung tâm mới đạt độ tin cậy chuẩn để làm đầu vào cho Mô hình 2 dự báo xu hướng tương lai.")

    add_image_box(os.path.join(FIGURES_DIR, "fig1_two_tier_ai_architecture.png"),
                  "Hình 1: Sơ đồ kiến trúc phân tầng 2 Mô hình AI trong hệ thống quan trắc (Tầng biên & Tầng máy chủ)",
                  width=Inches(6.2))

    add_h1("2. MÔ HÌNH 1: HIỆU CHUẨN CẢM BIẾN BÙ NỒM ẨM TẠI BIÊN (TINYML)")

    add_h2("2.1. Bản chất bài toán cần giải quyết")
    add_p("Các cảm biến bụi quang học giá rẻ đo nồng độ hạt dựa trên tán xạ chùm laser. Trong điều kiện nồm ẩm đặc thù (độ ẩm không khí tăng cao và xuất hiện sương mù li ti), các hạt bụi hút ẩm trương nở kích thước hình học kết hợp với các giọt nước siêu nhỏ bay vào buồng đo, khiến cảm biến lầm tưởng mật độ bụi tăng đột biến và xuất số đo ảo gấp nhiều lần thực tế.")
    add_p("Mô hình 1 được thiết kế để nạp trực tiếp vào firmware vi điều khiển tại trạm, tự động nhận diện tương quan nhiệt - ẩm để hiệu chỉnh số đo về sát với chuẩn thực tế ngay tại chân cắm cảm biến.")

    add_h2("2.2. Hướng tiếp cận và cơ chế bảo vệ vật lý độc quyền")
    add_bullet("Học từ tập dữ liệu chuẩn đối chứng quy mô lớn",
               "Mô hình được huấn luyện đối sánh từ tập dữ liệu thực nghiệm hàng trăm nghìn mẫu thu thập đồng bộ giữa cảm biến quang học và thiết bị đo chuẩn kiểm định quốc tế (nguyên lý suy giảm tia Beta) qua đa dạng miền khí hậu nhiệt đới.")
    add_bullet("Không gian đặc trưng vi khí hậu đa chiều",
               "Mô hình kết hợp đồng thời tín hiệu bụi đa buồng đo cùng các tham số nhiệt độ, độ ẩm tương đối và điểm bão hòa hơi nước tức thời để phản ánh đúng trạng thái ngưng tụ của không khí.")
    add_bullet("Cơ chế xuất hệ số điều chỉnh có chặn (Physics-Guided Bounded Output)",
               "Khác với các cách làm truyền thống cố gắng đoán trực tiếp nồng độ bụi (rất dễ sinh số âm hoặc số đo biến dị ngoài kiểm soát), giải pháp của đề tài sử dụng cơ chế nội suy hệ số điều chỉnh tự động giới hạn nghiêm ngặt trong một miền giá trị an toàn được dẫn dắt bởi vật lý khí dung. Nồng độ bụi sau hiệu chuẩn luôn là tích của số đo thô với hệ số điều chỉnh, loại trừ 100% rủi ro sinh số ảo.")
    add_bullet("Cơ chế chuyển tiếp vi khí hậu thích ứng (Adaptive Weather Gating)",
               "Mô hình tích hợp cổng kích hoạt phi tuyến tự động nhận diện trạng thái thời tiết: khi môi trường khô ráo, hệ thống tự động khóa cổng để nhường hoàn toàn quyền kiểm soát cho hiệu chuẩn gốc của nhà sản xuất; chỉ khi độ ẩm chạm ngưỡng gây sai số, thuật toán mới mở dần quyền can thiệp.")
    add_bullet("Kiểm định chéo cặp cảm biến kép (Dual-Sensor Validation)",
               "Thuật toán tại biên liên tục đối sánh tín hiệu giữa hai buồng đo song song, tự động phát hiện tình trạng đọng nước hoặc kẹt bụi cục bộ ở một bên để kích hoạt chế độ cách ly và cảnh báo bảo trì.")

    add_h2("2.3. Định hướng kiến trúc mạng và tối ưu hóa nhúng")
    add_bullet("Kiến trúc mạng nơ-ron học sâu siêu nhẹ",
               "Sử dụng cấu trúc mạng truyền thẳng nhiều tầng kết hợp các liên kết tắt phần dư (Residual Skip Connections), giúp trích xuất sâu các phi tuyến nhiệt ẩm phức tạp mà vẫn giữ kích thước mô hình siêu nhỏ gọn, tối ưu tuyệt đối cho bộ nhớ đệm của vi điều khiển.")
    add_bullet("Tối ưu hóa lượng tử hóa số nguyên (Integer Quantization)",
               "Mô hình sau huấn luyện được lượng tử hóa hoàn toàn sang dạng số nguyên 8-bit, thu nhỏ dung lượng chiếm dụng Flash xuống mức tối thiểu (chỉ vài chục Kilobyte) và bảo toàn độ chính xác tương đương mô hình số thực dấu phẩy động.")
    add_bullet("Kỷ luật thực thi Zero-Heap an toàn nhúng",
               "Toàn bộ thuật toán suy luận được đóng gói bằng C++ độc lập, không sử dụng cấp phát bộ nhớ động trong suốt thời gian chạy (0 Byte Heap), ngăn chặn triệt để nguy cơ tràn RAM hay treo vi điều khiển ngoài thực địa.")

    add_image_box(os.path.join(FIGURES_DIR, "fig2_model1_edge_pipeline.png"),
                  "Hình 2: Sơ đồ nguyên lý quy trình hiệu chuẩn tín hiệu bù nồm ẩm tại biên (Mô hình 1)",
                  width=Inches(6.2))

    add_h1("3. MÔ HÌNH 2: DỰ BÁO XU HƯỚNG Ô NHIỄM CHUỖI THỜI GIAN")

    add_h2("3.1. Bản chất bài toán cần giải quyết")
    add_p("Chất lượng không khí ngoài trời biến động phức tạp theo chu kỳ sinh hoạt đô thị (giờ cao điểm giao thông, sản xuất) kết hợp các hiện tượng khí tượng nghịch nhiệt, hướng gió và bức xạ mặt trời. Mô hình 2 được xây dựng nhằm mục đích khai phá chuỗi dữ liệu lịch sử để dự báo trước diễn biến nồng độ ô nhiễm trong tương lai gần (từ vài giờ đến một ngày tới).")

    add_h2("3.2. Không gian dữ liệu và phạm vi dự báo")
    add_bullet("Cửa sổ trượt chuỗi thời gian lịch sử",
               "Mô hình quan sát một chuỗi dữ liệu quá khứ liên tục (chu kỳ nhiều giờ trước đó), tích hợp đa biến gồm nồng độ bụi đã qua làm sạch, nhiệt độ, độ ẩm và các đặc trưng thời gian có tính tuần hoàn sinh học.")
    add_bullet("Đầu ra dự báo đa bước (Multi-Horizon Forecasting)",
               "Cung cấp bức tranh dự báo nồng độ bụi và chỉ số chất lượng không khí tại các mốc tương lai then chốt (như 3h, 6h, 12h và 24h tiếp theo), đồng thời cung cấp dải phân vị tin cậy để phục vụ công tác cảnh báo rủi ro sớm cho cộng đồng.")

    add_h2("3.3. Định hướng kiến trúc mạng nơ-ron chuỗi thời gian")
    add_bullet("Kiến trúc lai ghép đa chiều (Hybrid Temporal Architecture)",
               "Kết hợp giữa khối tích chập thời gian giãn cách (Temporal Convolutional Network) để bắt nhanh các xung đột biến ngắn hạn (như đột biến khí thải giờ tan tầm hoặc đợt không khí lạnh tràn về) và khối hồi quy tuần hoàn có cổng (Gated Recurrent Network) để duy trì trí nhớ xu hướng dài hạn qua nhiều ngày.")
    add_bullet("Cơ chế phân bổ chú ý theo ngữ cảnh (Attention Mechanism)",
               "Tự động tính toán trọng số quan trọng giữa các thời điểm trong quá khứ, giúp mô hình nhận biết được những tác động tích tụ qua đêm dẫn đến sương mù quang hóa vào buổi sáng.")
    add_bullet("Khối phát hiện dị thường bổ trợ (Anomaly Detection Engine)",
               "Mô hình hóa trạng thái bình thường của môi trường bằng mạng học không giám sát; khi phát hiện các biến động vượt ngưỡng hình học (như sự cố đốt rác nông nghiệp hoặc hỏng hóc cảm biến), hệ thống lập tức xuất cờ báo động khẩn cấp.")

    add_h2("3.4. Định hướng triển khai trên máy chủ biên")
    add_bullet("Môi trường thực thi tối ưu hóa phần cứng",
               "Mô hình được tối ưu hóa theo định dạng tiêu chuẩn công nghiệp (ONNX Runtime) để tận dụng các tập lệnh tăng tốc tính toán vector trên vi xử lý đa nhân của máy chủ biên Raspberry Pi 5.")
    add_bullet("Cơ chế vận hành tự động theo chu kỳ",
               "Được điều phối tự động định kỳ mỗi khi có khối dữ liệu đo mới được cập nhật từ trạm, chiếm dụng tài nguyên tính toán và bộ nhớ cực thấp, không làm ảnh hưởng đến các tác vụ cơ sở dữ liệu và truyền thông của máy chủ.")

    add_image_box(os.path.join(FIGURES_DIR, "fig3_model2_forecasting_pipeline.png"),
                  "Hình 3: Sơ đồ nguyên lý quy trình dự báo xu hướng ô nhiễm chuỗi thời gian (Mô hình 2)",
                  width=Inches(6.2))

    add_h1("4. BẢNG ĐỐI THOẠI KỸ THUẬT GIỮA 2 MÔ HÌNH")
    add_p("Bảng dưới đây tóm tắt định hướng xây dựng và phân vai kỹ thuật giữa hai mô hình trí tuệ nhân tạo trong hệ thống:")

    summary_rows = [
        ("Vai trò trọng tâm", "Hiệu chuẩn và khử nhiễu sai số nồm ẩm tại nguồn", "Khai phá quy luật và dự báo xu hướng tương lai"),
        ("Không gian thực thi", "Vi điều khiển nhúng tại trạm đo (ESP32-S3)", "Máy chủ biên cục bộ (Raspberry Pi 5)"),
        ("Dữ liệu đầu vào", "Vector tham số vi khí hậu tức thời tại chu kỳ đo", "Chuỗi trượt thời gian lịch sử đa biến nhiều ngày"),
        ("Dạng dữ liệu đầu ra", "Hệ số điều chỉnh có chặn an toàn vật lý", "Chuỗi giá trị nồng độ bụi dự báo đa bước tương lai"),
        ("Triết lý kiến trúc", "Mạng phần dư siêu nhẹ tối ưu hóa cho vi xử lý nhúng", "Mạng chuỗi thời gian lai ghép TCN kết hợp Attention"),
        ("Kỷ luật bộ nhớ", "Zero-Heap (0 Byte Heap), Stack tĩnh < 1 KB", "Đóng gói tối ưu ONNX Runtime tận dụng tăng tốc SIMD"),
        ("Độ trễ tính toán", "Cực thấp (dưới 1 phần nghìn giây)", "Dưới 1 giây trên máy chủ biên"),
        ("Mức độ độc lập", "Hoạt động độc lập 100% ngoại tuyến (Zero Network)", "Vận hành nội bộ trên máy chủ trạm quan trắc")
    ]
    create_table(["Tiêu chí kỹ thuật", "Mô hình 1: Hiệu chuẩn tại biên", "Mô hình 2: Dự báo chuỗi thời gian"], summary_rows,
                 col_widths=[Inches(1.8), Inches(2.3), Inches(2.3)])

    path_report = os.path.abspath(os.path.join(PROJECT_ROOT, "docs", "reports", "Du_Kien_Xay_Dung_2_Mo_Hinh_AI.docx"))
    path_tailieu = os.path.abspath(os.path.join(PROJECT_ROOT, "tailieu", "Du_Kien_Xay_Dung_2_Mo_Hinh_AI.docx"))

    os.makedirs(os.path.dirname(path_report), exist_ok=True)
    doc.save(path_report)
    print("SUCCESS: Saved to official directory:", path_report)

    if os.path.exists(os.path.dirname(path_tailieu)):
        shutil.copy2(path_report, path_tailieu)
        print("Synchronized to tailieu:", path_tailieu)

if __name__ == "__main__":
    generate_doc()
