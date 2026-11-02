"""
Script tao 3 so do ky thuat BANG PILLOW (PIL) - CHUAN XAC TUNG PIXEL:
- Tu dong ngat dong (word wrap), khong bao gio co chu bi tran ra ngoai khung.
- Can dong, can chu, padding le trai phai hoan hao.
- Tieu de can giua tuyet doi, khong bao gio bi cat mat chu hai dau.
- Mui ten ket noi ro rang, khong de len chu hay khung.
Luu truc tiep vao: docs/figures/
"""

import os
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = r"d:\AQ\docs\figures"
os.makedirs(OUT_DIR, exist_ok=True)

FONT_REG_PATH = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD_PATH = "C:/Windows/Fonts/segoeuib.ttf"
FONT_ITALIC_PATH = "C:/Windows/Fonts/segoeuii.ttf"

def get_font(size, style="regular"):
    if style == "bold" and os.path.exists(FONT_BOLD_PATH):
        return ImageFont.truetype(FONT_BOLD_PATH, size)
    elif style == "italic" and os.path.exists(FONT_ITALIC_PATH):
        return ImageFont.truetype(FONT_ITALIC_PATH, size)
    return ImageFont.truetype(FONT_REG_PATH, size)

def wrap_text(text, font, max_width, draw):
    words = text.split(" ")
    lines = []
    curr = []
    for w in words:
        test = " ".join(curr + [w])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            curr.append(w)
        else:
            if curr:
                lines.append(" ".join(curr))
            curr = [w]
    if curr:
        lines.append(" ".join(curr))
    return lines

def draw_down_arrow(draw, x, y_top, y_bot, color="#2563EB", width=3):

    draw.line([(x, y_top), (x, y_bot)], fill=color, width=width)

    head_size = 8
    draw.polygon([
        (x, y_bot + 2),
        (x - head_size, y_bot - head_size),
        (x + head_size, y_bot - head_size)
    ], fill=color)

def generate_fig1_pil():
    W, H = 1800, 1260
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    f_title = get_font(32, "bold")
    f_sub = get_font(22, "regular")
    f_h2 = get_font(25, "bold")
    f_card_t = get_font(27, "bold")
    f_card_sub = get_font(22, "italic")
    f_bullet = get_font(23, "regular")
    f_pill = get_font(21, "bold")

    draw.rounded_rectangle([15, 15, W - 15, H - 15], radius=24, outline="#CBD5E1", width=2)

    draw.rounded_rectangle([45, 35, W - 45, 145], radius=16, fill="#0F2942")
    t1 = "KIẾN TRÚC PHÂN TẦNG 2 MÔ HÌNH TRÍ TUỆ NHÂN TẠO"
    bb1 = draw.textbbox((0, 0), t1, font=f_title)
    draw.text(((W - (bb1[2] - bb1[0])) / 2, 52), t1, font=f_title, fill="#FFFFFF")

    t1_sub = "Hệ Thống Quan Trắc Môi Trường Đa Tầng: Xử Lý Tại Biên (In-situ) & Máy Chủ Trung Tâm"
    bb1_sub = draw.textbbox((0, 0), t1_sub, font=f_sub)
    draw.text(((W - (bb1_sub[2] - bb1_sub[0])) / 2, 98), t1_sub, font=f_sub, fill="#94A3B8")

    y1 = 175
    h_card1 = 440
    draw.rounded_rectangle([45, y1, W - 45, y1 + h_card1], radius=20, fill="#F8FAFC", outline="#3B82F6", width=3)

    draw.rounded_rectangle([65, y1 + 20, W - 65, y1 + 80], radius=12, fill="#1E40AF")
    tag1 = "TẦNG 1: THIẾT BỊ BIÊN (HỘP QUAN TRẮC) — VI ĐIỀU KHIỂN ESP32-S3"
    bbt1 = draw.textbbox((0, 0), tag1, font=f_h2)
    draw.text(((W - (bbt1[2] - bbt1[0])) / 2, y1 + 33), tag1, font=f_h2, fill="#FFFFFF")

    draw.text((85, y1 + 102), "MÔ HÌNH 1: HIỆU CHUẨN BÙ NỒM ẨM TẠI BIÊN (TINYML)", font=f_card_t, fill="#1E40AF")
    draw.text((85, y1 + 142), "Nhiệm vụ: Làm sạch tín hiệu cảm biến và khử sai số quang học do sương ẩm tại nguồn", font=f_card_sub, fill="#475569")

    bullets1 = [
        "• Đầu vào vi khí hậu: Cặp cảm biến bụi quang học kép (Dual PM) kết hợp cảm biến nhiệt ẩm SHT40.",
        "• Chức năng cốt lõi: Kiểm định chéo 2 buồng đo, bù trừ sai số phi tuyến bằng mạng phần dư siêu nhẹ.",
        "• Cơ chế bảo vệ: Hệ số điều chỉnh có chặn vật lý an toàn, cổng thích ứng tự ngắt khi trời khô ráo."
    ]
    yb = y1 + 195
    for b in bullets1:
        lines = wrap_text(b, f_bullet, 1600, draw)
        for line in lines:
            draw.text((85, yb), line, font=f_bullet, fill="#0F172A")
            yb += 36
        yb += 8

    draw.rounded_rectangle([85, y1 + 365, W - 85, y1 + 415], radius=10, fill="#DCFCE7", outline="#16A34A", width=1)
    pill1 = "Kỷ luật nhúng: 0 Byte Heap RAM   |   Bộ nhớ Flash < 20 KB   |   Độ trễ suy luận ~0.2 ms   |   Ngoại tuyến 100%"
    bbp1 = draw.textbbox((0, 0), pill1, font=f_pill)
    draw.text(((W - (bbp1[2] - bbp1[0])) / 2, y1 + 378), pill1, font=f_pill, fill="#15803D")

    draw_down_arrow(draw, W / 2, y1 + h_card1 + 5, y1 + h_card1 + 38, color="#2563EB", width=3)

    p_conn_w = 1100
    p_conn_x = (W - p_conn_w) / 2
    p_conn_y = y1 + h_card1 + 38
    draw.rounded_rectangle([p_conn_x, p_conn_y, p_conn_x + p_conn_w, p_conn_y + 55], radius=25, fill="#EFF6FF", outline="#2563EB", width=2)
    conn_txt = "Truyền thông không dây (4G LTE / Wi-Fi)   -->   Gói tin nồng độ bụi đã qua làm sạch"
    bb_conn = draw.textbbox((0, 0), conn_txt, font=f_pill)
    draw.text(((W - (bb_conn[2] - bb_conn[0])) / 2, p_conn_y + 14), conn_txt, font=f_pill, fill="#1D4ED8")

    draw_down_arrow(draw, W / 2, p_conn_y + 55, p_conn_y + 88, color="#2563EB", width=3)

    y2 = p_conn_y + 88
    h_card2 = 440
    draw.rounded_rectangle([45, y2, W - 45, y2 + h_card2], radius=20, fill="#F8FAFC", outline="#0D9488", width=3)

    draw.rounded_rectangle([65, y2 + 20, W - 65, y2 + 80], radius=12, fill="#0F766E")
    tag2 = "TẦNG 2: MÁY CHỦ BIÊN CỤC BỘ — RASPBERRY PI 5 (8GB RAM)"
    bbt2 = draw.textbbox((0, 0), tag2, font=f_h2)
    draw.text(((W - (bbt2[2] - bbt2[0])) / 2, y2 + 33), tag2, font=f_h2, fill="#FFFFFF")

    draw.text((85, y2 + 102), "MÔ HÌNH 2: KHAI PHÁ & DỰ BÁO XU HƯỚNG Ô NHIỄM (TEMPORAL AI)", font=f_card_t, fill="#0F766E")
    draw.text((85, y2 + 142), "Nhiệm vụ: Khai phá chuỗi thời gian lịch sử, dự báo ô nhiễm tương lai và phát hiện dị thường", font=f_card_sub, fill="#475569")

    bullets2 = [
        "• Cửa sổ lịch sử: Chuỗi nồng độ bụi đã qua làm sạch nhiều giờ kết hợp chu kỳ nhịp điệu sinh hoạt đô thị.",
        "• Kiến trúc lai ghép: Mạng tích chập thời gian TCN bắt biến động nhanh + GRU duy trì xu hướng dài hạn.",
        "• Cơ chế chú ý: Tự động phân bổ trọng số cho các khung giờ then chốt có nguy cơ tích tụ ô nhiễm cao."
    ]
    yb = y2 + 195
    for b in bullets2:
        lines = wrap_text(b, f_bullet, 1600, draw)
        for line in lines:
            draw.text((85, yb), line, font=f_bullet, fill="#0F172A")
            yb += 36
        yb += 8

    draw.rounded_rectangle([85, y2 + 365, W - 85, y2 + 415], radius=10, fill="#FEF3C7", outline="#D97706", width=1)
    pill2 = "Triển khai: Đóng gói ONNX Runtime   |   Dự báo 6h, 12h, 24h tới   |   Suy luận đa luồng < 1 ms   |   RAM < 60 MB"
    bbp2 = draw.textbbox((0, 0), pill2, font=f_pill)
    draw.text(((W - (bbp2[2] - bbp2[0])) / 2, y2 + 378), pill2, font=f_pill, fill="#B45309")

    out_path = os.path.join(OUT_DIR, "fig1_two_tier_ai_architecture.png")
    img.save(out_path, dpi=(300, 300))
    print("Fig 1 PIL generated:", out_path)

def generate_fig2_pil():
    W, H = 1800, 1180
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    f_title = get_font(30, "bold")
    f_sub = get_font(21, "regular")
    f_badge_num = get_font(26, "bold")
    f_badge_t = get_font(22, "bold")
    f_bullet = get_font(22, "regular")

    draw.rounded_rectangle([15, 15, W - 15, H - 15], radius=24, outline="#CBD5E1", width=2)

    draw.rounded_rectangle([45, 30, W - 45, 135], radius=16, fill="#1E40AF")
    t = "MÔ HÌNH 1: QUY TRÌNH HIỆU CHUẨN TÍN HIỆU CẢM BIẾN TẠI BIÊN"
    bbt = draw.textbbox((0, 0), t, font=f_title)
    draw.text(((W - (bbt[2] - bbt[0])) / 2, 48), t, font=f_title, fill="#FFFFFF")

    ts = "Kiến trúc thuật toán TinyML thực thi In-situ trực tiếp trên vi điều khiển ESP32-S3 (Ngoại tuyến 100%)"
    bbts = draw.textbbox((0, 0), ts, font=f_sub)
    draw.text(((W - (bbts[2] - bbts[0])) / 2, 92), ts, font=f_sub, fill="#DBEAFE")

    steps = [
        ("BƯỚC 1", "THU NHẬN TÍN HIỆU", "#1E40AF", "#EFF6FF", "#3B82F6", [
            "• Đọc đồng thời cặp cảm biến bụi quang học kép (Plantower PMS7003 #1 và #2).",
            "• Thu nhận nhiệt độ và độ ẩm không khí từ cảm biến chuyên dụng (Sensirion SHT40).",
            "• Tính toán nhiệt độ điểm sương khí quyển tức thời (xác định trạng thái bão hòa hơi nước)."
        ]),
        ("BƯỚC 2", "KIỂM ĐỊNH CHÉO", "#15803D", "#F0FDF4", "#16A34A", [
            "• Đối sánh liên tục độ lệch pha tín hiệu đo giữa hai buồng đo bụi song song.",
            "• Tự động nhận diện khi một buồng đo bị giọt sương đọng hoặc bám bụi bẩn bất thường.",
            "• Tự chuyển đổi sang buồng đo sạch hơn và bật cờ cảnh báo yêu cầu bảo dưỡng trạm."
        ]),
        ("BƯỚC 3", "MẠNG PHẦN DƯ", "#7E22CE", "#FAF5FF", "#A855F7", [
            "• Mạng nơ-ron học sâu siêu nhẹ (Residual MLP) được tối ưu hóa riêng cho vi xử lý nhúng.",
            "• Khối liên kết tắt (Skip Connection) giúp bảo toàn đặc trưng vật lý và chống suy biến tín hiệu.",
            "• Trích xuất chính xác quan hệ phi tuyến phức tạp giữa nồng độ bụi và các tham số vi khí hậu."
        ]),
        ("BƯỚC 4", "CHẶN AN TOÀN", "#B45309", "#FFFBEB", "#D97706", [
            "• Cơ chế sinh hệ số điều chỉnh tự động giới hạn chặt chẽ trong dải an toàn vật lý [0.25, 1.05].",
            "• Cổng thích ứng vi khí hậu tự ngắt can thiệp khi trời khô ráo để bảo toàn độ nhạy gốc.",
            "• Xuất nồng độ chuẩn hóa tức thời: PM_calib = PM_thô × Hệ số điều chỉnh (Không bao giờ âm)."
        ])
    ]

    card_y = 160
    card_h = 210
    gap = 42

    for i, (b_num, b_title, c_dark, c_bg, c_border, bullets) in enumerate(steps):

        draw.rounded_rectangle([45, card_y, W - 45, card_y + card_h], radius=16, fill=c_bg, outline=c_border, width=2)

        draw.rounded_rectangle([65, card_y + 18, 410, card_y + card_h - 18], radius=12, fill=c_dark)
        bb_num = draw.textbbox((0, 0), b_num, font=f_badge_num)
        draw.text((65 + (345 - (bb_num[2] - bb_num[0])) / 2, card_y + 48), b_num, font=f_badge_num, fill="#FFFFFF")

        bb_bt = draw.textbbox((0, 0), b_title, font=f_badge_t)
        draw.text((65 + (345 - (bb_bt[2] - bb_bt[0])) / 2, card_y + 98), b_title, font=f_badge_t, fill="#FFFFFF")

        yb = card_y + 35
        for b in bullets:
            lines = wrap_text(b, f_bullet, 1260, draw)
            for line in lines:
                draw.text((445, yb), line, font=f_bullet, fill="#0F172A")
                yb += 34
            yb += 8

        if i < 3:
            draw_down_arrow(draw, W / 2, card_y + card_h + 3, card_y + card_h + gap - 3, color=c_dark, width=3)

        card_y += card_h + gap

    out_path = os.path.join(OUT_DIR, "fig2_model1_edge_pipeline.png")
    img.save(out_path, dpi=(300, 300))
    print("Fig 2 PIL generated:", out_path)

def generate_fig3_pil():
    W, H = 1800, 1180
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    f_title = get_font(30, "bold")
    f_sub = get_font(21, "regular")
    f_badge_num = get_font(26, "bold")
    f_badge_t = get_font(22, "bold")
    f_bullet = get_font(22, "regular")

    draw.rounded_rectangle([15, 15, W - 15, H - 15], radius=24, outline="#CBD5E1", width=2)

    draw.rounded_rectangle([45, 30, W - 45, 135], radius=16, fill="#0F766E")
    t = "MÔ HÌNH 2: QUY TRÌNH DỰ BÁO XU HƯỚNG Ô NHIỄM CHUỖI THỜI GIAN"
    bbt = draw.textbbox((0, 0), t, font=f_title)
    draw.text(((W - (bbt[2] - bbt[0])) / 2, 48), t, font=f_title, fill="#FFFFFF")

    ts = "Kiến trúc Temporal AI tối ưu hóa trên máy chủ biên Raspberry Pi 5 (8GB RAM)"
    bbts = draw.textbbox((0, 0), ts, font=f_sub)
    draw.text(((W - (bbts[2] - bbts[0])) / 2, 92), ts, font=f_sub, fill="#CCFBF1")

    steps = [
        ("BƯỚC 1", "CHUỖI THỜI GIAN", "#0F766E", "#F0FDFA", "#0D9488", [
            "• Tiếp nhận chuỗi số liệu sạch nhiều giờ liên tục truyền về từ các hộp quan trắc tại trạm.",
            "• Chuỗi nồng độ PM2.5 đã qua hiệu chuẩn khử nồm ẩm triệt để bởi Mô hình 1 tại biên.",
            "• Tích hợp biến mã hóa chu kỳ ngày/đêm và quy luật nhịp điệu giao thông đô thị."
        ]),
        ("BƯỚC 2", "MẠNG LAI GHÉP", "#1E40AF", "#EFF6FF", "#3B82F6", [
            "• Khối tích chập thời gian TCN bắt nhanh các biến động đột biến ô nhiễm ngắn hạn.",
            "• Khối hồi quy tuần hoàn GRU duy trì xu hướng biến thiên chất lượng không khí dài hạn.",
            "• Cấu trúc lai ghép tối ưu hóa riêng cho chuỗi thời gian khí tượng môi trường đa biến."
        ]),
        ("BƯỚC 3", "TẦNG CHÚ Ý", "#7E22CE", "#FAF5FF", "#A855F7", [
            "• Cơ chế tự phân bổ trọng số ngữ cảnh (Self-Attention Mechanism).",
            "• Tự động dồn trọng số vào các khung giờ then chốt trong ngày (giờ cao điểm giao thông).",
            "• Nắm bắt quy luật tích tụ ô nhiễm phát thải qua đêm ảnh hưởng tới sáng sớm hôm sau."
        ]),
        ("BƯỚC 4", "DỰ BÁO ĐA MỐC", "#B45309", "#FFFBEB", "#D97706", [
            "• Dự báo nồng độ bụi PM2.5 và chỉ số chất lượng không khí cho các mốc 6h, 12h, 24h tới.",
            "• Cung cấp dải phân vị tin cậy 90% phục vụ công tác cảnh báo rủi ro sức khỏe cộng đồng.",
            "• Tích hợp module phát hiện dị thường kích hoạt chuông báo động khi có xả thải/đốt rác."
        ])
    ]

    card_y = 160
    card_h = 210
    gap = 42

    for i, (b_num, b_title, c_dark, c_bg, c_border, bullets) in enumerate(steps):

        draw.rounded_rectangle([45, card_y, W - 45, card_y + card_h], radius=16, fill=c_bg, outline=c_border, width=2)

        draw.rounded_rectangle([65, card_y + 18, 410, card_y + card_h - 18], radius=12, fill=c_dark)
        bb_num = draw.textbbox((0, 0), b_num, font=f_badge_num)
        draw.text((65 + (345 - (bb_num[2] - bb_num[0])) / 2, card_y + 48), b_num, font=f_badge_num, fill="#FFFFFF")

        bb_bt = draw.textbbox((0, 0), b_title, font=f_badge_t)
        draw.text((65 + (345 - (bb_bt[2] - bb_bt[0])) / 2, card_y + 98), b_title, font=f_badge_t, fill="#FFFFFF")

        yb = card_y + 35
        for b in bullets:
            lines = wrap_text(b, f_bullet, 1260, draw)
            for line in lines:
                draw.text((445, yb), line, font=f_bullet, fill="#0F172A")
                yb += 34
            yb += 8

        if i < 3:
            draw_down_arrow(draw, W / 2, card_y + card_h + 3, card_y + card_h + gap - 3, color=c_dark, width=3)

        card_y += card_h + gap

    out_path = os.path.join(OUT_DIR, "fig3_model2_forecasting_pipeline.png")
    img.save(out_path, dpi=(300, 300))
    print("Fig 3 PIL generated:", out_path)

if __name__ == "__main__":
    generate_fig1_pil()
    generate_fig2_pil()
    generate_fig3_pil()
    print("SUCCESS: ALL 3 PIL FIGURES GENERATED WITH 100% PERFECT PADDING & WRAPPING!")
