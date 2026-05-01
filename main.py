# main.py
import cv2
import numpy as np
import os
import math
from pdf2image import convert_from_path
import fitz

# --- CONFIG  ---
POPPLER_PATH = r"D:\poppler-25.12.0\Library\bin" 
DPI_SETTING = 600 
TARGET_DPI = 600
PIXEL_TO_MM = 25.4 / TARGET_DPI
MARGIN = 20        
Z_SAFE = 15.0; Z_CUT_TRACE = -0.1; Z_CUT_DRILL = -2.2; Z_CUT_OUT = -2.2; PASS_DEPTH = 0.6    
FEED_RATE = 300; FEED_RATE_Z = 150; SPINDLE_SPEED = 12000 
TOOL_DRILL_ID = 1; TOOL_VBIT_ID = 2; TOOL_CUTOUT_ID = 3  


def min_radius_from_center(cnt, cx, cy):
    pts = cnt.reshape(-1, 2)
    dists = np.sqrt((pts[:,0] - cx)**2 + (pts[:,1] - cy)**2)
    return np.min(dists)

# --- UTILS  ---
def read_image_unicode(path, flags=cv2.IMREAD_COLOR):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(stream, flags)
    except: return None

def write_image_unicode(path, img):
    try:
        ext = os.path.splitext(path)[1]
        result, n = cv2.imencode(ext, img)
        if result:
            with open(path, mode='wb') as f: n.tofile(f)
            return True
        return False
    except: return False

def read_image_safe(path):
    """
    Hàm đọc ảnh hỗ trợ đường dẫn Tiếng Việt/Unicode trên Windows
    Thay thế cho cv2.imread thông thường
    """
    try:
        # Đọc file dưới dạng nhị phân rồi decode, tránh lỗi đường dẫn
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        # Đọc ảnh màu (BGR) để tương thích với các bước xử lý sau
        return cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Lỗi đọc file {path}: {e}")
        return None

def get_image_from_path(path):
    """
    Đọc file input (PDF hoặc Ảnh). 
    Luôn trả về ảnh OpenCV (BGR) và cờ báo hiệu nguồn gốc (is_pdf).
    """
    try:
        # TRƯỜNG HỢP 1: File là PDF
        if path.lower().endswith('.pdf'):
            doc = fitz.open(path)
            page = doc.load_page(0)
            
            # Quan trọng: Render đúng 300 DPI để kích thước chuẩn xác
            pix = page.get_pixmap(dpi=TARGET_DPI) 
            
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            if pix.n == 3:   img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif pix.n == 4: img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:            img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            
            return img, True # True = Là file PDF gốc

        # TRƯỜNG HỢP 2: File là Ảnh
        else:
            stream = open(path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            return img, False # False = Không phải PDF (DPI có thể khác)

    except Exception as e:
        print(f"Lỗi đọc file {path}: {str(e)}")
        return None, False
     
def calculate_distance(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def sort_points_nearest_neighbor(points):
    if not points: return []
    path = [points.pop(0)]
    while points:
        current = path[-1]
        nearest = min(range(len(points)), key=lambda i: calculate_distance(current, points[i]))
        path.append(points.pop(nearest))
    return path

# --- CONVERT PIXEL TO MM ---
def pixel_to_mm_value(px):
    return px * (25.4 / DPI_SETTING)

def pixel_to_mm_coord(x_px, y_px, h_px, off_x, off_y):
    # Trừ đi MARGIN để tọa độ mép mạch khít đúng với vị trí đặt trên web
    x_mm = ((x_px - MARGIN) * PIXEL_TO_MM) + off_x
    y_mm = ((h_px - MARGIN - y_px) * PIXEL_TO_MM) + off_y
    return round(x_mm, 3), round(y_mm, 3)

# ==========================================
# GIAI ĐOẠN 1: PHÂN TÍCH & CẮT ẢNH 
# ==========================================
def phase_1_analyze_and_crop(trace_path, drill_path, outline_path, output_folder):
    """
    Logic: 
    1. Đọc ảnh.
    2. Tìm vùng mạch thực tế dựa trên Outline.
    3. Cắt (Crop) cả 3 ảnh theo vùng đó.
    4. Tính kích thước thật.
    """
    # 1. Đọc dữ liệu
    trace_img, is_pdf_trace = get_image_from_path(trace_path)
    drill_img, _ = get_image_from_path(drill_path)
    outline_img, _ = get_image_from_path(outline_path)

    if trace_img is None: return None, "Lỗi: Không đọc được file Trace."
    if drill_img is None: return None, "Lỗi: Không đọc được file Drill."
    if outline_img is None: return None, "Lỗi: Không đọc được file Outline."

    # 2. Xử lý Outline để tìm "Vùng Mạch Thực Tế"
    try:
        # Chuyển xám
        gray = cv2.cvtColor(outline_img, cv2.COLOR_BGR2GRAY)
        
        # Tự động phát hiện nền trắng hay đen để Threshold đúng
        # Nếu ảnh trung bình rất sáng (>200) -> Nền trắng, nét đen -> Cần đảo ngược (THRESH_BINARY_INV)
        if np.mean(gray) > 200:
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        else:
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Tìm các đường bao
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, "Không tìm thấy đường bao mạch (Outline) nào."
        
        # Lấy đường bao lớn nhất (chính là biên dạng mạch)
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        
        # Thêm lề an toàn (Padding) khoảng 2mm (~24px tại 300dpi) để không cắt sát quá
        padding = int(2 * (300/25.4)) 
        
        h_img, w_img = outline_img.shape[:2]
        
        # Tính toán tọa độ cắt an toàn (không văng ra ngoài ảnh)
        x_crop = max(0, x - padding)
        y_crop = max(0, y - padding)
        w_crop = min(w_img - x_crop, w + 2*padding)
        h_crop = min(h_img - y_crop, h + 2*padding)

        # 3. CẮT ẢNH (Quan trọng: Cắt cả 3 tấm y hệt nhau)
        trace_final = trace_img[y_crop : y_crop+h_crop, x_crop : x_crop+w_crop]
        drill_final = drill_img[y_crop : y_crop+h_crop, x_crop : x_crop+w_crop]
        outline_final = outline_img[y_crop : y_crop+h_crop, x_crop : x_crop+w_crop]

        # 4. Lưu đè file kết quả để hiển thị trên Web
        paths = {
            'trace': os.path.join(output_folder, 'FINAL_trace.png'),
            'drill': os.path.join(output_folder, 'FINAL_drill.png'),
            'outline': os.path.join(output_folder, 'FINAL_outline.png')
        }
        
        cv2.imwrite(paths['trace'], trace_final)
        cv2.imwrite(paths['drill'], drill_final)
        cv2.imwrite(paths['outline'], outline_final)

        # 5. TÍNH KÍCH THƯỚC THẬT (MM) DỰA TRÊN ẢNH ĐÃ CẮT
        # Nếu là PDF gốc -> Dùng công thức chuẩn 300 DPI
        # Nếu là PNG upload -> Tạm thời vẫn dùng hệ số này (hoặc có thể tùy chỉnh nếu cần chính xác tuyệt đối cho ảnh scan)
        
        real_w_mm = round( (w_crop - 2 * MARGIN) * PIXEL_TO_MM, 2)
        real_h_mm = round( (h_crop - 2 * MARGIN) * PIXEL_TO_MM, 2)

        return {
            "size": {"w_mm": real_w_mm, "h_mm": real_h_mm},
            "dims_px": (w_crop, h_crop),
            "paths": paths
        }, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Lỗi xử lý ảnh: {str(e)}"

# ==========================================
# GIAI ĐOẠN 2: TẠO G-CODE ĐA ĐIỂM (MULTI-OFFSET)
# ==========================================
def phase_2_generate_multi(final_paths, dims_px, offset_list, filename, output_folder):
    """
    offset_list: Danh sách các dictionary [{'id':0, 'x': 10, 'y':10}, {'id':1, 'x': 50, 'y':10}]
    """
    w_px, h_px = dims_px
    
    # 1. Chuẩn bị ảnh Masking 
    img_trace = read_image_unicode(final_paths["trace"], cv2.IMREAD_GRAYSCALE)
    img_drill = read_image_unicode(final_paths["drill"], cv2.IMREAD_GRAYSCALE)
    img_outline = read_image_unicode(final_paths["outline"], cv2.IMREAD_GRAYSCALE)

    # Masking
    _, bin_drill_holes = cv2.threshold(img_drill, 200, 255, cv2.THRESH_BINARY_INV)
    contours_holes, _ = cv2.findContours(bin_drill_holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_filled = img_trace.copy()
    for cnt in contours_holes:
        (cx, cy), _ = cv2.minEnclosingCircle(cnt)

        r = min_radius_from_center(cnt, cx, cy)

        # Có thể trừ thêm chút để an toàn (tránh chạm pad)
        r = r * 0.65  

        cv2.circle(img_filled, (int(cx), int(cy)), int(r), (0), -1)
    masked_trace = cv2.bitwise_not(img_filled) # Ảnh dùng để tạo trace

    # Lưu ảnh Optimized để user xem chơi
    write_image_unicode(os.path.join(output_folder, f"OPTIMIZED_{filename}.png"), img_filled)

    # 2. BẮT ĐẦU VIẾT G-CODE
    gcode = []
    gcode.append("%")
    gcode.append(f"(Generated by PCB Tool - Multi: {len(offset_list)} pcs)")
    gcode.append("G21 G90 G17 G40 G49 G80")

    def add_tool_change(tid, tname):
        gcode.append(f"\n(=== TOOL CHANGE: {tname} ===)")
        gcode.append("M05")
        gcode.append("G00 G53 Z0")
        gcode.append(f"T{tid} M06")
        gcode.append(f"S{SPINDLE_SPEED} M03")
        gcode.append("G04 P3")
        gcode.append(f"G43 H{tid} Z{Z_SAFE}")

    # --- LOOP 1: DRILL (Khoan hết tất cả các mạch) ---
    add_tool_change(TOOL_DRILL_ID, "DRILL")
    _, bin_drill = cv2.threshold(img_drill, 200, 255, cv2.THRESH_BINARY_INV)
    cnt_drill, _ = cv2.findContours(bin_drill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Duyệt qua từng mạch (Offset)
    for item in offset_list:
        off_x, off_y = float(item['x']), float(item['y'])
        gcode.append(f"(--- DRILL: PCB #{item['id']} at X{off_x} Y{off_y} ---)")
        
        points_mm = []
        for cnt in cnt_drill:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx, cy = bx + bw/2, by + bh/2
            mmx, mmy = pixel_to_mm_coord(cx, cy, h_px, off_x, off_y)
            points_mm.append((mmx, mmy))
        
        sorted_pts = sort_points_nearest_neighbor(points_mm)
        if sorted_pts:
            gcode.append(f"G00 X{sorted_pts[0][0]} Y{sorted_pts[0][1]}")
            gcode.append(f"G81 Z{Z_CUT_DRILL} R2.0 F{FEED_RATE_Z}")
            for p in sorted_pts[1:]:
                gcode.append(f"X{p[0]} Y{p[1]}")
            gcode.append("G80")

    # --- LOOP 2: TRACE (Phay mạch hết tất cả) ---
    add_tool_change(TOOL_VBIT_ID, "TRACE")
    cnt_trace, _ = cv2.findContours(masked_trace, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for item in offset_list:
        off_x, off_y = float(item['x']), float(item['y'])
        gcode.append(f"(--- TRACE: PCB #{item['id']} ---)")
        
        for cnt in cnt_trace:
            if cv2.contourArea(cnt) < 5: continue
            start = cnt[0][0]
            sx, sy = pixel_to_mm_coord(start[0], start[1], h_px, off_x, off_y)
            
            gcode.append(f"G00 X{sx} Y{sy}")
            gcode.append(f"G01 Z{Z_CUT_TRACE} F{FEED_RATE_Z}")
            for pt in cnt[1:]:
                px, py = pixel_to_mm_coord(pt[0][0], pt[0][1], h_px, off_x, off_y)
                gcode.append(f"G01 X{px} Y{py} F{FEED_RATE}")
            gcode.append(f"G01 X{sx} Y{sy}") # Close
            gcode.append(f"G00 Z{Z_SAFE}")

    # --- LOOP 3: OUTLINE (Cắt viền hết tất cả) ---
    add_tool_change(TOOL_CUTOUT_ID, "OUTLINE")
    _, bin_out = cv2.threshold(img_outline, 200, 255, cv2.THRESH_BINARY_INV)
    cnt_out, _ = cv2.findContours(bin_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if cnt_out:
        max_out = max(cnt_out, key=cv2.contourArea)
        for item in offset_list:
            off_x, off_y = float(item['x']), float(item['y'])
            gcode.append(f"(--- CUTOUT: PCB #{item['id']} ---)")
            
            start = max_out[0][0]
            sx, sy = pixel_to_mm_coord(start[0], start[1], h_px, off_x, off_y)
            
            gcode.append(f"G00 X{sx} Y{sy}")
            gcode.append(f"G00 Z{Z_SAFE}")
            
            curr_z = 0.0
            while curr_z > Z_CUT_OUT:
                curr_z -= PASS_DEPTH
                if curr_z < Z_CUT_OUT: curr_z = Z_CUT_OUT
                gcode.append(f"G01 Z{curr_z} F{FEED_RATE_Z}")
                for pt in max_out[1:]:
                    px, py = pixel_to_mm_coord(pt[0][0], pt[0][1], h_px, off_x, off_y)
                    gcode.append(f"G01 X{px} Y{py} F{FEED_RATE}")
                gcode.append(f"G01 X{sx} Y{sy}") # Close loop
            
            gcode.append(f"G00 Z{Z_SAFE}")

    # Footer
    gcode.append("M05")
    gcode.append("G00 G53 Z0")
    gcode.append("M30")
    gcode.append("%")

    # Save
    if not filename.endswith('.nc'): filename += '.nc'
    final_path = os.path.join(output_folder, filename)
    with open(final_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(gcode))
    
    return final_path, None