
import os
import time
import base64
import serial
import cv2
import numpy as np
import random


# Flask & Web
from flask import Flask, render_template, request, redirect, url_for, send_file, session, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Logic CNC
from main import phase_1_analyze_and_crop, phase_2_generate_multi

# Logic AI (PyTorch & Albumentations)
import torch
import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.ops import batched_nms 
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =====================================================================
# KHỞI TẠO ỨNG DỤNG FLASK VÀ CẤU HÌNH
# =====================================================================
app = Flask(__name__)
app.secret_key = 'pcb_tool_secret_super_secure'
CORS(app)  # Hỗ trợ API cho hệ thống AI

# Cấu hình thư mục CNC
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Cấu hình Serial CNC
SERIAL_PORT = 'COM3' 
BAUD_RATE = 115200

# =====================================================================
# KHỞI TẠO MÔ HÌNH AI (CHẠY 1 LẦN KHI START SERVER)
# =====================================================================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 7 
custom_anchor_generator = DefaultBoxGenerator(
    aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], 
    scales=[0.03, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80], 
    steps=[8, 16, 32, 64, 100, 300]
)

model = torchvision.models.detection.ssd300_vgg16(weights=None, weights_backbone=None)
model.anchor_generator = custom_anchor_generator

in_channels = [layer.in_channels for layer in model.head.classification_head.module_list]
num_anchors = model.anchor_generator.num_anchors_per_location()
model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)

try:
    model.load_state_dict(torch.load('best_ssd300_deeppcb.pth', map_location=device))
    print("Model loaded successfully! Ready for Inference.")
except Exception as e:
    print(f"Error loading model: {e}")

model.to(device)
model.eval()

transform = A.Compose([
    A.Resize(300, 300),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

class_names = {
    1: ('open', (0, 0, 255)),       
    2: ('short', (0, 255, 0)),      
    3: ('mousebite', (255, 0, 0)),  
    4: ('spur', (0, 255, 255)),     
    5: ('copper', (255, 0, 255)),   
    6: ('pinhole', (255, 255, 0))   
}

# =====================================================================
# HÀM BỔ TRỢ: TIỀN XỬ LÝ ẢNH CHO AI
# =====================================================================
def preprocess_image(img):
    # 1. Đo lường "độ sặc sỡ" của bức ảnh để phân loại
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1] 
    mean_saturation = np.mean(s_channel)
    
    if mean_saturation > 5:
        # ẢNH MÀU
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        clahe_img = clahe.apply(blur)
        
        _, binary_img = cv2.threshold(
            clahe_img, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        kernel = np.ones((3,3), np.uint8)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
        min_area = 50000 
        cleaned = np.zeros_like(binary_img)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 255
                
        binary_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        return binary_rgb
        
    else:
        # ẢNH TRẮNG ĐEN
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        binary_rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
        return binary_rgb




def generate_realistic_defects(input_img_path, output_img_path, num_defects=10):
    """
    Hàm đọc ảnh mạch sạch, tạo lỗi ngẫu nhiên và lưu thành ảnh mới.
    """
    try:
        clean_img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
        if clean_img is None:
            return False

        defect_img = clean_img.copy()
        h, w = defect_img.shape

        # Nhận diện màu
        bg_pixel = defect_img[0, 0]
        bg_color = 255 if bg_pixel > 128 else 0
        trace_color = 0 if bg_color == 255 else 255

        trace_mask = (defect_img == trace_color).astype(np.uint8) * 255
        num_labels, labels = cv2.connectedComponents(trace_mask, connectivity=8)

        dist_transform = cv2.distanceTransform(trace_mask, cv2.DIST_L2, 3)
        core_coords = np.column_stack(np.where(dist_transform > 3))
        bg_coords = np.column_stack(np.where(defect_img == bg_color))
        trace_coords = np.column_stack(np.where(defect_img == trace_color))

        for i in range(num_defects):
            defect_type = random.choice(["OPEN", "SHORT", "COPPER"])

            if defect_type == "OPEN" and len(core_coords) > 0:
                pt = random.choice(core_coords)
                y, x = pt[0], pt[1]
                local_thickness = dist_transform[y, x] * 2
                window_size = 15
                y_min, y_max = max(0, y - window_size), min(h, y + window_size)
                x_min, x_max = max(0, x - window_size), min(w, x + window_size)
                local_patch = trace_mask[y_min:y_max, x_min:x_max]
                local_pts = np.column_stack(np.where(local_patch > 0))

                if len(local_pts) > 5:
                    local_pts_xy = np.array([(p[1], p[0]) for p in local_pts], dtype=np.float32)
                    [vx, vy, x0, y0] = cv2.fitLine(local_pts_xy, cv2.DIST_L2, 0, 0.01, 0.01)
                    perp_vx, perp_vy = -vy[0], vx[0]
                    cut_length = local_thickness + random.randint(6, 15)
                    thickness = random.randint(2, 4)
                    dx, dy = perp_vx * (cut_length / 2), perp_vy * (cut_length / 2)
                    x1, y1 = int(x - dx), int(y - dy)
                    x2, y2 = int(x + dx), int(y + dy)
                    cv2.line(defect_img, (x1, y1), (x2, y2), bg_color, thickness)

            elif defect_type == "COPPER" and len(bg_coords) > 0:
                pt = random.choice(bg_coords)
                y, x = pt[0], pt[1]
                radius = random.randint(2, 3)
                cv2.circle(defect_img, (x, y), radius, trace_color, -1)

            elif defect_type == "SHORT":
                for _ in range(50):
                    pt1 = random.choice(trace_coords)
                    y1, x1 = pt1[0], pt1[1]
                    label1 = labels[y1, x1]
                    search_radius = 25 
                    y_min, y_max = max(0, y1 - search_radius), min(h, y1 + search_radius)
                    x_min, x_max = max(0, x1 - search_radius), min(w, x1 + search_radius)
                    local_window = labels[y_min:y_max, x_min:x_max]
                    mask_other_traces = (local_window != label1) & (local_window > 0)
                    
                    if np.any(mask_other_traces):
                        other_trace_pts_local = np.column_stack(np.where(mask_other_traces))
                        other_trace_pts_global = other_trace_pts_local + [y_min, x_min]
                        distances = np.sum((other_trace_pts_global - [y1, x1])**2, axis=1)
                        closest_pt = other_trace_pts_global[np.argmin(distances)]
                        y2, x2 = closest_pt[0], closest_pt[1]
                        thickness = random.randint(2, 4)
                        cv2.line(defect_img, (x1, y1), (x2, y2), trace_color, thickness)
                        break 

        # Lưu ảnh đã có lỗi
        cv2.imwrite(output_img_path, defect_img)
        return True
    except Exception as e:
        print(f"Lỗi tạo defect: {e}")
        return False
# =====================================================================
# HỆ THỐNG 1: CÁC ROUTE CƠ BẢN VÀ XỬ LÝ G-CODE (CNC)
# =====================================================================

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'drill' not in request.files or 'trace' not in request.files or 'outline' not in request.files:
        return "Thiếu file! Vui lòng chọn đủ 3 file.", 400
    
    files_map = {}
    for key in ['drill', 'trace', 'outline']:
        file = request.files[key]
        if file.filename == '': return "Chưa chọn file", 400
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, f"{key}_{filename}")
        file.save(save_path)
        files_map[key] = save_path
    
    try:
        result, error = phase_1_analyze_and_crop(
            files_map['trace'], files_map['drill'], files_map['outline'], OUTPUT_FOLDER
        )
        if error: return f"Lỗi xử lý ảnh: {error}", 500
        session['analysis'] = result
        session['quantity'] = 1 
        return redirect(url_for('options_page'))
    except Exception as e:
        return f"Lỗi hệ thống Phase 1: {str(e)}", 500

@app.route('/options')
def options_page():
    data = session.get('analysis')
    if not data: return redirect(url_for('index'))
    return render_template('options.html', w_mm=data['size']['w_mm'], h_mm=data['size']['h_mm'])

@app.route('/set_options', methods=['POST'])
def set_options():
    try:
        qty = int(request.form.get('quantity', 1))
        session['quantity'] = max(1, qty)
        return redirect(url_for('position_page'))
    except:
        return redirect(url_for('options_page'))

@app.route('/position')
def position_page():
    data = session.get('analysis')
    qty = session.get('quantity', 1)
    if not data: return redirect(url_for('index'))
    return render_template('position.html', qty=qty, w_mm=data['size']['w_mm'], h_mm=data['size']['h_mm'])

@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/generate', methods=['POST'])
def generate():
    req = request.json
    offset_list = req.get('offsets')
    
    raw_filename = req.get('filename', 'output')
    if not raw_filename.endswith('.nc'):
        raw_filename += '.nc'
    
    filename = secure_filename(raw_filename)

    data = session.get('analysis')
    if not data or not offset_list: 
        return jsonify({"status":"error", "error":"Dữ liệu phiên làm việc không hợp lệ"}), 400

    try:
        final_path, error = phase_2_generate_multi(
            data['paths'], data['dims_px'], offset_list, filename, OUTPUT_FOLDER
        )
        if error: return jsonify({"status":"error", "error":error}), 500
        
        session['gcode_path'] = final_path
        session['gcode_filename'] = filename 
        
        return jsonify({"status": "ok", "filename": filename})
    except Exception as e:
        return jsonify({"status":"error", "error":str(e)}), 500

@app.route('/download')
def download():
    filename = request.args.get('filename')
    if not filename:
        filename = session.get('gcode_filename', 'result.nc')
    
    filename = secure_filename(filename)
    path = os.path.join(OUTPUT_FOLDER, filename)
    
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name=filename)
    return "File không tồn tại", 404

@app.route('/upload_serial', methods=['POST'])
def upload_serial():
    path = session.get('gcode_path')
    if not path or not os.path.exists(path):
        return jsonify({"status": "error", "error": "Không tìm thấy file G-code"}), 404
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            time.sleep(2)
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        ser.write((line.strip() + '\n').encode('utf-8'))
                        time.sleep(0.05) 
        return jsonify({"status": "ok", "message": "Đã gửi G-code xong!"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/preview')
def preview_page():
    return render_template('preview.html')

@app.route('/api/get_last_gcode')
def get_last_gcode():
    try:
        filename = request.args.get('filename', 'output.nc')
        filename = secure_filename(filename)
        
        file_path = os.path.join(OUTPUT_FOLDER, filename) 
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({'status': 'ok', 'content': content})
        else:
            file_path_bk = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(file_path_bk):
                with open(file_path_bk, 'r') as f:
                      content = f.read()
                return jsonify({'status': 'ok', 'content': content})
            
            return jsonify({'status': 'error', 'message': f'File {filename} not found'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})



# =====================================================================
# HỆ THỐNG 2: CÁC ROUTE PHỤC VỤ ĐÁNH GIÁ LỖI MẠCH (AI)
# =====================================================================

# --- ROUTE CHO TRANG NHẬN DIỆN LỖI AI ---
@app.route('/assessment')
def assessment_page():
    return render_template('assessment.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'image_name' not in data:
        return jsonify({"error": "No image_name"}), 400

    filename = secure_filename(data['image_name'])
    image_path = os.path.join(OUTPUT_FOLDER, filename)

    if not os.path.exists(image_path):
        return jsonify({"error": f"File not found: {filename}"}), 404

    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Cannot read image"}), 500
    
    # original_draw là ảnh GỐC CÓ MÀU dùng để vẽ khung hiển thị cho đẹp
    original_draw = image.copy() 
    h_orig, w_orig = image.shape[:2]
    
    # GỌI HÀM TIỀN XỬ LÝ (Để đưa vào mạng AI tính toán)
    image_rgb = preprocess_image(image)

    images_to_infer = []
    # Lưu lại thông tin (width_ratio, height_ratio, offset_x, offset_y) cho quá trình dịch tọa độ
    patch_infos = [] 

    # ---------------------------------------------------------
    # 1A. THÊM ẢNH GỐC VÀO (GLOBAL CONTEXT)
    # ---------------------------------------------------------
    full_tensor = transform(image=image_rgb)['image']
    images_to_infer.append(full_tensor)
    # Với ảnh gốc, tỷ lệ quy đổi là kích thước ảnh thật, offset = 0
    patch_infos.append((w_orig, h_orig, 0, 0))

    # ---------------------------------------------------------
    # 1B. THÊM 9 ẢNH CẮT VÀO (LOCAL DETAILS)
    # ---------------------------------------------------------
    patch_h = int(h_orig * 0.45) 
    patch_w = int(w_orig * 0.45)
    
    y_starts = [0, (h_orig - patch_h) // 2, h_orig - patch_h]
    x_starts = [0, (w_orig - patch_w) // 2, w_orig - patch_w]

    for y in y_starts:
        for x in x_starts:
            patch = image_rgb[y:y+patch_h, x:x+patch_w]
            tensor = transform(image=patch)['image']
            images_to_infer.append(tensor)
            # Với ảnh cắt, tỷ lệ quy đổi là patch_w, patch_h, kèm offset x, y
            patch_infos.append((patch_w, patch_h, x, y))

    # ---------------------------------------------------------
    # 2. XỬ LÝ LÔ (BATCH INFERENCE) - Quét 10 ảnh cùng lúc
    # ---------------------------------------------------------
    batch_tensor = torch.stack(images_to_infer).to(device)
    with torch.no_grad():
        predictions = model(batch_tensor) 

    # ---------------------------------------------------------
    # 3. GOM KẾT QUẢ VÀ TRẢ TỌA ĐỘ VỀ ẢNH GỐC
    # ---------------------------------------------------------
    all_boxes = []
    all_scores = []
    all_labels = []

    for i, pred in enumerate(predictions):
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        # Lấy tỷ lệ và offset tương ứng với ảnh gốc hoặc ảnh cắt
        p_w, p_h, x_off, y_off = patch_infos[i]

        for box, label, score in zip(boxes, labels, scores):
            if score >= 0.5: 
                x1, y1, x2, y2 = box
                
                # Quy đổi kích thước từ 300x300 về kích thước tương ứng (ảnh gốc hoặc mảnh cắt)
                x1 = x1 * p_w / 300
                y1 = y1 * p_h / 300
                x2 = x2 * p_w / 300
                y2 = y2 * p_h / 300
                
                # Cộng thêm offset
                x1 += x_off
                y1 += y_off
                x2 += x_off
                y2 += y_off
                
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(float(score))
                all_labels.append(int(label))

    detections = []
    
    # ---------------------------------------------------------
    # 4. LỌC CÁC KHUNG BỊ TRÙNG LẶP GIAO NHAU (NMS)
    # ---------------------------------------------------------
    if len(all_boxes) > 0:
        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
        labels_tensor = torch.tensor(all_labels, dtype=torch.int64)
        
        # Lọc những box trùng nhau (đến từ việc giao thoa giữa ảnh gốc và ảnh cắt)
        keep_indices = batched_nms(boxes_tensor, scores_tensor, labels_tensor, iou_threshold=0.3)
        
        final_boxes = boxes_tensor[keep_indices].numpy()
        final_scores = scores_tensor[keep_indices].numpy()
        final_labels = labels_tensor[keep_indices].numpy()
        
        # 5. VẼ KẾT QUẢ CUỐI CÙNG LÊN ẢNH
        for box, score, label in zip(final_boxes, final_scores, final_labels):
            x1, y1, x2, y2 = map(int, box)
            name, color = class_names.get(int(label), ('unknown', (255, 255, 255)))
            
            # Khung được vẽ lên original_draw (ảnh màu gốc chưa qua tiền xử lý)
            cv2.rectangle(original_draw, (x1, y1), (x2, y2), color, 2)
            text = f"{name}: {score:.1%}"
            cv2.putText(original_draw, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            detections.append({
                "class_id": int(label),
                "class_name": name,
                "score": float(score),
                "box": [x1, y1, x2, y2]
            })

# 6. TRẢ KẾT QUẢ VỀ GIAO DIỆN
    # Mã hóa ảnh gốc có vẽ khung
    _, buffer = cv2.imencode('.jpg', original_draw)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # MỚI: Mã hóa thêm ảnh trắng đen (image_rgb đã được xử lý ở trên)
    _, buffer_bw = cv2.imencode('.jpg', image_rgb)
    img_base64_bw = base64.b64encode(buffer_bw).decode('utf-8')

    return jsonify({
        "detections": detections,
        "result_image": f"data:image/jpeg;base64,{img_base64}",
        "preprocessed_image": f"data:image/jpeg;base64,{img_base64_bw}" # Gửi thêm ảnh này
    })

@app.route('/api/get_latest_optimized_image')
def get_latest_optimized_image():
    try:
        # 1. Lấy filename từ session
        filename = request.args.get('filename') or session.get('gcode_filename', 'output.nc')


        # 3. Tạo tên ảnh đúng
        clean_filename = f"OPTIMIZED_{filename}.png"
        defective_filename = f"DEFECTIVE_{filename}.png"

        clean_path = os.path.join(OUTPUT_FOLDER, clean_filename)
        defective_path = os.path.join(OUTPUT_FOLDER, defective_filename)

        # 4. Kiểm tra file tồn tại
        if not os.path.exists(clean_path):
            return jsonify({
                'status': 'error',
                'message': f'Không tìm thấy ảnh: {clean_filename}'
            })

        # 5. Tạo defect
        success = generate_realistic_defects(clean_path, defective_path, num_defects=10)

        if not success:
            return jsonify({
                'status': 'error',
                'message': 'Tạo defect thất bại'
            })

        # 6. Trả về ảnh
        return jsonify({
            'status': 'ok',
            'defective': defective_filename,
            'optimized': clean_filename
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
# =====================================================================
# CHẠY ỨNG DỤNG
# =====================================================================
if __name__ == '__main__':
    # Thêm host='0.0.0.0' để cho phép các máy khác trong mạng local truy cập
    app.run(host='0.0.0.0', debug=True, port=5000)