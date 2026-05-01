document.addEventListener("DOMContentLoaded", function() {

    // ============================================================
    // PHẦN 1: LOGIC CHO TRANG OPTIONS (BƯỚC 2 - TABS)
    // ============================================================
    const tabLinks = document.querySelectorAll('.nav-link');
    if (tabLinks.length > 0) {
        const triggerTabList = [].slice.call(tabLinks);
        triggerTabList.forEach(function(triggerEl) {
            triggerEl.addEventListener('click', function(event) {
                event.preventDefault();
                triggerTabList.forEach(btn => btn.classList.remove('active'));
                this.classList.add('active');
                document.querySelectorAll('.tab-pane').forEach(pane => {
                    pane.classList.remove('active');
                    pane.style.display = 'none';
                });
                const targetId = this.getAttribute('data-bs-target');
                const targetPane = document.querySelector(targetId);
                if (targetPane) {
                    targetPane.style.display = 'block';
                    setTimeout(() => targetPane.classList.add('active'), 10);
                }
            });
        });
    }

    // ============================================================
    // PHẦN 2: LOGIC CHO TRANG POSITION (BƯỚC 3)
    // ============================================================
    const bed = document.getElementById('bed-container');
    const NGROK_URL = "https://modish-jay-unviolently.ngrok-free.dev";
    if (bed) {
        // --- CHỐT CỨNG KÍCH THƯỚC BÀN MÁY ---
        const BOARD_REAL_W_MM = 200.0;
        const BOARD_REAL_H_MM = 150.0;

        const panel = document.getElementById('coords-panel');
        const btnCreate = document.getElementById('btn-create');
        const actionDiv = document.getElementById('action-buttons');
        const dataEl = document.getElementById('server-data');

        // Kích thước PCB
        //const PCB_QTY = parseInt(dataEl.dataset.qty) || 1;
        //const PCB_W_MM = parseFloat(dataEl.dataset.w) || 10;
        //const PCB_H_MM = parseFloat(dataEl.dataset.h) || 10;
        const PCB_QTY = parseInt(localStorage.getItem('pcb_qty') || 1);
        const PCB_W_MM = parseFloat(localStorage.getItem('pcb_w_mm') || 0);
        const PCB_H_MM = parseFloat(localStorage.getItem('pcb_h_mm') || 0);
        let pixelsPerMM = 1; 
        let pcbList = [];   

        // --- QUẢN LÝ MODAL ---
        const filenameModal = document.getElementById('filenameModal');
        const successModal = document.getElementById('successModal');
        const successMsg = document.getElementById('successMessage');

        window.handleCreateClick = function() {
            if (filenameModal) {
                filenameModal.classList.add('show');
                const input = document.getElementById('gcodeFilename');
                if(input) { input.focus(); input.select(); }
            }
        };

        window.closeModal = function() { if (filenameModal) filenameModal.classList.remove('show'); };
        window.closeSuccessModal = function() { if (successModal) successModal.classList.remove('show'); };

        window.onclick = function(event) {
            if (event.target === filenameModal) closeModal();
            if (event.target === successModal) closeSuccessModal();
        };

        const inputName = document.getElementById('gcodeFilename');
        if (inputName) {
            inputName.addEventListener("keypress", function(event) {
                if (event.key === "Enter") {
                    event.preventDefault();
                    submitGcodeGeneration();
                }
            });
        }

        window.submitGcodeGeneration = async function() {
            // Tự động tìm đúng ID dù giao diện có thay đổi
            const inputEl = document.getElementById('gcodeFilename') || document.getElementById('filenameInput');
            let filename = inputEl ? inputEl.value : "output.nc";
            
            if (!filename.trim()) filename = "output.nc";
            if (!filename.endsWith('.nc')) filename += ".nc";

            const analysisData = JSON.parse(localStorage.getItem('pcb_analysis') || '{}');
            
            const btnConfirm = document.querySelector('#filenameModal .btn-success');
            if(btnConfirm) {
                btnConfirm.innerHTML = '⏳ Đang xử lý...';
                btnConfirm.disabled = true;
            }

            try {
                // SỬA LỖI Ở ĐÂY: Đổi x_mm thành x, y_mm thành y để Python đọc được
                const formattedOffsets = pcbList.map(p => ({ 
                    id: p.id, 
                    x: p.x_mm, 
                    y: p.y_mm 
                }));

                const response = await fetch(`${NGROK_URL}/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        offsets: formattedOffsets, // Đã gửi đúng format
                        filename: filename,
                        analysis: analysisData
                    })
                });
                const data = await response.json();
                
                if (data.status === 'ok') {
                    localStorage.setItem('pcb_gcode_filename', data.filename);
                    
                    const fModal = document.getElementById('filenameModal');
                    if (fModal) fModal.classList.remove('show');
                    
                    const successMsg = document.getElementById('successMessage');
                    if (successMsg) successMsg.innerText = `Đã tạo xong file: ${data.filename}`;              
                    
                    const modalFooter = document.querySelector('#successModal .modal-footer');
                    if(modalFooter) {
                        modalFooter.innerHTML = `
                            <button type="button" class="btn btn-secondary" onclick="closeSuccessModal()">Đóng</button>
                            <a href="preview.html?filename=${data.filename}" class="btn btn-primary fw-bold">
                                👁️ Xem mô phỏng (3D)
                            </a>
                        `;
                    }
                    
                    const successModal = document.getElementById('successModal');
                    if (successModal) successModal.classList.add('show');
                    
                    const btnCreate = document.getElementById('btn-create');
                    if(btnCreate) btnCreate.style.display = 'none';
                    
                    const actionDiv = document.getElementById('action-buttons');
                    if(actionDiv) {
                        actionDiv.classList.remove('d-none');
                        actionDiv.style.display = 'flex';     
                    }
                } else {
                    alert("❌ Lỗi: " + data.error);
                }
            } catch (e) {
                alert("Lỗi kết nối server Ngrok!");
                console.error(e);
            } finally {
                if(btnConfirm) {
                    btnConfirm.innerHTML = 'Xác nhận tạo';
                    btnConfirm.disabled = false;
                }
            }
        }
        function executeGeneration(filename) {
            if (!filename.endsWith('.nc')) filename += ".nc";
            btnCreate.innerHTML = '⏳ Đang xử lý...';
            btnCreate.disabled = true;

            const payload = {
                filename: filename,
                offsets: pcbList.map(p => ({ id: p.id, x: p.x_mm, y: p.y_mm }))
            };

            fetch('/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'ok') {
                    if (successMsg) successMsg.innerText = `Đã tạo xong file: ${data.filename}`;              
                    document.querySelector('#successModal .modal-footer').innerHTML = `
                        <button type="button" class="btn btn-secondary" onclick="closeSuccessModal()">Đóng</button>
                        <a href="/preview?filename=${data.filename}" class="btn btn-primary fw-bold">
                            👁️ Xem mô phỏng (3D)
                        </a>
                    `;
                    if (successModal) successModal.classList.add('show');
                    
                    btnCreate.style.display = 'none';
                    actionDiv.classList.remove('d-none');
                    actionDiv.style.display = 'flex';     
                } else {
                    alert(`❌ Lỗi: ${data.error}`);
                    btnCreate.innerHTML = '🛠️ TẠO V-CODE';
                    btnCreate.disabled = false;
                }
            }).catch(err => {
                alert("Lỗi kết nối Server!");
                btnCreate.innerHTML = '🛠️ TẠO V-CODE';
                btnCreate.disabled = false;
            });
        }

        // --- C. XÂY DỰNG KHÔNG GIAN BÀN MÁY TUYỆT ĐỐI ---
        function updateScale() {
            // Xóa bỏ quyền tự co dãn của CSS
            bed.classList.remove('flex-grow-1');
            bed.classList.remove('h-100');
            
            // Tính toán tỷ lệ dựa trên chiều rộng cột chứa nó
            let parentW = bed.parentElement.clientWidth - 30; // Trừ padding
            if (parentW > 800) parentW = 800; // Khống chế độ rộng tối đa cho đẹp mắt
            
            pixelsPerMM = parentW / BOARD_REAL_W_MM;
            
            // Kích thước Pixel chính xác đến số thập phân của bàn máy
            const exactBedW_px = BOARD_REAL_W_MM * pixelsPerMM;
            const exactBedH_px = BOARD_REAL_H_MM * pixelsPerMM;
            
            // Ép khuôn bàn máy
            bed.style.width = exactBedW_px + 'px';
            bed.style.height = exactBedH_px + 'px';
            bed.style.position = 'relative';
            bed.style.boxSizing = 'content-box'; // Không cho viền xâm phạm vào không gian trong
            bed.style.overflow = 'hidden';       // Giấu ngay những pixel thừa nếu có
            bed.style.backgroundColor = '#1a202c'; // Màu bàn xám đen dễ nhìn

            // Cập nhật kích thước & vị trí cho các mạch đang có trên bàn
            pcbList.forEach(item => {
                const el = document.getElementById(`pcb-${item.id}`);
                if (el) {
                    const elW_px = PCB_W_MM * pixelsPerMM;
                    const elH_px = PCB_H_MM * pixelsPerMM;
                    
                    el.style.width = elW_px + 'px';
                    el.style.height = elH_px + 'px';
                    
                    const leftPx = item.x_mm * pixelsPerMM;
                    const topPx = exactBedH_px - (item.y_mm * pixelsPerMM) - elH_px;
                    
                    el.style.left = leftPx + 'px';
                    el.style.top = topPx + 'px';
                }
            });
        }

        function initPCBs() {
            updateScale();
            bed.innerHTML = '';
            pcbList = [];
            
            for (let i = 0; i < PCB_QTY; i++) {
                const el = document.createElement('div');
                el.className = 'pcb-workpiece';
                el.id = `pcb-${i}`;
                el.innerText = "#" + (i + 1);
                el.style.position = 'absolute';
                el.style.backgroundColor = 'rgba(255, 193, 7, 0.85)';
                el.style.border = '1px solid #ff9800';
                
                const elW_px = PCB_W_MM * pixelsPerMM;
                const elH_px = PCB_H_MM * pixelsPerMM;
                el.style.width = elW_px + 'px';
                el.style.height = elH_px + 'px';

                // Khởi tạo tọa độ mm (Xếp cách nhau ra 1 chút)
                let startX_mm = 2 + (i * (PCB_W_MM + 2)); 
                let startY_mm = 2;
                
                // --- BẮT BUỘC RÀNG BUỘC KỂ CẢ LÚC KHỞI TẠO ---
                startX_mm = Math.max(0, Math.min(startX_mm, BOARD_REAL_W_MM - PCB_W_MM));
                startY_mm = Math.max(0, Math.min(startY_mm, BOARD_REAL_H_MM - PCB_H_MM));

                pcbList.push({ id: i, x_mm: startX_mm, y_mm: startY_mm });

                const exactBedH_px = BOARD_REAL_H_MM * pixelsPerMM;
                const leftPx = startX_mm * pixelsPerMM;
                const topPx = exactBedH_px - (startY_mm * pixelsPerMM) - elH_px;
                
                el.style.left = leftPx + 'px'; 
                el.style.top = topPx + 'px';

                bed.appendChild(el);
                addDragEvent(el, i);
            }
            updatePanelDisplay();
        }

        function addDragEvent(element, index) {
            let isDragging = false;
            let startX, startY, initLeft, initTop;

            element.addEventListener('mousedown', (e) => {
                isDragging = true;
                startX = e.clientX; startY = e.clientY;
                initLeft = parseFloat(element.style.left) || 0; 
                initTop = parseFloat(element.style.top) || 0;
                
                element.style.zIndex = 1000; 
                element.style.cursor = 'grabbing';
                element.style.border = '2px solid #dc3545';
            });

            window.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                e.preventDefault(); 
                
                const dx = e.clientX - startX; 
                const dy = e.clientY - startY;
                let newLeft = initLeft + dx; 
                let newTop = initTop + dy;

                const exactBedW_px = BOARD_REAL_W_MM * pixelsPerMM;
                const exactBedH_px = BOARD_REAL_H_MM * pixelsPerMM;
                const elW_px = PCB_W_MM * pixelsPerMM;
                const elH_px = PCB_H_MM * pixelsPerMM;

                // TOÁN HỌC CHẶN BIÊN PIXEL TRÊN MÀN HÌNH
                const maxLeft_px = exactBedW_px - elW_px;
                const maxTop_px = exactBedH_px - elH_px;
                
                if (newLeft < 0) newLeft = 0; 
                if (newTop < 0) newTop = 0;
                if (newLeft > maxLeft_px) newLeft = maxLeft_px; 
                if (newTop > maxTop_px) newTop = maxTop_px;

                element.style.left = newLeft + 'px'; 
                element.style.top = newTop + 'px';
                
                // QUY ĐỔI RA MM (Chuẩn CNC)
                let x_mm = newLeft / pixelsPerMM;
                let y_mm = (exactBedH_px - newTop - elH_px) / pixelsPerMM;
                
                // KHÓA CHẶT DỮ LIỆU ĐẦU RA (Tọa độ không thể vọt khỏi 200x150)
                x_mm = Math.max(0, Math.min(x_mm, BOARD_REAL_W_MM - PCB_W_MM));
                y_mm = Math.max(0, Math.min(y_mm, BOARD_REAL_H_MM - PCB_H_MM));
                
                // Làm tròn 2 chữ số gửi cho server
                pcbList[index].x_mm = parseFloat(x_mm.toFixed(2)); 
                pcbList[index].y_mm = parseFloat(y_mm.toFixed(2));
                updatePanelDisplay();
            });

            window.addEventListener('mouseup', () => {
                if (isDragging) {
                    isDragging = false;
                    element.style.zIndex = ''; 
                    element.style.cursor = 'grab';
                    element.style.border = '1px solid #ff9800'; 
                }
            });
        }

        function updatePanelDisplay() {
            let html = '<ul class="list-group shadow-sm">';
            if(pcbList.length > 0) {
                 html += `<li class="list-group-item bg-light d-flex" style="font-weight:bold; font-size:0.9em; padding: 5px 10px;">
                            <span style="width: 70px;">ID</span>
                            <span class="ms-auto">TOẠ ĐỘ (mm)</span>
                          </li>`;
            }
            pcbList.forEach(item => {
                html += `<li class="list-group-item d-flex align-items-center" style="padding: 8px 10px;">
                            <div style="width: 70px;" class="fw-bold text-primary">#${item.id + 1}</div>
                            <div class="ms-auto text-end" style="font-family: monospace; font-size: 1.1em; color: #333;">
                                <span class="me-3">X: <b>${item.x_mm}</b></span>
                                <span>Y: <b>${item.y_mm}</b></span>
                            </div>
                        </li>`;
            });
            html += '</ul>';
            panel.innerHTML = html;
        }

        window.saveFile = function() { 
            const fname = localStorage.getItem('pcb_gcode_filename');
            window.location.href = `${NGROK_URL}/download?filename=${fname}`; 
        };
        
        window.uploadToPi = function() {
            if (!confirm("Nạp code vào máy CNC?")) return;
            fetch(`${NGROK_URL}/upload_serial`, { 
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: localStorage.getItem('pcb_gcode_filename') })
            })
            .then(r => r.json()).then(d => {
                alert(d.status === 'ok' ? "✅ Gửi thành công!" : "❌ Lỗi: " + d.error);
            });
        };

        window.addEventListener('resize', updateScale);
        initPCBs();
    }
});
