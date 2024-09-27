import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import time
from scipy.spatial import distance

# Memuat model YOLOv8
model = YOLO('lasttrain.pt')

# Mengatur konfigurasi halaman Streamlit
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Pengaturan sidebar
st.sidebar.header('Deteksi Area Parkir oleh: JKAS')

# Pemilihan sumber input
input_source = st.sidebar.selectbox("Pilih sumber input", ["Gambar", "Video", "Live Stream"])

# Pengaturan Confidence dan IOU
confidence = st.sidebar.slider("Ambang Batas Confidence", 0.0, 1.0, 0.5)
iou_threshold = st.sidebar.slider("Ambang Batas IOU", 0.0, 1.0, 0.5)

# Pengaturan checkbox untuk menampilkan label dan ID
show_labels = st.sidebar.checkbox("Tampilkan Label", True)
show_ids = st.sidebar.checkbox("Tampilkan ID", True)

# Penyimpanan ID slot dan posisinya
slot_positions = {}
base_letter = 'A'
id_counter = 1

# Fungsi untuk menetapkan ID yang konsisten untuk setiap slot parkir
def assign_slot_id(x1, y1, x2, y2, threshold=50):
    global id_counter
    
    box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    closest_id = None
    closest_distance = float('inf')
    
    for slot_id, pos in slot_positions.items():
        dist = distance.euclidean(box_center, pos)
        if dist < closest_distance and dist < threshold:
            closest_distance = dist
            closest_id = slot_id
    
    if closest_id is None:
        # Menetapkan ID baru jika tidak ada slot yang cukup dekat
        closest_id = f"{base_letter}{id_counter}"
        slot_positions[closest_id] = box_center
        id_counter += 1
    else:
        # Memperbarui posisi untuk slot yang sudah ada
        slot_positions[closest_id] = box_center
    
    return closest_id

# Fungsi untuk mendeteksi objek pada gambar menggunakan YOLOv8
def detect_objects(image, conf_threshold, iou_threshold):
    results = model(image, conf=conf_threshold, iou=iou_threshold)
    return results

# Fungsi untuk menggambar kotak pembatas dan menampilkan FPS serta jumlah slot dengan ID yang konsisten
def draw_boxes(image, results, fps=None):
    num_free_slots = 0
    num_occupied_slots = 0
    empty_slots = []  # List untuk menyimpan ID slot yang kosong
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            slot_id = assign_slot_id(x1, y1, x2, y2)
            
            # Menghitung slot kosong dan terisi
            if model.names[class_id] == "free":
                num_free_slots += 1
                empty_slots.append(slot_id)  # Tambahkan ID slot kosong ke list
                label_color = (0, 255, 0)  # Hijau untuk slot kosong
            elif model.names[class_id] == "car":
                num_occupied_slots += 1
                label_color = (0, 0, 255)  # Merah untuk mobil
            
            if show_labels:
                label = f"{model.names[class_id]} {confidence:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1)
                
            if show_ids:
                id_label = f"ID:{slot_id}"
                cv2.putText(image, id_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1)

            cv2.rectangle(image, (x1, y1), (x2, y2), label_color, 2)

    if fps is not None:
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(image, fps_text, (image.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Menampilkan total slot, jumlah kosong, dan terisi
    total_slots = num_free_slots + num_occupied_slots
    parking_summary = f"Total: {total_slots} Kosong: {num_free_slots} Mobil: {num_occupied_slots}"
    cv2.putText(image, parking_summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image, num_free_slots, num_occupied_slots, empty_slots

# Pemrosesan input berdasarkan sumber yang dipilih
if input_source == "Gambar":
    uploaded_file = st.sidebar.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        results = detect_objects(image, confidence, iou_threshold)
        image, num_free_slots, num_occupied_slots, empty_slots = draw_boxes(image, results)

        st.image(image, channels="BGR", caption="Deteksi Awal")

        # Menampilkan jumlah slot parkir
        col1, col2 = st.columns(2)
        col1.metric("Jumlah Slot Parkir Kosong", num_free_slots)
        col2.metric("Jumlah Slot Parkir Terisi", num_occupied_slots)
        
        # Tampilkan status parkir
        warning_placeholder = st.empty()
        if num_free_slots == 0:
            warning_placeholder.warning("Parkir Penuh")
        else:
            empty_slots_str = ", ".join(empty_slots)  # Gabungkan list ID menjadi string
            warning_placeholder.success(f"Slot Kosong: {empty_slots_str}")

elif input_source == "Video":
    uploaded_file = st.sidebar.file_uploader("Unggah video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())

        video_cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        warning_placeholder = st.empty()  # Tempat untuk menampilkan status slot kosong atau penuh

        prev_time = time.time()
        results = []  # Pastikan results didefinisikan sebelum digunakan
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break
            
            # Melakukan deteksi di setiap frame
            results = detect_objects(frame, confidence, iou_threshold)

            # Menghitung FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if current_time - prev_time > 0 else 0
            prev_time = current_time
            
            frame, num_free_slots, num_occupied_slots, empty_slots = draw_boxes(frame, results, fps)

            stframe.image(frame, channels="BGR", use_column_width=True)

            # # Tampilkan status parkir
            # if num_free_slots == 0:
            #     warning_placeholder.warning("Parkir Penuh")
            # else:
            #     empty_slots_str = ", ".join(empty_slots)
            #     warning_placeholder.success(f"Slot Kosong: {empty_slots_str}")


            # Tampilkan status parkir
            if num_free_slots == 0:
                warning_placeholder.markdown("<h3 style='color: red;'>Parkir Penuh</h3>", unsafe_allow_html=True)
            else:
                empty_slots_str = ", ".join(empty_slots)
                warning_placeholder.markdown(f"<h3 style='color: green;'>Slot Kosong: {empty_slots_str}</h3>", unsafe_allow_html=True)
            

        video_cap.release()

elif input_source == "Live Stream":
    camera_index = 0  # Menggunakan kamera default (biasanya kamera laptop)
    video_cap = cv2.VideoCapture(camera_index)
    stframe = st.empty()
    warning_placeholder = st.empty()  # Tempat untuk menampilkan status slot kosong atau penuh

    prev_time = time.time()
    results = []  # Pastikan results didefinisikan sebelum digunakan
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break
        
        # Melakukan deteksi di setiap frame
        results = detect_objects(frame, confidence, iou_threshold)

        # Menghitung FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time - prev_time > 0 else 0
        prev_time = current_time
        
        frame, num_free_slots, num_occupied_slots, empty_slots = draw_boxes(frame, results, fps)

        stframe.image(frame, channels="BGR", use_column_width=True)

        # Tampilkan status parkir
        if num_free_slots == 0:
            warning_placeholder.warning("Parkir Penuh")
        else:
            empty_slots_str = ", ".join(empty_slots)
            warning_placeholder.success(f"Slot Kosong: {empty_slots_str}")

    video_cap.release()