from cv2 import cv2
import os
import streamlit as st
from ultralytics import YOLO
import uuid
from PIL import Image
import numpy as np

def app():
    # Mengatur konfigurasi halaman Streamlit
    st.set_page_config(page_title="Pendeteksi Objek")
    st.header('Deteksi gambar Menggunakan YOLOv8')
    st.subheader('created by: sya') 
    st.write('Unpload gambar dan pilih objek yang ingin Anda deteksi.')

    # Inisialisasi model YOLOv8
    model = YOLO('yolov8n.pt')
    object_names = list(model.names.values())

    # bikin form untuk input
    with st.form("my_form"):
        uploaded_file = st.file_uploader("Unggah gambar", type=['jpg', 'png'])
        selected_objects = st.multiselect('Pilih objek untuk dideteksi', object_names, default=['person'])
        min_confidence = st.slider('Skor kepercayaan', 0.0, 1.0, 0.25)
        submit_button = st.form_submit_button(label='Submit')
            
    # Proses gambar yang diunggah jika tombol submit ditekan
    if uploaded_file and submit_button:
        unique_id = str(uuid.uuid4().hex)[:8]  # Membuat ID unik untuk nama file
        input_path = os.path.join(os.getcwd(), f"temp_{unique_id}.jpg")

        try:
            # Menyimpan gambar yang diupload sementara di disk
            with open(input_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            # Membaca gambar yang diunggah
            image = Image.open(input_path)
            image = np.array(image)

            with st.spinner('Memproses gambar...'):
                # Melakukan deteksi objek menggunakan model YOLOv8
                result = model(image)
                for detection in result[0].boxes.data:
                    x0, y0 = (int(detection[0]), int(detection[1]))
                    x1, y1 = (int(detection[2]), int(detection[3]))
                    score = round(float(detection[4]), 2)
                    cls = int(detection[5])
                    object_name = model.names[cls]
                    label = f'{object_name} {score}'

                    # Menggambar kotak dan label di sekitar objek yang terdeteksi
                    if object_name in selected_objects and score > min_confidence:
                        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 2)
                        cv2.putText(image, label, (x0, y0 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Menampilkan hasil deteksi
                detections = result[0].verbose()
                cv2.putText(image, detections, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            st.image(image, caption='Gambar yang Diproses', use_column_width=True)

            # Menghapus file sementara setelah pemrosesan selesai
            if os.path.exists(input_path):
                os.remove(input_path)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    app()
