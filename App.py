import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Configuração MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

st.title("VisageScore - Analisador de Beleza Facial")

uploaded_file = st.file_uploader("Escolha uma foto", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape

    resultados = face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    if resultados.multi_face_landmarks:
        rosto = resultados.multi_face_landmarks[0]
        pontos = [(int(lm.x*w), int(lm.y*h)) for lm in rosto.landmark]

        olho_esq, olho_dir = pontos[33], pontos[263]
        distancia_olhos = np.linalg.norm(np.array(olho_esq) - np.array(olho_dir))
        st.write(f"Distância entre olhos: {distancia_olhos:.1f}px")

        # Mostrar imagem com landmarks
        for x, y in pontos:
            cv2.circle(img_cv, (x, y), 1, (0, 255, 0), -1)

        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Rosto detectado", use_column_width=True)
    else:
        st.error("❌ Nenhum rosto detectado.")
