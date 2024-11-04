import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from PIL import Image
import os
import zipfile
import wget

# Configuración de la aplicación y título
st.set_page_config(page_title="Pediatric Chest X-ray Pneumonia Detection", page_icon="🫁", layout="centered")
st.title("🫁 SCI Pediatric Chest X-ray Pneumonia Classification")
st.markdown("<h4 style='text-align: center; color: gray;'>Aplicación de deep learning para la detección de neumonía en rayos X pediátricos</h4>", unsafe_allow_html=True)

# Función para preprocesar la imagen
def preprocess_image(image):
    img = image.resize((300, 300))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizar la imagen
    return img_array

# Función para realizar la predicción
def predict_pneumonia(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0][0]

# Función para descargar y extraer el modelo
@st.cache_resource
def download_and_extract_model():    
    model_url = 'https://dl.dropboxusercontent.com/s/vdwjryozdznulojz21bfj/best_model_saved.zip?rlkey=14fgn7ssuw11iq2a6e2barcrv&st=tqty8tup'
    zip_path = "best_model_saved.zip"
    extract_folder = "extracted_files"

    # Descargar el archivo zip si no existe
    if not os.path.exists(zip_path):
        try:
            wget.download(model_url, zip_path)
        except Exception as e:
            return None, f"Error al descargar el modelo: {e}"

    # Descomprimir el archivo
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
    except zipfile.BadZipFile:
        return None, "El archivo descargado no es un archivo ZIP válido."
    
    return os.path.join(extract_folder, 'best_model_saved.h5'), None

# Mostrar spinner para el proceso de carga y configuración del modelo
with st.spinner('Preparando el modelo, por favor espera...'):
    modelo_path, error = download_and_extract_model()

    # Verificar si hubo algún error durante la descarga o descompresión
    if error:
        st.error(error)
    elif not modelo_path or not os.path.exists(modelo_path):
        st.error("No se encontró el archivo del modelo.")
    else:
        # Definir el modelo base ResNet50V2 y cargar los pesos
        base_model = ResNet50V2(weights=None, include_top=False, input_shape=(300, 300, 3))
        base_model.trainable = False

        # Añadir capas de clasificación
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # Cargar los pesos
        try:
            model.load_weights(modelo_path)
            st.success("El modelo está listo para realizar predicciones.")
        except Exception as e:
            st.error(f"Error al cargar los pesos del modelo: {e}")

# Verificación de carga de archivo
uploaded_file = st.file_uploader("Elige una imagen de rayos X", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Radiografía cargada", use_column_width=True)

    st.markdown("<style>.stButton>button {background-color: #007bff; color: white; border-radius: 5px;}</style>", unsafe_allow_html=True)
    if st.button("Realizar diagnóstico"):
        with st.spinner('Analizando la imagen...'):
            prediction = predict_pneumonia(image, model)

            # Resultados
            prob_pneumonia = prediction * 100
            color = "red" if prob_pneumonia > 50 else "green"
            diagnosis = "NEUMONÍA DETECTADA" if prob_pneumonia > 50 else "NORMAL"
            
            st.markdown(f"<h3 style='color: {color}; text-align: center;'>{diagnosis}</h3>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            
            st.subheader("Probabilidades:")
            st.write("Probabilidad de neumonía:")
            st.progress(prob_pneumonia / 100, text=f"{prob_pneumonia:.1f}%")
            
            st.write("Probabilidad de normal:")
            st.progress((100 - prob_pneumonia) / 100, text=f"{100 - prob_pneumonia:.1f}%")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray;'><p>Desarrollado con ❤️ para SCI Pediatric Chest X-ray Competition</p></div>", unsafe_allow_html=True)
