import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import json  # Import necessário para lidar com o arquivo JSON


@st.cache_resource
def carrega_modelo():
    # Link direto para o modelo no Google Drive
    url = 'https://drive.google.com/uc?id=1ODh59KUq998DU3TmGP9ttXMZvbfxZvRl'
    
    # Baixa o arquivo
    gdown.download(url, 'modelo_quantizado16bits.tflite', quiet=False)
    
    # Carrega o modelo TensorFlow Lite
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()
    return interpreter


def carrega_classes():
    """
    Carrega os nomes das classes a partir do arquivo class_names.json.
    """
    try:
        with open("class_names.json", "r") as f:
            classes = json.load(f)
        return classes
    except Exception as e:
        st.error(f"Erro ao carregar as classes: {e}")
        return []


def ajustar_classes(interpreter, classes):
    """
    Ajusta o número de classes para corresponder ao número de saídas do modelo.
    """
    output_details = interpreter.get_output_details()
    numero_saidas = output_details[0]['shape'][1]  # Número de saídas do modelo

    if len(classes) < numero_saidas:
        st.warning("O arquivo JSON contém menos classes do que o modelo espera. Ajustando automaticamente.")
        classes.extend([f"Classe {i}" for i in range(len(classes), numero_saidas)])
    elif len(classes) > numero_saidas:
        st.warning("O arquivo JSON contém mais classes do que o modelo espera. Ajustando automaticamente.")
        classes = classes[:numero_saidas]

    return classes


def carrega_imagem():
    uploaded_file = st.file_uploader('Arraste e solte uma imagem aqui ou clique para selecionar uma', 
                                     type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image, caption="Imagem Original")
        st.success('Imagem foi carregada com sucesso')

        # Redimensionar a imagem para o tamanho esperado pelo modelo
        image = image.resize((256, 256))  # Substitua (256, 256) pelo tamanho que o modelo espera

        # Converter a imagem para array numpy
        image = np.array(image, dtype=np.float32)

        # Normalizar os valores dos pixels para o intervalo [0, 1]
        image = image / 255.0

        # Adicionar uma dimensão para representar o batch (modelo espera batch)
        image = np.expand_dims(image, axis=0)

        return image
    else:
        st.warning("Por favor, envie uma imagem válida.")
        return None


def previsao(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Certifique-se de que a imagem está no formato esperado
    expected_shape = input_details[0]['shape']
    if image.shape != tuple(expected_shape):
        st.error(f"Erro: o modelo espera uma imagem com o formato {expected_shape}, mas recebeu {image.shape}")
        return

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Obter os resultados da predição
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Carregar os nomes das classes do arquivo JSON
    classes = carrega_classes()
    if not classes:
        st.error("Erro: Não foi possível carregar as classes.")
        return

    # Ajustar as classes para o número esperado pelo modelo
    classes = ajustar_classes(interpreter, classes)

    # Criar DataFrame para visualização
    df = pd.DataFrame()
    df['classes'] = classes  # Usar os nomes das classes
    df['probabilidades (%)'] = 100 * output_data[0]

    # Ordenar por probabilidades e selecionar as top N classes
    top_n = 10  # Mostre as top 10 classes
    df = df.sort_values(by='probabilidades (%)', ascending=False).head(top_n)
    
    fig = px.bar(
        df,
        y='classes',
        x='probabilidades (%)',
        orientation='h',
        text='probabilidades (%)',
        title=f'Top {top_n} Classes Previstos pelo Modelo'
    )
    st.plotly_chart(fig)


def main():
    st.set_page_config(
        page_title="Classifica Peças Baldan",
        page_icon="🔧",
    )
    st.write("# Classifica Peças Baldan! 🔧")

    # Carrega o modelo
    interpreter = carrega_modelo()

    # Carrega a imagem
    image = carrega_imagem()

    # Faz a previsão
    if image is not None:
        previsao(interpreter, image)


if __name__ == "__main__":
    main()
