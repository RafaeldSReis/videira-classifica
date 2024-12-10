import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import os
import json


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


@st.cache_resource
def carrega_classes():
    # Diretório usado no treinamento para organizar as classes
    train_dir = "./dataset_treinamento"  # Substitua pelo caminho correto

    # Verificar se o diretório existe
    if not os.path.exists(train_dir):
        st.error(f"O diretório especificado não existe: {train_dir}")
        return []

    # Obter os nomes das classes a partir das subpastas
    classes = sorted(os.listdir(train_dir))
    
    # Salvar as classes em um arquivo JSON para uso futuro
    with open("classes.json", "w") as f:
        json.dump(classes, f)

    st.success(f"Número de classes carregadas: {len(classes)}")
    return classes


def carrega_imagem():
    uploaded_file = st.file_uploader('Arraste e solte uma imagem aqui ou clique para selecionar uma', 
                                     type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image, caption="Imagem Original")
        st.success('Imagem foi carregada com sucesso')

        # Redimensionar a imagem para o tamanho esperado pelo modelo (256x256)
        image = image.resize((256, 256))

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


def previsao(interpreter, image, classes):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Obter os resultados da predição
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Validar se o tamanho de classes é compatível com a saída do modelo
    if len(classes) != len(output_data[0]):
        st.error(f"Erro: o número de classes definidas ({len(classes)}) não corresponde ao número de saídas do modelo ({len(output_data[0])}).")
        return

    # Criar DataFrame para visualização
    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100 * output_data[0]

    # Exibir gráfico de probabilidades
    fig = px.bar(df, 
                 y='classes', 
                 x='probabilidades (%)',  
                 orientation='h', 
                 text='probabilidades (%)', 
                 title='Probabilidade de Classes')
    st.plotly_chart(fig)


def main():
    st.set_page_config(
        page_title="Classifica Folhas de Videira",
        page_icon="🍇",
    )
    st.write("# Classifica Folhas de Videira! 🍇")

    # Carrega modelo
    interpreter = carrega_modelo()

    # Carrega classes
    classes = carrega_classes()
    if not classes:
        return  # Sai do app se não houver classes

    # Carrega imagem
    image = carrega_imagem()

    # Classifica
    if image is not None:
        previsao(interpreter, image, classes)


if __name__ == "__main__":
    main()
