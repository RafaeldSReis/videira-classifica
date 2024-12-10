import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
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
    # Carrega os nomes das classes do arquivo JSON
    with open("class_namess.json", "r") as f:
        class_names = json.load(f)
    return class_names


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


def numero_classes_do_modelo(interpreter):
    output_details = interpreter.get_output_details()
    num_classes = output_details[0]['shape'][1]
    return num_classes


def previsao(interpreter, image, class_names):
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

    # Verificar se o número de classes no modelo coincide com o arquivo class_names.json
    if len(output_data[0]) != len(class_names):
        st.error("Erro: o número de classes no modelo não coincide com o arquivo class_names.json")
        return

    # Criar DataFrame para visualização
    df = pd.DataFrame()
    df['classes'] = [class_names[i] for i in range(len(output_data[0]))]
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

    # Exibir a classe com maior probabilidade
    top_class = df.iloc[0]
    st.success(f"A peça identificada é: {top_class['classes']} com {top_class['probabilidades (%)']:.2f}% de certeza.")

    
    with open("class_namess.json", "r") as f:
        class_names = json.load(f)
    
    print(f"O arquivo JSON contém {len(class_names)} classes.")


def main():
    st.set_page_config(
        page_title="Classifica Folhas de Videira",
        page_icon="🍇",
    )
    st.write("# Classifica Folhas de Videira! 🍇")

    # Carrega modelo
    interpreter = carrega_modelo()

    # Exibe o número de classes do modelo
    num_classes = numero_classes_do_modelo(interpreter)
    st.info(f"O modelo possui {num_classes} classes.")

    # Carrega os nomes das classes
    class_names = carrega_classes()

    # Carrega imagem
    image = carrega_imagem()

    # Classifica
    if image is not None:
        previsao(interpreter, image, class_names)


if __name__ == "__main__":
    main()
