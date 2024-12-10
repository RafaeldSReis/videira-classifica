import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px


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


def carrega_classes_automaticamente(model_path):
    """
    Extrai as classes automaticamente do modelo.
    """
    try:
        # Aqui, você deve carregar as classes de onde o modelo foi treinado.
        # Ajuste conforme sua estrutura de treinamento.
        classes = [
            "ABRAÇADEIRA CLIP DE FIXAÇÃO PINHEIRO FURO 8,0MM 15 A 200MM",
            "ABRAÇADEIRA DE 044 X 057 X 14,5MM ROSCA SEM FIM AÇO",
            "ABRAÇADEIRA DE ALUMÍNIO COM REVEST. 1 INCH",
            "PARAFUSO ABC XYZ",
            # Adicione outros nomes de classes que correspondem ao treinamento
        ]
        return classes
    except Exception as e:
        st.error(f"Erro ao carregar classes automaticamente: {e}")
        return []


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

    # Carregar nomes reais das classes automaticamente
    classes = carrega_classes_automaticamente('modelo_quantizado16bits.tflite')
    if not classes:
        st.error("Não foi possível carregar as classes automaticamente.")
        return

    # Verificar se o número de classes no modelo corresponde ao número de nomes
    if len(classes) != len(output_data[0]):
        st.error(f"Erro: o número de classes ({len(classes)}) não corresponde à saída do modelo ({len(output_data[0])}).")
        return

    # Criar DataFrame para visualização
    df = pd.DataFrame()
    df['classes'] = classes  # Usar os nomes reais das classes
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
        page_title="Classifica Folhas de Videira",
        page_icon="🍇",
    )
    st.write("# Classifica Folhas de Videira! 🍇")

    # Carrega modelo
    interpreter = carrega_modelo()

    # Carrega imagem
    image = carrega_imagem()

    # Classifica
    if image is not None:
        previsao(interpreter, image)


if __name__ == "__main__":
    main()
