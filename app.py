import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import os


@st.cache_resource
def carrega_modelo():
    """
    Baixa e carrega o modelo TensorFlow Lite.
    """
    url = 'https://drive.google.com/uc?id=1ODh59KUq998DU3TmGP9ttXMZvbfxZvRl'
    gdown.download(url, 'modelo_quantizado16bits.tflite', quiet=False)
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()
    return interpreter


def gerar_lista_classes(diretorio_base):
    """
    Gera a lista de classes com base nas subpastas do diretório de treinamento.
    """
    try:
        classes = sorted([f.name for f in os.scandir(diretorio_base) if f.is_dir()])
        if not classes:
            raise Exception("Nenhuma classe encontrada no diretório base.")
        return classes
    except Exception as e:
        print(f"Erro ao gerar lista de classes: {e}")
        return []


def carrega_classes_automaticamente():
    """
    Carrega os nomes das classes automaticamente a partir do diretório base de treinamento.
    """
    diretorio_base = 'pecas_imagens'  # Substitua pelo caminho correto
    classes = gerar_lista_classes(diretorio_base)
    if not classes:
        raise Exception("Erro: Não foi possível carregar as classes do diretório.")
    return classes


def carrega_imagem():
    """
    Faz o upload da imagem e a prepara para a entrada no modelo.
    """
    uploaded_file = st.file_uploader('Arraste e solte uma imagem aqui ou clique para selecionar uma', 
                                     type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        st.image(image, caption="Imagem Original")
        st.success('Imagem foi carregada com sucesso')
        image = image.resize((256, 256))  # Ajuste o tamanho conforme necessário
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    else:
        st.warning("Por favor, envie uma imagem válida.")
        return None


def previsao(interpreter, image):
    """
    Faz a previsão com o modelo e exibe os resultados.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    expected_shape = input_details[0]['shape']
    if image.shape != tuple(expected_shape):
        st.error(f"Erro: o modelo espera uma imagem com o formato {expected_shape}, mas recebeu {image.shape}")
        return

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Carregar as classes do diretório
    classes = carrega_classes_automaticamente()
    if len(classes) != len(output_data[0]):
        st.error(f"Erro: o número de classes ({len(classes)}) não corresponde à saída do modelo ({len(output_data[0])}).")
        return

    # Criar o DataFrame para visualização
    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100 * output_data[0]
    top_n = 10
    df = df.sort_values(by='probabilidades (%)', ascending=False).head(top_n)

    # Exibir o gráfico
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
    """
    Função principal para execução do aplicativo Streamlit.
    """
    st.set_page_config(page_title="Classifica Peças", page_icon="🔧")
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
