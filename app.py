
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------------------------------
# IMPORTANT: Custom layer definition for serialization
# ----------------------------------------------------
# This class must be defined in the same script that loads the model.
from tensorflow.keras.applications.densenet import preprocess_input

@tf.keras.saving.register_keras_serializable(package="MyLayers")
class DenseNetPreprocessLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return preprocess_input(inputs)

    def get_config(self):
        return super().get_config()
# ----------------------------------------------------

# 모델 로드 (캐싱을 통해 앱 실행 시 한 번만 로드)
@st.cache_resource
def load_model():
    model_path = 'best_densenet121.keras'
    # custom_objects 인자에 커스텀 레이어 클래스를 전달할 필요가 없습니다.
    # `@tf.keras.saving.register_keras_serializable` 데코레이터가 자동으로 처리합니다.
    model = tf.keras.models.load_model(model_path)
    return model

# 이미지 전처리 함수
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit 앱 인터페이스 구성
st.title('X-ray 기반 코로나19 진단기')
st.write('X-ray 이미지를 업로드하여 코로나19 양성 여부를 판단해 보세요.')

# 모델 로드
model = load_model()

# 파일 업로드 위젯
uploaded_file = st.file_uploader('X-ray 이미지 업로드', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', use_column_width=True)
    st.write("")

    if st.button('진단 시작'):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        prediction_score = prediction[0][0]

        st.subheader('진단 결과')
        if prediction_score >= 0.5:
            st.error(f'**코로나19 양성**일 가능성이 높습니다. (확률: {prediction_score:.2f})')
        else:
            st.success(f'**코로나19 음성**일 가능성이 높습니다. (확률: {1 - prediction_score:.2f})')
        st.info("이 결과는 참고용이며, 정확한 진단은 의료 전문가와 상담해야 합니다.")
