
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# VGG16 모델 로드 함수 (캐싱을 통해 앱 실행 시 한 번만 로드)
@st.cache_resource
def load_vgg16_model():
    # 구글 드라이브 파일 ID. 위 1단계에서 복사한 ID를 여기에 붙여넣으세요.
    file_id = 'https://drive.google.com/file/d/1UMqaCmS2ahz9SPDgV5PrW1ztUcyErSsw/view?usp=sharing' 
    output_path = 'best_vgg16.keras'

    # gdown을 사용하여 구글 드라이브에서 파일 다운로드
    # 이미 다운로드된 경우 다시 다운로드하지 않음 (cached=False 옵션을 통해 제어 가능)
    gdown.download(id=file_id, output=output_path, quiet=False)

    model = tf.keras.models.load_model(output_path)
    return model


# 이미지 전처리 함수
def preprocess_image_for_vgg16(image):
    # 모델의 입력 크기(224x224)에 맞게 이미지 크기 조정
    image = image.resize((224, 224))
    # numpy 배열로 변환
    img_array = np.array(image)
    # 3채널(RGB)로 변환 (만약 흑백 이미지인 경우)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    
    # 픽셀 값을 0-1 사이로 정규화 (VGG16의 기본 전처리)
    # DenseNet과 달리 별도의 복잡한 전처리 함수가 필요 없음
    img_array = img_array / 255.0

    # 배치 차원 추가
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit 앱 인터페이스 구성
st.title('VGG16 기반 X-ray 코로나19 진단기')
st.write('X-ray 이미지를 업로드하여 코로나19 양성 여부를 판단해 보세요.')

# 모델 로드
model = load_vgg16_model()

# 파일 업로드 위젯
uploaded_file = st.file_uploader('X-ray 이미지 업로드', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 이미지 로드 및 표시
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', use_column_width=True)
    st.write("")

    if st.button('진단 시작'):
        # 이미지 전처리
        processed_image = preprocess_image_for_vgg16(image)

        # 모델 예측
        prediction = model.predict(processed_image)
        prediction_score = prediction[0][0]

        # 결과 표시
        st.subheader('진단 결과')
        if prediction_score >= 0.5:
            st.error(f'**코로나19 양성**일 가능성이 높습니다. (확률: {prediction_score:.2f})')
        else:
            st.success(f'**코로나19 음성**일 가능성이 높습니다. (확률: {1 - prediction_score:.2f})')
        st.info("이 결과는 참고용이며, 정확한 진단은 의료 전문가와 상담해야 합니다.")
