
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from PIL import Image
import cv2 # OpenCV
import gdown # Google Drive에서 모델 다운로드를 위해 필요
import os # 파일 존재 여부 확인을 위해 필요

# ====================================================================
# 0. 설정 및 모델 로드
# ====================================================================

# Grad-CAM에 사용할 마지막 합성곱 레이어 이름 설정
# DenseNet121의 경우 보통 'relu' 또는 'conv5_block16_2_conv' 입니다.
LAST_CONV_LAYER_NAME = "densenet121"

st.set_page_config(page_title="COVID-19 X-ray Classifier", layout="wide")

# 모델 로딩 (Streamlit의 캐시 기능으로 한번만 로드)
@st.cache_resource
def load_keras_model():
    """
    구글 드라이브에서 모델을 다운로드하고 로드합니다.
    - 파일 ID는 구글 드라이브 공유 링크에서 'd/'와 '/view' 사이의 문자열입니다.
    """
    # *** 여기에 구글 드라이브 파일 ID를 입력하세요. ***
    # 예시: '1UMqaCmS2ahz9SPDgV5PrW1ztUcyErSsw'
    file_id = '19OGKeMEFT_pDNb-FXTVb4Yg7sKBvf1A8'
    output_path = 'best_densenet121.keras'
    
    # 파일이 존재하지 않을 경우에만 다운로드
    if not os.path.exists(output_path):
        with st.spinner("모델 파일을 다운로드 중입니다... 잠시만 기다려 주세요."):
            gdown.download(id=file_id, output=output_path, quiet=True)
            
    # Lambda 레이어의 'preprocess_input' 함수를 인식할 수 있도록
    # custom_objects를 사용하여 모델을 로드합니다.
    model = keras.models.load_model(
        output_path,
        custom_objects={'preprocess_input': preprocess_input}
    )
    return model

model = load_keras_model()

# ====================================================================
# 1. Grad-CAM 함수 구현
# ====================================================================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Grad-CAM 히트맵을 생성하는 함수
    """
    # 마지막 conv layer와 모델의 출력을 얻는 모델을 새로 구성
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # GradientTape를 사용하여 마지막 conv layer의 출력에 대한 top prediction의 그래디언트 계산
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 그래디언트 계산
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 채널별 평균 그래디언트 계산
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 마지막 conv layer의 출력에 평균 그래디언트를 곱하여 히트맵 생성
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU 활성화 함수처럼 음수는 0으로 만들고, 0과 1 사이로 정규화
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_gradcam(img, heatmap, alpha=0.4):
    """
    원본 이미지 위에 Grad-CAM 히트맵을 겹쳐서 보여주는 함수
    """
    # 히트맵을 0-255 범위의 8비트 이미지로 변환
    heatmap = np.uint8(255 * heatmap)

    # 'jet' 컬러맵 적용
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB) # OpenCV는 BGR, PIL은 RGB

    # 원본 이미지와 히트맵을 겹침
    superimposed_img = jet * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img

# ====================================================================
# 2. Streamlit UI 구성
# ====================================================================

st.title("흉부 X-ray COVID-19 분류 및 Grad-CAM 시각화  pneumonia-detection-using-x-ray-images")
st.write("DenseNet121 기반의 딥러닝 모델을 사용하여 COVID-19 감염 여부를 예측하고, AI가 어느 부분을 보고 판단했는지 히트맵으로 보여줍니다.")

# 파일 업로더
uploaded_file = st.file_uploader("흉부 X-ray 이미지를 업로드하세요 (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 이미지 로드 및 전처리
    pil_img = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(pil_img.resize((224, 224)))

    # 예측을 위한 전처리
    processed_img = np.expand_dims(img_array.copy(), axis=0)
    processed_img = preprocess_input(processed_img)

    # 예측 버튼
    if st.button("예측 실행"):
        with st.spinner('모델이 이미지를 분석 중입니다...'):
            # 예측 수행
            prediction = model.predict(processed_img)[0][0]
            is_positive = prediction > 0.5
            class_names = ['Negative', 'Positive'] # 훈련 시 라벨 순서에 맞게 조정
            result_text = f"**{class_names[int(is_positive)]}**일 확률이 **{prediction*100:.2f}%** 입니다."

            # Grad-CAM 생성
            heatmap = make_gradcam_heatmap(processed_img, model, LAST_CONV_LAYER_NAME)

            # 원본 이미지 위에 히트맵 겹치기
            superimposed_img = superimpose_gradcam(img_array, heatmap)

            # 결과 출력
            st.subheader("분석 결과")
            if is_positive:
                st.error(result_text)
            else:
                st.success(result_text)

            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_img, caption="원본 이미지", use_column_width=True)
            with col2:
                st.image(superimposed_img, caption="Grad-CAM 분석 결과", use_column_width=True)

            st.info("""
            **Grad-CAM 해석:**
            - **붉은색 영역**은 모델이 'Positive'라고 판단하는 데 가장 큰 영향을 미친 부분입니다.
            - **푸른색 영역**은 판단에 거의 영향을 미치지 않은 부분입니다.
            - 이 시각화는 모델의 판단을 해석하는 데 도움을 주지만, 100% 정확한 의학적 진단을 의미하지는 않습니다.
            """)
