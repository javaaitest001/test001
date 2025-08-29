
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import gdown
from torchcam.methods import GradCAM
import os

# 1. 학습된 모델과 동일한 CheXNet 클래스 정의
# 로컬 pth 파일을 불러올 것이므로 pretrained는 사용하지 않습니다.
class CheXNet(nn.Module):
    def __init__(self, num_classes=1):
        super(CheXNet, self).__init__()
        # PyTorch 최신 권장 방식인 weights=None 사용
        densenet = models.densenet121(weights=None)
        num_ftrs = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_ftrs, num_classes)
        self.model = densenet

    def forward(self, x):
        return self.model(x)

# 2. 모델 로딩 (Streamlit의 캐시 기능으로 한번만 로드)
@st.cache_resource
def load_pytorch_model():
    """
    구글 드라이브에서 모델을 다운로드하고 로드합니다.
    - 파일 ID는 구글 드라이브 공유 링크에서 'd/'와 '/view' 사이의 문자열입니다.
    - 예시: https://drive.google.com/file/d/19OGKeMEFT_pDNb-FXTVb4Yg7sKBvf1A8/view?usp=sharing
      -> file_id는 '19OGKeMEFT_pDNb-FXTVb4Yg7sKBvf1A8' 입니다.
    """
    # ★★★ 여기에 구글 드라이브 파일 ID를 입력하세요. ★★★
    # 이 ID는 사용자의 best_chexnet.pth 파일의 ID로 바꿔주셔야 합니다.
    file_id = '16wzBCEg8eYV8gaDf90c15SIbp7KPFb1p' # 예시 ID
    output_path = 'best_chexnet.pth'

    # 파일이 존재하지 않을 경우에만 다운로드
    if not os.path.exists(output_path):
        with st.spinner("모델 파일을 구글 드라이브에서 다운로드 중입니다... 잠시만 기다려 주세요."):
            # gdown 라이브러리를 사용하여 파일 ID로 다운로드
            gdown.download(id=file_id, output=output_path, quiet=True)
            st.success("모델 다운로드 완료!")
    
    # 모델 구조 정의
    model = CheXNet(num_classes=1)
    
    # 모델 가중치 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(output_path, map_location=device))
    model.eval()
    
    return model.to(device)

# 3. 이미지 전처리 및 모델 예측 함수
def predict(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        # 'Covid-19'가 0, 'Normal'이 1이므로 0.5보다 작으면 '코로나'로 예측합니다.
        prediction = "코로나19" if prob < 0.5 else "정상"
        
    return prediction, prob

# 4. Grad-CAM 히트맵 생성 함수
def generate_grad_cam(model, image_path, device):
    model.eval()
    
    # Grad-CAM 객체 생성 (마지막 특징 추출 레이어 지정)
    # DenseNet121의 마지막 특징 추출 레이어는 'features'
    cam_extractor = GradCAM(model, target_layer=model.model.features)
    
    # PIL 이미지 로드 및 전처리
    pil_img = Image.open(image_path).convert('RGB')
    transform_cam = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor_cam = transform_cam(pil_img).unsqueeze(0).to(device)

    # Grad-CAM 히트맵 계산을 위해 모델에 입력
    out = model(input_tensor_cam)
    
    # 예측된 클래스 인덱스 (0: 코로나, 1: 정상)를 얻습니다.
    # 이진 분류이므로 확률에 따라 0 또는 1을 사용합니다.
    predicted_class_idx = 0 if torch.sigmoid(out).item() < 0.5 else 1
    
    # out.squeeze()로 차원을 줄여 cam_extractor에 전달
    cam = cam_extractor(predicted_class_idx, out.squeeze(0))[0]
    
    # 히트맵 시각화
    cam_np = cam.cpu().numpy()
    
    # 원본 이미지와 히트맵 합성
    original_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img, pil_img

# 5. Streamlit UI 구성
st.title("코로나19 흉부 X-ray 진단 및 Grad-CAM 시각화")
st.markdown("---")

# 모델 로드 (캐시된 함수 호출)
model = load_pytorch_model()
device = model.device

st.sidebar.header("이미지 업로드")
uploaded_file = st.sidebar.file_uploader("X-ray 이미지 파일을 선택하세요", type=["jpg", "png"])

if uploaded_file is not None:
    # 파일명으로 임시 저장
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 이미지 열기
    image = Image.open(uploaded_file)
    
    st.subheader("업로드된 이미지")
    st.image(image, use_column_width=True)
    
    # 예측 수행
    st.subheader("모델 예측 결과")
    prediction, probability = predict(image, model, device)
    
    st.write(f"예측 결과: **{prediction}**")
    st.write(f"예측 확률: **{probability:.4f}**")
    
    # Grad-CAM 시각화
    st.subheader("Grad-CAM 시각화 (모델이 주목한 부분)")
    cam_image, _ = generate_grad_cam(model, temp_image_path, device)
    st.image(cam_image, caption="Grad-CAM 히트맵", use_column_width=True)
    
    st.info("빨간색에 가까울수록 모델이 **'코로나19'**를 판단하는 데 중요하게 본 영역입니다.")
