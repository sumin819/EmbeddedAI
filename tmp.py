import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

# 1. 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MaskClassifier().to(device)

# 학습된 가중치 로드
model.load_state_dict(torch.load("mask_classifier.pth", map_location=device))
model.eval()

# 2. 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 3. 단일 이미지 추론 함수
def infer_image(image_path):
    image = Image.open(image_path).convert("RGB")  # 이미지 로드 및 RGB 변환
    input_tensor = transform(image).unsqueeze(0).to(device)  # 전처리 및 배치 차원 추가

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)  # 클래스 예측

    label = dataset.classes[pred.item()]  # 클래스 이름
    print(f"Prediction: {label}")
    return label

# 테스트용 단일 이미지 추론
image_path = "data/with_mask/example.jpg"  # 테스트 이미지 경로
infer_image(image_path)

# 4. 실시간 웹캠 추론 함수
def infer_webcam():
    cap = cv2.VideoCapture(0)  # 웹캠 열기

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV 이미지를 PIL 이미지로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)

        # 전처리 수행
        input_tensor = transform(image_pil).unsqueeze(0).to(device)

        # 추론
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)

        # 결과 표시
        label = dataset.classes[pred.item()]
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam Inference", frame)

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 실시간 웹캠 추론 실행
infer_webcam()
