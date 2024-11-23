import torch
from torchvision import models
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms

# 체크포인트 파일 경로
checkpoint_path = "model_best_20241122-233501.pth.tar"

# 모델 초기화 및 로드
model_arch = "efficientnet-b1"  # 모델 아키텍처
if "efficientnet" in model_arch:
    model = EfficientNet.from_name(model_arch)
else:
    model = models.__dict__[model_arch]()

# 학습된 가중치 로드
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()  # GPU 사용 시
model.eval()  # 평가 모드로 전환

# 이미지 경로
image_path = "./test_data/tower.jpg"

# 이미지 전처리
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 모델 입력 크기에 맞춤
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 이미지 불러오기 및 변환
image = Image.open(image_path).convert("RGB")  # RGB로 변환
input_tensor = image_transforms(image).unsqueeze(0)  # 배치 차원 추가
input_tensor = input_tensor.cuda()  # GPU 사용 시

# 모델 예측
with torch.no_grad():
    output = model(input_tensor)

# 확률 계산 (Softmax)
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# 상위 k개 클래스 추출
topk_probs, topk_indices = torch.topk(probabilities, k=5)

# 클래스 이름 매핑 로드 (class_mapping.txt 파일에서 읽기)
class_mapping_file = "class_mapping.txt"


idx_to_class = {}
with open(class_mapping_file, "r") as f:
    for line in f:
        idx, class_name = line.strip().split(": ")
        idx_to_class[int(idx)] = class_name

# 모델 인퍼런스 결과 출력
for i in range(topk_probs.size(0)):
    class_name = idx_to_class.get(topk_indices[i].item(), "Unknown")  # 인덱스를 클래스 이름으로 변환
    print(f"{class_name}: {topk_probs[i].item() * 100:.2f}%")

