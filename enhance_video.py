import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from RRDBNet import RRDBNet
import time

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'models/RRDB_ESRGAN_x4.pth'
model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
model.load_state_dict(torch.load(model_path), strict=True)
model.to(device)
model.eval()


# 이미지 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)


# 이미지 후처리 함수
def postprocess_image(tensor):
    tensor = tensor.squeeze().detach().cpu().numpy()
    tensor = (tensor * 0.5 + 0.5) * 255.0
    tensor = tensor.clip(0, 255).astype(np.uint8)
    return Image.fromarray(tensor.transpose(1, 2, 0))


# 동영상 프레임 화질 개선 함수 (진행 상황 표시 추가)
def enhance_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 4), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 4)))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_image = postprocess_image(output_tensor)
        output_frame = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
        out.write(output_frame)

        # 진행 상황 출력
        elapsed_time = time.time() - start_time
        print(f"Processed frame {i + 1}/{frame_count}, Elapsed time: {elapsed_time:.2f}s")

    cap.release()
    out.release()


# 예제 실행
if __name__ == "__main__":
    input_video_path = 'input_video.mp4'
    output_video_path = 'output_video.mp4'
    enhance_video(input_video_path, output_video_path)
