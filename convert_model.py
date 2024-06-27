import torch
import coremltools as ct
from RRDBNet import RRDBNet

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'models/RRDB_ESRGAN_x4.pth'
model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
model.load_state_dict(torch.load(model_path), strict=True)
model.to(device)
model.eval()

# PyTorch 모델을 Core ML 모델로 변환
input_example = torch.rand(1, 3, 256, 256).to(device)
traced_model = torch.jit.trace(model, input_example)

# Core ML 모델로 변환 (ML Program 형식 사용)
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=input_example.shape)],
    minimum_deployment_target=ct.target.iOS15
)

# 모델 저장
mlmodel.save("ESRGAN.mlpackage")
