import torch
import time
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, parameter_count
from ptflops import get_model_complexity_info

class ModelEvaluator:
    def __init__(self, model, device=None):
        """
        모델 평가 클래스 초기화
        Args:
            model (torch.nn.Module): 평가할 PyTorch 모델
            device (torch.device, optional): 사용할 디바이스 (CPU/GPU)
        """
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def evaluate_model_size(self):
        """
        모델의 총 파라미터 수와 학습 가능한 파라미터 수를 출력
        """
        params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total Parameters: {params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        return params, trainable_params

    def evaluate_flops(self, input_size=(3, 112, 112)):
        """
        모델의 FLOPs(Floating Point Operations) 계산
        Args:
            input_size (tuple): 입력 텐서 크기 (채널, 높이, 너비)
        """
        self.model.eval()
        input_tensor = torch.randn(1, *input_size).to(self.device)

        # FLOPs 계산 및 타입 변환 처리
        flops = FlopCountAnalysis(self.model, input_tensor)
        total_flops = flops.total()  # FLOPs 계산 결과
        if isinstance(total_flops, (int, float)):  # 안전한 타입 처리
            total_flops = float(total_flops)
        print(f"FLOPs: {total_flops / 1e6:.2f} MFLOPs")
        return total_flops


    def evaluate_inference_speed(self, input_size=(3, 112, 112), iterations=100):
        """
        모델의 평균 추론 속도 측정
        Args:
            input_size (tuple): 입력 텐서 크기 (채널, 높이, 너비)
            iterations (int): 추론 반복 횟수
        """
        self.model.eval()
        input_tensor = torch.randn(1, *input_size).to(self.device)
        torch.cuda.synchronize()  # GPU 사용 시 동기화

        start_time = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                self.model(input_tensor)
        torch.cuda.synchronize()  # GPU 사용 시 동기화
        end_time = time.time()

        avg_time_per_inference = (end_time - start_time) / iterations
        print(f"Average Inference Time: {avg_time_per_inference * 1000:.2f} ms")
        return avg_time_per_inference

    def summarize_model(self, input_size=(3, 112, 112)):
        """
        모델의 레이어별 출력 크기와 파라미터 요약
        Args:
            input_size (tuple): 입력 텐서 크기 (채널, 높이, 너비)
        """
        print("\n===== Model Summary =====")
        summary(self.model, input_size=input_size)

    def evaluate_all(self, input_size=(3, 112, 112), iterations=100):
        """
        모델의 모든 평가(파라미터 수, FLOPs, 실행 속도, 요약)를 수행
        Args:
            input_size (tuple): 입력 텐서 크기 (채널, 높이, 너비)
            iterations (int): 추론 반복 횟수
        """
        print("\n===== Model Evaluation =====")
        self.evaluate_model_size()
        self.evaluate_flops(input_size=input_size)
        self.evaluate_inference_speed(input_size=input_size, iterations=iterations)
        self.summarize_model(input_size=input_size)
