## 1. 수행 Task

- 주제: Face Mask 착용 여부를 분류하는 Binary classification Task
- 데이터셋
    - Kaggle Dataset 사용(https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection)
    - with_mask: 1,620장, 128x128, RGB
    - without_mask: 1,656장, 128x128, RGB
- 사용 입출력 장치: 카메라

## 2. 기본 모델 및 공통 코드 설명

### (1) training

- Base 모델

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/f3a72ca0-5198-4cf9-bb32-097d5a7032ca/image.png)
- 학습 코드

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/658758a1-1c45-41f5-9311-e6add9b98b4c/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/903c8584-1a88-435a-ab3f-50a0a6a9a5d1/image.png)

- 테스트 코드

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/8e217108-01b7-483b-95ab-b362437e20ed/image.png)

### (2) inference

젯슨 나노에서 모델을 inference 해 실시간 추론을 시행 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/e98810e4-dcbd-4217-a4c2-8a0ccebf5791/image.png)

`infer_webcam` 함수

- 노트북 상에서 inference한 모델을 웹캠을 사용해 test 하는 코드
- 추론 시간(실행 시간)을 측정
- cv2의 `cascade classifier`를 사용해 얼굴을 먼저 detect한 후, 마스크 착용 여부를 구별

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/a0203baf-4b79-49ab-8350-69d67d2d69d1/image.png)

`infer_csi_camera` 함수

- Jetson-Nano의 CSI 카메라를 사용해 실시간으로 추론하는 함수
- 추론 시간(실행 시간)을 측정
- `GStreamer` 사용
- cv2의 `cascade classifier`를 사용해 얼굴을 먼저 detect한 후, 마스크 착용 여부를 구별

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/db76ab57-cd7e-48b0-823b-52ad01421aa2/image.png)

`count_parameters` 함수

- 파라미터 개수를 count
- zero 파라미터와 non-zero 파라미터를 구분

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/d423617f-82e1-4336-b21e-a10cb161fa9e/image.png)

`calculate_flops` 함수

- `from thop import profile` 사용
- profile로 구한 것은 MACs 이기 때문에, FLOPs 구하기 위해 macs ***2** 계산

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/67c1cefe-861c-4a2e-aab4-22e856941e20/image.png)

`get_model_size` 함수

- 모델 사이즈를 계산

## 3. 사용한 경량화 기법

### (1) Knowledge Distillation

### training_distillation_total.ipynb

`Student` **모델의 설계 간소화**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/d1504023-f971-4e2b-a2f5-691294c19ded/image.png)

`SmallMaskClassifier` **모델의 설계 간소화**:

- 합성곱 계층의 필터 수를 대폭 감소
    - Teacher 모델: 32→64→128 채널
    - Small 모델: 8→16→32 채널 (약 75% 감소된 채널 수)
- Dropout 층 제거로 구조 단순화
    - Teacher 모델: Dropout(0.2) 사용
    - Small 모델: Dropout 층 완전히 제거
- 구조를 크게 간소화하여 파라미터 수 93% 이상 감소
    - Teacher 모델: 93,954개 파라미터
    - Small 모델: 6,210개 파라미터

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/c60a74eb-ed33-4fa7-8824-bc539c3dc3d2/image.png)

`MediumMaskClassifier` **모델 설계 간소화:**

- 합성곱 계층의 필터 수를 중간 정도로 감소
    - Teacher 모델: 32→64→128 채널
    - Medium 모델: 24→48→96 채널 (약 25% 감소된 채널 수)
- Dropout 비율 조정
    - Teacher 모델: 0.2
    - Medium 모델: 0.1 (더 적은 정규화)
- 구조는 Teacher와 동일하게 유지하면서 파라미터 수만 약 43% 감소
    - Teacher 모델: 93,954개 파라미터
    - Medium 모델: 53,186개 파라미터

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/67138438-c7e5-4cd7-9b2c-a3f663154c26/image.png)

`Knowledge Distillation` **적용**:

- Teacher 모델에서 얻은 소프트 라벨(출력)을 Student 모델에 전달하여 학습 성능을 유지.
- Knowledge Distillation을 통해 Student 모델이 Teacher 모델의 지식을 효율적으로 압축 및 학습하도록 설계.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/8823d4fa-da73-47e3-84a2-74d150d6ef4c/image.png)

- Student Model 학습 코드

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/915c0b5f-05c3-455f-8095-72b6f0d6de92/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/3ed88c76-b9bc-41ae-877a-ecfb889f5754/image.png)

- 경량화 전 후의 파라미터 비교

### (2) Pruning

### training_prune_total.ipynb

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/6468c9dd-1e82-4608-ac0b-b5393f2cfdd7/image.png)

`apply_pruning` 함수:

- `torch_pruning` 사용
- `MagnitudePruner`를 사용하여 채널의 중요도가 낮은 순서대로 제거
- 마지막 분류 층(classifier)은 프루닝에서 제외
- 30%, 50%, 70% 로 Pruning Ratio를 다르게 해 경량화

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/c3baa3fc-9808-4aa2-be6e-1305d189fdb1/image.png)

- 경량화 전 후의 파라미터 비교

## 4. Jetson nano에서의 경량화 효과 (젯슨 나노-주피터 노트북에서 실행 후, html 다운로드해 화면 캡쳐 한 것

## (1) Distillation Model

- 잿슨나노의 CSI 카메라 사용 시의 추론 시간 비교

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/bc7c7bb4-52b5-4fa7-bb5c-a0ef96803a6c/image.png)

- 파라미터 수 비교

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/da42ec1f-127f-4f7e-9c80-7bd27fb7cbc7/image.png)

- 모델 사이즈 비교

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/98aee064-f005-4ade-b031-45bb5a818321/image.png)

- FLOPs 비교

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/95b88243-fe97-4395-b414-aa1729876cf5/image.png)

- 추론 시간 비교(100회 추론 비교)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/97dac0c7-4b85-41dd-aa16-1e826867c3a0/image.png)

<aside>
💡

Knowledge Distillation을 통해 Student 모델들의 크기를 대폭 줄이면서(Small: 93.4%, Medium: 43.4% 감소) 성능은 유지했습니다. Small Student 모델의 경우 파라미터 수가 6,210개로 크게 감소했고, 추론 시간도 Teacher 모델(72.48ms)에 비해 상당히 개선되어 20.76ms로 단축되었습니다.

실제 젯슨나노 CSI 카메라를 사용해 테스트 했을 때, Base 모델의 대부분의 파라미터를 감소시킨 Small Student Model에서 without mask를 classification하는데 성능이 약간 떨어짐을 볼 수 있었습니다. 

</aside>

## (2) Pruning Distillation 모델

### a. 30%, 50%, 70%로 진행

- 잿슨나노의 CSI 카메라 사용 시의 추론 시간 비교

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/867b2fac-aeef-4e22-bf59-e8f17eeb406a/image.png)

- 파라미터 수 비교

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/1b1c8b4a-edaa-45a9-bd00-d2097ed872db/image.png)

- 모델 사이즈 비교

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/384bd8c7-1d60-4b97-b346-4131133c4c99/image.png)

- FLOPs 비교

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/c4680512-a8ca-4e14-a957-06e8733bd2c4/image.png)

- 추론 시간 비교(100회 평균)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6336caf6-8680-43b9-a4b4-934deaf0affa/73fd268c-dd6b-4333-9e54-83798b8e4d58/image.png)

<aside>
💡

채널 프루닝을 적용한 결과, 원본 모델(93,954 파라미터, 0.36MB)에 비해 30% 프루닝 시 45,195 파라미터(0.17MB)로 감소하고 추론 속도는 25.71ms에서 15.77ms로 개선되었습니다. 50% 프루닝의 경우 23,938 파라미터(0.09MB)로 줄었으며 추론 시간은 12.19ms로 단축되었고, 70% 프루닝에서는 8,556 파라미터(0.03MB)와 8.98ms의 추론 시간을 달성했습니다. FPS도 원본 모델의 5.1에서 각각 5.5(30%), 6.5(50%), 6.8(70%)로 점진적으로 향상되었습니다.

실제 젯슨나노 CSI 카메라를 사용해 테스트 했을 때, 70%, 50% 비율로 Pruning시킨 모델의 성능은 Base Model과 비슷하였으나, 30% 비율로 Pruning시킨 모델의 성능은 약간 떨어졌으나, 대부분 classification에 성공하였음을 볼 수 있었습니다. 

</aside>

## 5. 제출 코드

- training.ipynb: Base Model 학습 코드
- training_distillation_total.ipynb: Distillation(Small Student, Medium Student) Model 학습 코드
- training_prune_total.ipynb: Pruning(30%, 50%, 70%) Model 학습 코드
- inference_distillation_total.ipynb: 학습된 Distillation Model을 inference하는 코드
- inference_prune_total.ipynb: 학습된 Pruning Model을 inference하는 코드
