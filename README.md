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

<img width="251" alt="image" src="https://github.com/user-attachments/assets/88d76dc1-fe55-43b1-a325-9d3273da9f60" />
- 학습 코드

<img width="251" alt="image" src="https://github.com/user-attachments/assets/d33c4c3f-87dc-4adb-954c-6e58966f926f" />

<img width="252" alt="image" src="https://github.com/user-attachments/assets/990ee4de-9bef-4bff-98d5-904c4d1cda44" />

- 테스트 코드

<img width="576" alt="image" src="https://github.com/user-attachments/assets/42460138-0116-4913-ac94-a02c8309f581" />

### (2) inference

젯슨 나노에서 모델을 inference 해 실시간 추론을 시행 

<img width="551" alt="image" src="https://github.com/user-attachments/assets/95de9536-d58f-4c53-85fd-9d28a362fd61" />

`infer_webcam` 함수

- 노트북 상에서 inference한 모델을 웹캠을 사용해 test 하는 코드
- 추론 시간(실행 시간)을 측정
- cv2의 `cascade classifier`를 사용해 얼굴을 먼저 detect한 후, 마스크 착용 여부를 구별

<img width="550" alt="image" src="https://github.com/user-attachments/assets/75dcaaaf-fc34-437f-a21d-c58b02245e5d" />

`infer_csi_camera` 함수

- Jetson-Nano의 CSI 카메라를 사용해 실시간으로 추론하는 함수
- 추론 시간(실행 시간)을 측정
- `GStreamer` 사용
- cv2의 `cascade classifier`를 사용해 얼굴을 먼저 detect한 후, 마스크 착용 여부를 구별

<img width="477" alt="image" src="https://github.com/user-attachments/assets/8ff4f37c-23b1-4fb8-b1d1-15b85529086d" />

`count_parameters` 함수

- 파라미터 개수를 count
- zero 파라미터와 non-zero 파라미터를 구분

<img width="473" alt="image" src="https://github.com/user-attachments/assets/78055edb-b954-45bb-8c26-ddc07a615494" />

`calculate_flops` 함수

- `from thop import profile` 사용
- profile로 구한 것은 MACs 이기 때문에, FLOPs 구하기 위해 macs ***2** 계산

<img width="429" alt="image" src="https://github.com/user-attachments/assets/63b736dd-3206-433c-8a3a-da5a2ead4423" />

`get_model_size` 함수

- 모델 사이즈를 계산

## 3. 사용한 경량화 기법

### (1) Knowledge Distillation

### training_distillation_total.ipynb

`Student` **모델의 설계 간소화**

<img width="286" alt="image" src="https://github.com/user-attachments/assets/7f83fb29-db87-41b0-8387-f2f388142c42" />

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

<img width="332" alt="image" src="https://github.com/user-attachments/assets/6e503a85-6127-477f-8347-25828a94da6b" />

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

<img width="551" alt="image" src="https://github.com/user-attachments/assets/a5a2b636-466d-4650-b079-7c1391004008" />

`Knowledge Distillation` **적용**:

- Teacher 모델에서 얻은 소프트 라벨(출력)을 Student 모델에 전달하여 학습 성능을 유지.
- Knowledge Distillation을 통해 Student 모델이 Teacher 모델의 지식을 효율적으로 압축 및 학습하도록 설계.

<img width="573" alt="image" src="https://github.com/user-attachments/assets/1ab5bbe3-9679-48b1-8811-c47f0c98b2ca" />

- Student Model 학습 코드

<img width="552" alt="image" src="https://github.com/user-attachments/assets/2f814dff-9c0a-446b-9b9b-533e70bf2d7d" />

<img width="353" alt="image" src="https://github.com/user-attachments/assets/26d0e238-cddb-42fc-845b-5a5c500a7878" />

- 경량화 전 후의 파라미터 비교

### (2) Pruning

### training_prune_total.ipynb

<img width="552" alt="image" src="https://github.com/user-attachments/assets/99ff7ab1-59c0-44e8-888c-10c2cd35ad1e" />

`apply_pruning` 함수:

- `torch_pruning` 사용
- `MagnitudePruner`를 사용하여 채널의 중요도가 낮은 순서대로 제거
- 마지막 분류 층(classifier)은 프루닝에서 제외
- 30%, 50%, 70% 로 Pruning Ratio를 다르게 해 경량화

<img width="552" alt="image" src="https://github.com/user-attachments/assets/be39dce8-db40-4618-a05c-ec8ed3439a53" />

- 경량화 전 후의 파라미터 비교

## 4. Jetson nano에서의 경량화 효과 (젯슨 나노-주피터 노트북에서 실행 후, html 다운로드해 화면 캡쳐 한 것

## (1) Distillation Model

- 잿슨나노의 CSI 카메라 사용 시의 추론 시간 비교
<img width="356" alt="image" src="https://github.com/user-attachments/assets/ab9c0949-99e2-46e9-8966-6e1f135bcfda" />

- 파라미터 수 비교
<img width="550" alt="image" src="https://github.com/user-attachments/assets/f723eaf8-1c51-43cd-9274-4662114aaec4" />

- 모델 사이즈 비교
<img width="551" alt="image" src="https://github.com/user-attachments/assets/2d19996b-df98-474c-b38d-75d29f660ba1" />

- FLOPs 비교
<img width="548" alt="image" src="https://github.com/user-attachments/assets/79085f47-a8e7-47f4-be57-1036922f4380" />

- 추론 시간 비교(100회 추론 비교)
<img width="362" alt="image" src="https://github.com/user-attachments/assets/2249d4db-751b-45f8-8908-a7b1822a3dca" />

<aside>
💡

Knowledge Distillation을 통해 Student 모델들의 크기를 대폭 줄이면서(Small: 93.4%, Medium: 43.4% 감소) 성능은 유지했습니다. Small Student 모델의 경우 파라미터 수가 6,210개로 크게 감소했고, 추론 시간도 Teacher 모델(72.48ms)에 비해 상당히 개선되어 20.76ms로 단축되었습니다.

실제 젯슨나노 CSI 카메라를 사용해 테스트 했을 때, Base 모델의 대부분의 파라미터를 감소시킨 Small Student Model에서 without mask를 classification하는데 성능이 약간 떨어짐을 볼 수 있었습니다. 

</aside>

## (2) Pruning Distillation 모델

### a. 30%, 50%, 70%로 진행

- 잿슨나노의 CSI 카메라 사용 시의 추론 시간 비교
<img width="294" alt="image" src="https://github.com/user-attachments/assets/0cd50756-563c-426b-aeb6-be46de190b7b" />


- 파라미터 수 비교
<img width="553" alt="image" src="https://github.com/user-attachments/assets/d3b90034-dbca-452b-b127-c00f680cc07b" />

- 모델 사이즈 비교
<img width="546" alt="image" src="https://github.com/user-attachments/assets/86498aae-7237-4ea6-a29f-39b7f607bf75" />

- FLOPs 비교
<img width="558" alt="image" src="https://github.com/user-attachments/assets/dd9e300d-0f5b-4c47-b897-e91f1a95df12" />

- 추론 시간 비교(100회 평균)
<img width="488" alt="image" src="https://github.com/user-attachments/assets/d0070c70-f2e4-4a56-9d94-c60dca8b9097" />

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
