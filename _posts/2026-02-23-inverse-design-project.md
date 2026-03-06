--- 
layout: post
title: "나노광학 입자 역설계(Inverse Design) AI 웹 서비스 구축기"
date: 2026-02-23 00:00:00 +0900 
categories: [Project, AI]
math: true
tags: [Deep Learning, PyTorch, Inverse Design, Nanophotonics, Streamlit]
---

# 1. 프로젝트 배경 및 목표
본 프로젝트는 [Nanophotonic particle simulation and inverse design using artificial neural networks](https://arxiv.org/abs/1712.03222) 논문을 기반으로, 딥러닝을 활용해 나노 입자의 구조를 역설계(Inverse Design)하는 모델을 구현했습니다.

## 다루는 데이터: Multi-shell Nanoparticle
- **구조:** 총 8개 층의 다층 구
    
- **재료:** Silica ($n \approx 1.45$)와 $TiO_2$ ($n \approx 2.7$)의 교차 적층 구조(중심은 Silica)
    
- **변수:** 각 층의 두께 8개 (30nm ~ 70nm)

## 문제 정의: Forward vs Inverse
- **Forward:** 두께를 알 때 $\rightarrow$ 스펙트럼(입사 파장 400~800nm사이 2nm씩 201개에 대한 각각의 산란 단면적 값)을 계산하는 것은 물리 공식(Mie Theory)으로 가능합니다.
    
- **Inverse:** 반대로, 특정 스펙트럼을 얻기 위한 **최적의 두께 조합**을 찾는 것은 연산 비용이 매우 높고 해석적으로 풀기 어렵습니다.

## 해결 방안
저는 이 문제를 해결하기 위해 **3가지 딥러닝 모델**을 구축했습니다.

이 모델들은 201개의 스펙트럼 데이터를 입력받아, 그에 해당하는 8개의 두께 파라미터를 실시간으로 예측합니다. 
결과적으로 기존 수치 해석 방식 대비 비약적인 속도 향상과 높은 정확도를 달성했습니다.

# 2. 기본 구조에 대한 이론
딥러닝 모델을 만들기 전에, 우리가 다루는 나노 입자가 물리적으로 어떻게 정의되는지, 그리고 우리가 예측하고자 하는 데이터가 무엇인지 명확히 할 필요가 있습니다.

## 2-1. 구조 설명: 8겹의 양파 (Multi-shell Nanoparticle)
시뮬레이션 대상은 **다층 구** 입니다. 
마치 양파같은 겹겹이 쌓여있는 구조입니다.
- **기하학 구조:** 중심(Core)에서부터 바깥쪽으로 총 **8개의 껍질(Shell)**이 감싸고 있습니다.
    
- **재료 구성:** 굴절률이 낮은 **Silica ($n \approx 1.45$)** 와 굴절률이 높은 **$TiO_2$ ($n \approx 2.7$)** 가 번갈아 가며 나타납니다. (Core는 Silica)
    
- **변수 (Input):** 각 층의 두께($d_1, d_2, ..., d_8$)는 **30nm에서 70nm** 사이의 값을 가집니다.
    
- **핵심:** 층마다 두께가 달라지면 내부에서 빛의 간섭, 반사 등 운동양상이 완전히 다릅니다.


## 2-2. 구하고자 하는 데이터: 산란 단면적 (Scattering Cross-section)
우리가 얻고자 하는 출력 데이터(Output)는 입자에 빛을 비췄을 때 나타나는 **스펙트럼**입니다.
- **입사광:** 400nm ~ 800nm 파장 대역의 빛 (가시광선 영역).
    
- **데이터 포인트:** 2nm 간격으로 총 **201개**의 파장 지점을 계산합니다.
    
- **물리량 (output):** 각 파장의 **산란 단면적($\sigma_{sca}$, $y_1, y_2, ..., y_{201}$)** 을 계산합니다.
    
    - **의미:** 입자가 해당 파장의 빛을 얼마나 강하게 튕겨내는지를 나타내는 지표입니다. 특정 파장에서 값이 급격히 커지는 **공명(Resonance)** 현상이 관찰됩니다.


### 2-3. 물리 엔진: Mie Theory & TMM
이 구조의 스펙트럼을 계산하기 위해 물리학의 정석인 **Mie Theory(미 이론)**와 **TMM(전달행렬법)** 을 사용했습니다.
#### **(1) TMM (Transfer Matrix Method)**
나노 입자의 8개 층을 통과하는 빛의 거동을 계산하는 **'솔버(Solver)'** 역할을 합니다. 각 층의 경계면에서 빛의 반사와 투과를 나타내는 **$2 \times 2$ 행렬**을 순차적으로 곱해나감으로써, 입자 전체의 **반사 계수($r_l$)**를 구해냅니다.

#### **(2) Mie Theory (미 이론)**
TMM이 구해준 반사 계수를 이용해 최종적인 **에너지(산란 단면적)** 를 계산하는 **'공식'** 입니다. 맥스웰 방정식의 구형 해를 바탕으로, 각 진동 모드($l$)별 기여도를 합산하여 전체 산란 단면적을 도출합니다.

$$\sigma_{sca} = \frac{2\pi}{k^2} \sum_{l=1}^{\infty} (2l+1) (|a_l|^2 + |b_l|^2)$$

---

> [수정된 미 이론에 대하여]
> 
> 프로젝트 진행 중, 최근 발표된 수정된 미 이론(Akimov, 2024) 논문을 검토하였습니다. 이 이론은 기존 미 이론이 입자 표면에서의 불연속성 때문에 광학적 힘(Optical Force) 계산 시 한계가 있음을 지적하고, 이를 보정하기 위해 전이 층(Transition Layer) 개념을 도입했습니다.
> 
> 하지만 본 프로젝트의 목표는 '힘'이 아닌 '원거리장 스펙트럼(Far-field Spectrum)' 예측이며, 스펙트럼 계산에 있어서는 기존 TMM 기반의 미 이론을 사용해도 충분한 정확도와 효율성을 보장한다고 판단하여 표준 방식을 채택했습니다.



# 3. 전체 프로젝트 개요 (Project Pipeline)
본 프로젝트는 단순히 AI 모델 하나를 만드는 것에 그치지 않고, **물리 시뮬레이션 데이터 생성부터 역설계까지의 전 과정** 을 수행했습니다. 전체 워크플로우는 아래와 같이 3단계로 구성됩니다.

+추가로 여러 역설계 모델을 웹사이트에 올려놔서 모두가 볼 수 있게 만들었습니다.
![전체 파이프라인](/assets/img/inverse_design_images/pipeline.png)
##  Step 1. 데이터 생성 (Data Generation)
AI를 정확히 학습시키기 위해서는 정확한 정답 데이터가 필요합니다.
맨 처음에는 comsol multiphysics툴을 공부해보는 겸 데이터 생성에 적절할 것 같아서 사용했었습니다.
하지만 5만개의 데이터가 필요한데 1개 만드는데 3분이 걸리는 걸 보고 빠르게 중단했습니다.
(15만분 $\approx$ 2500시간 $\approx$ 104일 $\approx$ 0.3년)
따라서 파이썬에서 다층 박막 계산해 주는 외부 라이브러리(scattnlay)를 사용해 계산을 진행하였습니다.
(5만개 데이터 생성 시 대략 2시간 정도 걸림)
- **도구:** Python 외부 라이브러리 scattnlay를 사용. c언어 기반 라이브러리이기 때문에 데이터 생성에 상당한 시간단축을 보임
    
- **과정:** 8개 층의 두께를 무작위로 생성하고, 이에 대응하는 스펙트럼을 계산하여 약 50,000쌍의 데이터셋(8개의 두께조합 -> 201개의 스팩트럼)을 구축했습니다.
    
- **성과:** 기존 COMSOL 대비 데이터 생성 속도를 대략 **99.88%** 단축했습니다.

## Step 2. 정방향 모델링 (Forward Modeling)
AI에게 정답을 가르치는 단계입니다.
입력으로 8개의 두께조합을 가해주면 출력으로 201개의 스펙트럼을 예측하는 모델입니다.

- **목표:** 구조(두께)를 입력하면 $\rightarrow$ 스펙트럼을 예측하는 **시뮬레이터 대체 모델** 생성.
    
- **구조:** Fully Connected Network (FCN).
    
- **의미:** 이 모델은 이후 역설계 모델의 근간이 됩니다. 모든 역설계 모델이 이 forward model에 가장 근접하게 학습을 진행하게 됩니다.

## Step 3. 역방향 모델링 (Inverse Modeling)
프로젝트의 핵심인 "원하는 스펙트럼(201) $\to$ 구조(8)"을 구현하는 단계입니다. 
논문에서는 Neural Adjoint만 소개했지만 동작 시간 문제가 있어 새로운 모델을 제시합니다.

1. **Neural Adjoint:** 별도의 역방향 네트워크 없이, forward model의 출력값(output)에 대해 역전파를 어러번 수행해 랜덤 입력값(input)을 경사하강법으로 조금씩 업데이트하여 정답 출력값을 만드는 예측 입력을 찾아내는 방식입니다.
    
2. **Tandem Network:** 정방향 모델을 정답으로 활용하여, 역방향 모델이 뱉어낸 구조가 실제 정답과 얼마나 가까운지 검증하며 학습하는 구조입니다.
    
3. **Hybrid Model:** 위 두 방식의 장점을 결합했습니다. **Tandem Network**가 빠르게 대략적인 구조를 추론하면(Initial Guess), 이를 초기값으로 삼아 **Neural Adjoint**가 정밀하게 튜닝합니다. 이를 통해 속도와 정확도를 모두 잡았습니다.

## +웹 서비스 공개
혼자 코드상으로 돌리지 말고 누구나 AI를 사용할 수 있게 웹사이트로 배포했습니다.
- **프레임워크:** **Streamlit** (Python 기반 웹 프레임워크)
    
- **기능:** 슬라이더를 이용한 직관적인 시뮬레이션. 랜덤 두께를 설정해 이에 따른 정답 스펙트럼을 만들고, 역설계 AI들이 예측 두께조합을 생성 후 그래프로 시각화(Plotly)
    
- **최적화:** 역설계 모델들이 예측 시 시간이 걸리기에 inverse model들을 경량화하여 모바일에서도 접속 가능한 웹사이트로 만들었습니다


# 4. 데이터 생성 (Data Generation)
AI 모델을 학습시키기 위해 입력(구조)과 정답(스펙트럼)이 짝지어진 양질의 데이터셋이 필수적입니다. 
논문에서 학습 데이터로 5만개의 데이터를 사용했기에 저도 **50,000개**의 데이터 샘플 갯수를 목표로 잡았습니다.
## 4-1. COMSOL Multiphysics 시도와 한계
실제로 구형 모양에 대응하는 스팩트럼 데이터를 얻기 위해 Comsol Multiphysics를 사용했습니다.
맨 처음에는 구조를 만들고, 기본상태 지정하고, 스터디를 진행했습니다.
자바 메소드를 이용해서 데이터 대량생산을 진행할 때, 자바 힙 메모리 초과로 인해 문제가 생긴 적도 있었습니다.
당시에는 당황해서 별 짓을 다했지만, 결론적으로는 exe파일까지 내려가서 자바 힙 메모리 할당을 128GB까지 늘려버렸습니다.
결론적으로 데이터 확보는 가능하지만 1개 만드는데 30분 정도의 연산시간이 걸리기 시작했습니다.

![초기 comsol 실행|554](/assets/img/inverse_design_images/comsol_play_first.png)

위 사진은 실제로 구현 화면입니다.
학습용 데이터만 5만개 필요한데 1개에 30분이면 몇년이 걸리는 상황이었습니다.
그래서 시간단축을 위해 구를 1/4로 자른 후 mesh 갯수를 1.5만개 -> 4천개 로 줄여서 시간을 단축했습니다.

![1_4round](/assets/img/inverse_design_images/1_4round.png)

실제로 위 방법을 적용한 결과 데이터 1개에 대략 3분정도로 매우 괜찮은 성능이 나왔습니다.
하지만 학습용데이터 5만개를 만들기 위해서는 아직도 대략 0.6년의 시간이 걸리기에 이 또한 정답이 될 수 없었습니다.
따라서 이번 프로젝트에 맞는 대안을 찾아내야 했습니다.

## 4-2. Python Scattnlay 라이브러리 도입

대안으로 찾은 것이 scattnlay 라이브러리입니다.
이 라이브러리는 Mie Theory(미 이론) 를 다층 박막 구에 적용하여 계산해 주는데, 핵심 코드가 **C언어**로 작성되어 있어 연산 속도가 비약적으로 빠릅니다.

- **재료 설정:**
    - **Silica:** 굴절률 약 1.45 (상수로 가정)
        
    - **TiO2:** 파장에 따라 굴절률이 변하는 분산(Dispersion) 특성을 반영하기 위해 실험 식을 적용했습니다.
        
- **구조:** 8개 층의 두께를 30nm~70nm 사이에서 랜덤하게 생성(Random Uniform)했습니다.

결과적으로 Comsol로 100일 걸릴 작업을 **단 2시간** 만에 끝낼 수 있었습니다.

### 데이터 생성 코드
실제 데이터 생성에 사용한 Python 코드입니다. scattnlay를 이용해 시뮬레이션을 돌리고 pandas로 저장하는 과정을 담고 있습니다.

``` python
import numpy as np
import pandas as pd
from scattnlay import scattnlay
from tqdm import tqdm

# 물질 굴절률 정의 함수
# TiO2는 파장에 따라 굴절률이 변하므로(Dispersion), 실험 데이터를 근사한 식을 사용했습니다.
def get_n_TiO2(wavelength_nm):
    wl_um = wavelength_nm / 1000.0
    epsilon = 5.913 + 0.2441 / (wl_um**2 - 0.0803)
    return np.sqrt(epsilon)

# Silica는 굴절률 변화가 크지 않아 상수로 가정했습니다.
def get_n_Silica():
    return 1.428

# 2. 단일 시뮬레이션 실행 함수
def run_simulation_once(shell_thicknesses):
    wavelengths = np.arange(400, 801, 2)  # 400~800nm, 2nm 간격
    spectrum = []
    
    for wl in wavelengths:
        m_list = [get_n_Silica()]
        
        # 8개 층의 굴절률 교차 적층 (TiO2 <-> Silica)
        for i in range(8):
            m_list.append(get_n_TiO2(wl) if i % 2 == 0 else get_n_Silica())
        m_arr = np.array(m_list, dtype=np.complex128)
        
        r_cumulative = np.cumsum(shell_thicknesses)
        x_arr = 2 * np.pi * r_cumulative / wl
        x_arr = np.array(x_arr, dtype=np.float64)
        terms, Qext, Qsca, Qabs, Qbk, Qpr, g, Albedo, *rest = scattnlay(x_arr, m_arr[1:])
        spectrum.append(Qsca)
        
    return spectrum

# 3. 대량 생산 함수
def generate_dataset(num_samples=5000, filename="training_data.csv"):
    data_rows = []
    print(f"{num_samples}개 데이터 생성을 시작합니다...")
    
    for i in tqdm(range(num_samples)):
        thicknesses = np.random.uniform(30, 70, 8) # 두께 조합을 전부 랜덤으로 설정
        spectrum = run_simulation_once(thicknesses)
        row = list(thicknesses) + list(spectrum)
        data_rows.append(row)
    
    col_thickness = [f"thick_{i+1}" for i in range(8)]
    col_spectrum = [f"wl_{wl}nm" for wl in range(400, 801, 2)]
    columns = col_thickness + col_spectrum
    
    # DataFrame 변환 및 CSV 저장
    df = pd.DataFrame(data_rows, columns=columns)
    df.to_csv(filename, index=False)

# 실행부
if __name__ == "__main__":
    generate_dataset(num_samples=50000, filename="dataset_50k.csv")
```


# 5. 정방향 모델링 (Forward Modeling)
Forward Model의 목표는 복잡한 수치 해석(TMM) 과정을 아주 빠른 행렬 연산(AI)으로 대체하는 것 입니다.

- **Input:** 8개 층의 두께 (30nm ~ 70nm)
    
- **Output:** 201개의 스펙트럼 포인트


## 5-1. 데이터 전처리: 정규화 (Normalization)

모델을 설계하기 전에 가장 중요한 작업이 있습니다. 바로 **데이터 정규화**입니다.

입력값인 두께(30~70)는 AI 입장에서 꽤 큰 숫자입니다. 이를 그대로 넣으면 학습이 불안정해질 수 있습니다. 따라서 학습 데이터 전체의 평균($\mu$)과 표준편차($\sigma$)를 구해, 평균 0, 표준편차 1인 분포(Standard Normal Distribution)로 변환해 주었습니다.

$$x_{norm} = \frac{x - \mu}{\sigma + \epsilon}$$

이 과정 덕분에 모델이 훨씬 빠르고 안정적으로 수렴할 수 있었습니다.

## 5-2. 모델 구조 (Architecture)

논문에서 제안한 구조를 그대로 차용하여 **완전 연결 신경망(Fully Connected Network, MLP)**을 구축했습니다.
![forward_model 파이프라인](/assets/img/inverse_design_images/forward_model_architecture.png)
- **Hidden Layers:** 4개 층
    
- **Nodes:** 각 층마다 250개 뉴런
    
- **Activation Function:** ReLU
    
- **Initialization:** 가중치는 정규분포(mean=0, std=0.1)로 초기화


``` python
import torch
import torch.nn as nn

# 완전 연결 MLP 모델 구현
class MLP(nn.Module):
    def __init__(self, input_dim=8, output_dim=201, hidden_dim=250):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            # Layer 1: 8 -> 250
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # Layer 2, 3, 4: 250 -> 250
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Output Layer: 250 -> 201
            nn.Linear(hidden_dim, output_dim)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.model(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0.1)
```

## 5-3. 학습 진행 (Training)
학습 과정에서도 논문의 하이퍼파라미터를 따랐습니다.
- **Loss Function:** MSE Loss (평균 제곱 오차)
    
- **Optimizer:** RMSprop (lr=0.0006, decay=0.99)
    
- **Epochs:** 1,000회

``` python
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, epochs=1000):
    # 논문 Hyperparameters 적용
    optimizer = optim.RMSprop(model.parameters(), lr=0.0006, alpha=0.99)
    criterion = nn.MSELoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    print("🚀 학습 시작")
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    print("✅ 학습 완료")
```

## 5-4. 성능 평가 (Evaluation)
학습이 끝난 후, 모델이 물리 법칙을 얼마나 잘 모사하는지 확인해 보았습니다.
랜덤한 두께를 입력했을 때 **AI가 예측한 스펙트럼(점선)** 과 **실제 시뮬레이션 결과(실선)** 를 겹쳐 그려보았습니다.


``` python
# 1. 랜덤한 두께 생성 (Test Input)
my_design = np.random.uniform(30, 70, 8).astype(np.float32)

# 2. 전처리: 학습 데이터의 mean/std를 이용해 정규화 필수!
input_norm = (my_design - train_dataset.mean) / (train_dataset.std + 1e-7)
input_tensor = torch.tensor(input_norm).unsqueeze(0).to(device)

# 3. AI 예측
model.eval()
with torch.no_grad():
    predicted_spectrum = model(input_tensor).cpu().numpy().flatten()

# 4. 실제 정답 (TMM 계산)
actual_spectrum = run_simulation_once(my_design)

# 5. 시각화
plt.plot(predicted_spectrum, label='AI Prediction', linestyle='--')
plt.plot(actual_spectrum, label='Real Simulation', alpha=0.7)
plt.legend()
plt.show()
```

**[결과 분석]**
![forward_model 실행예제](/assets/img/inverse_design_images/forward_test_1.png)
위 결과를 보면 알겠지만 정답 그래프와 AI의 실제 그래프가 상당히 근접한걸 알 수 있습니다.
MSE(오차) 값은 거의 0에 수렴했으며, 이는 **AI가 0.001초 만에 복잡한 물리 시뮬레이션을 완벽하게 대체할 수 있음**을 의미합니다.

## 5-5. 코너 케이스(극값) 문제
위 forward model을 가지고 여러가지 실험을 해본 결과, 데이터가 극값에 위치한 경우에는 정답 그래프를 잘 따라가지 못하는 경향을 발겼했습니다.
아래는 8층의 두께를 전부 30~35nm 부근에서 랜덤으로 생성을 했을 때 forward model의 예측 그래프 입니다.
![극값문제](/assets/img/inverse_design_images/극값문제_forward.png)
두께조합은 아래와 같습니다.
[32.652996 33.33089 32.076088 31.146091 34.279697 33.66407 32.976448 32.60075]

위 그림같이 극값 주변에서 AI가 잘 예측하지 못하는 이유는 데이터 생성을 완전랜덤으로 만들었기 때문입니다.
따라서 에초에 데이터를 생성할 때 극값인 데이터를 추가해서 모델을 재학습 시키면 아래와 같은 결과가 나옵니다.
![극값문제해결](/assets/img/inverse_design_images/극값문제해결_forward.png)
두께조합은 아래와 같습니다.
[31.466333 32.997612 31.565607 34.375065 31.857618 33.02835 34.224354 30.780802]

실제로 이레귤러 데이터를 학습 데이터에 추가하니 상당히 잘 예측하는 경향을 볼 수 있습니다.

이제 위의 Forward Model을 정답으로 삼아, 본격적인 역설계(Inverse Design)을 진행할 수 있습니다.


# 6. 역방향 모델링 (Inverse Modeling)
Forward Model이 "두께 $\to$ 스펙트럼"을 예측했다면, 역설계 모델은 "원하는 스펙트럼(Target)을 주면, 그걸 만드는 두께 조합을 예측하는 것" 이 목표입니다.
이 과정에서 겪은 시행착오와 최종적으로 도달한 해결책을 소개합니다.
## 6-1. Neural Adjoint
맨 처음 방식은 논문에서 소개한 **Neural Adjoint** 방법입니다.
이 방식의 특징은 새로운 AI 모델을 만드는게 아닌, 원래 forward model의 역전파를 입력단까지 진행해서 랜덤 두께조합을 경사하강법으로 정답 스펙트럼에 맞는 두께조합으로 바꿔나가는 기법입니다.
![Adjoint](/assets/img/inverse_design_images/Neural_adjoint.png)
### 1) 작동 원리 (Gradient Descent on Input)
일반적인 역전파는 모델 속 가중치를 업데이트하지만, 여기서는 가중치를 고정하고 입력값(Input, 두께)을 업데이트합니다.
1. 랜덤한 두께조합($x$)을 Forward Model에 넣고 스펙트럼($y_{pred}$)을 뽑습니다.
    
2. 목표 스펙트럼($y_{target}$)과의 오차(Loss)를 구합니다.
    
3. 역전파(Backpropagation)를 통해 **오차를 줄이는 방향으로 입력값($x$)을 조금씩 수정**합니다.
    
4. 이 과정을 반복하면 입력값 $x$가 점점 정답에 가까워집니다.

``` python
# Neural Adjoint 핵심 로직 (Pseudo-code)
inp = torch.randn(1, 8, requires_grad=True) # 입력값에 미분 가능 설정
optimizer = optim.Adam([inp], lr=0.01)      # 가중치 대신 입력을 최적화

for i in range(1000):
    pred = forward_model(inp)
    loss = mse_loss(pred, target)
    loss.backward()
    optimizer.step()
```

### 2) 문제 발생: 로컬 미니멈 (Local Minimum)
하지만 이는 심각한 문제가 하나 있습니다.
시작 두께조합 x를 랜덤으로 지정하고 위 method를 진행하다 보니, loss가 작은 지점인 로컬 미니멈(Local Minimum)에 빠지는 문제가 발생합니다.
이는 경사하강법의 고질적인 문제이며, 논문에서는 이를 해결할 해결책이 제시되어 있었습니다.
![minimum_problum](/assets/img/inverse_design_images/global_local_mimnimum.png)

아래는 로컬미니멈에 빠진 예측 그래프입니다.
![errer_adjoint](/assets/img/inverse_design_images/errer_adjoint.png)
정답 두께: [34.3 67.7 64.5 31.4 30.7 43.5 62. 58.4] 
예측 두께: [49.1 53.7 69.9 57.6 70. 70. 43.8 70. ]
실제 Loss값은 0.274704 으로 더이상 유의미하게 줄어들지 않지만 그래프가 전혀 정답에 근접하지 않는걸 볼 수 있습니다.

### 3) 해결책: 동시실행
위와 같이 로컬 미니멈 문제를 해결하기 위해 무작위 x 지점을 여러가지로 시작 하였습니다.
한 번에 1개의 입력값만 최적화 하는게 아니라, 여러개의 서로 다른 랜덤 입력값을 동시에 던져서 최적화를 진행해 loss가 가장 작은 x를 정답으로 사용했습니다.
실제 논문에서는 50개를 동시에 실행하였지만 저는 128개로 동시 진행하였습니다.
[동시시작 시연](/assets/video/inverse_design_videos/구슬_생성_위치_무작위_변경.gif)


아래는 128개를 동시에 실행해 총 step을 2000번으로 실행한 결과입니다.
![start128_adjoint](/assets/img/inverse_design_images/start128_adjoint.png)
Step [500/2000], Avg Loss: 0.095071 
Step [1000/2000], Avg Loss: 0.094677 
Step [1500/2000], Avg Loss: 0.093489 
Step [2000/2000], Avg Loss: 0.093265
Best Candidate Index: 11, Loss: 0.008570
정답 두께: [30.9 49.1 39.4 35.3 68.1 37.8 60.4 44.9] 
Best AI 예측두께: [42.3 40.5 38.4 34.8 67.8 36.5 60.6 42.5]

추세를 상당히 잘 맞추는걸 볼 수 있습니다.

### 4) 또 다른 문제: 시간
정확도는 step 수를 늘리거나 시작 갯수를 늘리는 것으로 잡을 수 있었습니다.
하지만 문제는 수행하는데 걸리는 시간 이었습니다.
실제로 마지막에는 웹사이트에 공개해서 누구나 사용하게 만들건데 이 모델만 혼자 5~10초동안 시간이 걸리게 할 수 없었습니다.
따라서 이 문제에 대해 찾아보다가 에초에 다른 역설계 모델에 대해 알게 되었습니다.

## 6-2. Tandem Network
속도 문제를 해결하기 위해 도입한 것이 Tandem Network입니다.
이는 맨 처음 소개한 forward model처럼 만들되, 입력을 201개의 스펙트럼으로, 출력을 8개의 두께 조합으로 만드는 기법입니다.
여기서부터는 논문에 작성되있지 않은 미개척지역으로 혼자 힘으로 진행할 수 밖에 없었습니다.
### 1) 전체 학습구조 (Inverse Net + Forward Net)
forward model처럼 단순하게 Input(201) -> Output(8)로 학습하면 일대다(One-to-Many) 문제 때문에 학습이 잘 안 됩니다.
마치 $y = x^2$의 역함수를 구하라 할 때 정답이 4라면 AI는 x값을 2라고 해야할지, -2라고 해야할지 문제에 빠지는 것과 같습니다.
따라서 기존 forward model의 MLP보다 모델을 더 복잡하게, step scheduler, batch norm, drop out 등을 추가 사용하여 최대한 정확도를 높였습니다.
실제 모델은 스펙트럼을 입력으로 받으면 예측 두께조합 x를 내뱉는데, 만약 예측 x와 정답 x를MSE loss로 사용하면 하나의 답만 정답 처리될 수 있습니다.
예를 들어$y=x^2$에서 입력 4에 대해 모델이 2를 예측하고 정답이 -2라면 MSE를 사용 시 2를 반드시 틀렸다고 하기에 문제가 있습니다.

따라서 예측 x를 forward model에 넣은 후에 나온 예측 스펙트럼과 입력으로 들어간 정답 스펙트럼간의 MSE를 loss로 활용해 모델을 학습시켰습니다.
![텐덤 모델](/assets/img/inverse_design_images/tandem_model.png)

아래는 실제 Tandem network로 예측한 그래프입니다.
![tandem_test](/assets/img/inverse_design_images/tandem_test.png)
AI가 찾은 두께: [47.7 38.6 42.2 54.3 47.2 44.7 43.6 65.2] 
실제 두께 답: [48.6 40.5 42.4 54.5 46.7 45.5 43.6 64.6]

### 2) 속도적 우위
이 방식은 MLP 모델을 직접 만들어 간단한 행렬연산만 몇번 진행하면 되기 때문에 위의 adjoint method보다 속도가 획기적으로 빠릅니다.

### 3) 정확도의 한계
하지만 Tandem 모델은 정확도에서 문제가 발생합니다.
- $y=x^2$인 문제에서 y=4에 대한 정답 x를 찾는 과정에서 무작위 x에 대해 loss를 최소화하기 위해 $loss = (x-2)^2 + (x+2)^2$에서 loss가 가장 작게 되는 x=0을 정답으로 예측할 수 밖에 없습니다.
- adjoint 방식은 현 지점에서 loss가 가장 최소로 가는 방식을 개별맞춤으로 최적화 하지만 tandem 모델은 추세를 외운거기 때문에 정확도에서 차이가 날 수 밖에 없습니다.
결론적으로 속도는 빠르지만 adjoint만큼 정밀하지는 못합니다.

여기에서 Tandem model과 adjoint method 각각의 특징에 대해 속도도 빠르고 정확도도 빠른 모델에 대해 고민하고 찾아본 결과 어떠한 아이디어가 떠올랐습니다.

## 6-3. 최종 해결책: 하이브리드(Hybrid) 모델
Tandem의 문제점은 adjoint보다 정확도가 낮기 때문입니다.
Adjoint method의 문제점은 시작 지점이 완전한 랜덤이기 때문에 local minimum에 빠지기 때문입니다.
그렇다면 Tandem 모델로 초기 x값을 글로벙 미니멈에 가까이 배치한 다음 adjoint method를 사용한다면?
-> local에 빠지지 않고 global로 바로 수럼 할 것이다!!
이는 hybrid model의 시작 아이디어가 되었습니다.

### hybrid model의 과정
1. **Initial x Guess (Tandem):** 스펙트럼을 받으면 Tandem Network가 0.001초 만에 대략적인 두께 조합($x_{init}$)을 추론합니다.
    
2. **Fine-tuning (Adjoint):** 이 $x_{init}$을 Neural Adjoint의 시작점으로 설정합니다. 이후 상당한 step을 가해줘 $x$가 가장 최소의 loss를 가지는 값으로 수렴하게 합니다.
    
3. **Result:** 이미 정답 근처에서 시작했기 때문에, 로컬 미니멈에 빠질 위험도 없고, 몇 번의 반복(Iteration)만으로도 완벽한 정답에 도달합니다.

### 결과
실제로 시간도 adjoint method에 비해 획기적으로 줄었고, 정확도도 tandem model에 비해 정확한 걸 볼 수 있었습니다.
이래는 hybrid model로 예측한 그래프입니다.


![하이브리드 모델 예측](/assets/img/inverse_design_images/hybrid_1.png)
정답 두께 (nm): [39.81 50.95 56.56 60.17 56.75 46.42 43.12 67.19] 
예측 두께 (nm): [44.29 49.82 54.44 60.68 55.98 47.62 42.29 67.57]
최종 Loss: 0.00495


# 7. 웹 서비스 구현
단순히 AI 모델을 만들어서 로컬 환경에서만 돌리기보다는 실제 웹사이트상에 올려 구동 가능한 모습을 보이는게 좋겠다 싶어서 온라인상에 올려두었습니다.
아래는 URL과 QR코드 입니다.
https://viewinversedesignproject-lv8ns7c3pdqwjbeggga7mu.streamlit.app/

![QR](/assets/img/inverse_design_images/QR_code.png)

사이트에 들어가면 아래와 같은 창이 나옵니다.
![웹사이트 이미지](/assets/img/inverse_design_images/웹사이트_이미지_1.png)
위 사이트는 제가 만든 3가지 역설계 모델들을 비교할 수 있게 만들어 두었습니다.
본래는 스펙트럼 201개의 지점을 주면 그에 따른 두께조합을 예측해야 하지만, 201개를 지정하기엔 기술적 문제가 있어서 두께조합을 준다면 '주어진 두께 -> 정답 스팩트럼 201개' 로 변환해 입력을 진행합니다.
3가지 inverse model은 주어진 스펙트럼을 가지고 각자의 예측 두께 조합을 테이블에 작성합니다.
그래프는 각각이 예측한 두께를 가지고 스펙트럼을 그린 모습입니다.
웹사이트상 빠른 동작을 위해 adjnoint는 초기 시작 갯수랑 반복횟수를 상당히 줄여 MSE 성능이 좀 낮은걸 볼 수 있습니다.


# 8. 결론 및 마무리
이번 **'나노광학 입자 역설계(Inverse Design)AI 구축'** 프로젝트는 단순히 논문을 따라서 구현하는것을 넘어 추가적인 문제점을 해결하고, 실제로 웹 사이트상에 올리는 것 까지 진행하였습니다.
프로젝트를 진행하며 얻은 성과와 지식은 다음과 같습니다.

### 1. COMSOL Multiphysics 사용
실제 프로젝트를 맡아 인터넷 검색, LLM 사용으로 실제 사용 가능한 시뮬레이션 데이터를 생성해 내였습니다. 
8층 박막 구를 구현해서 실제로 시뮬레이션을 돌려보고, 문제점을 찾기도 하였습니다.
실제로 시간이 너무 오래 걸리는 문제를 해결하기 위해 구 모형 물체를 1/4로 쪼개고, 메쉬 수를 1/10으로 감소시켜 계산시간을 대폭 줄였고, 데이터 생성 자동화를 위해 java method를 사용하였습니다.
비록 comsol에서 만들어낸 데이터를 사용하지는 않았지만, 이번 프로젝트를 통해 위 프로그램 사용법을 익혔습니다.

### 2. 물리 법칙 학습한 AI (Forward Modeling)
복잡한 수치 해석(TMM) 과정을 완전 연결 신경망(MLP)의 행렬 연산으로 대체했습니다.
정규화(Normalization)를 거친 이 모델은 실제 물리 시뮬레이션 결과와 오차(MSE)가 거의 0에 수렴할 정도로 정확하면서도, 연산 속도를 **0.001초** 수준으로 비약적으로 끌어올렸습니다.
AI가 그저 학습 데이터를 외운게 아닌, 데이터 상의 어떠한 상관관계를 익히고 이는 실제로 정답에 근사한 것을 볼 수 있었습니다.

### 3. 여러 문제점 해결 
실제로 위 프로젝트를 진행하면서 수많은 문제를 겪었습니다.
comsol로 만드는 데이터도 너무 시간이 오래걸렸고, AI도 어디에 어떻게 구현할지도 문제였습니다.
또한 데이터가 너무 50nm 근처에만 생성되어 있어 극값 데이터는 잘 학습하지 못하거나, AI모델의 파라미터 문제로 정확한 답과 상당히 멀어지는 등 여러 문제가 많았습니다.
하지만 문제점들을 전부 해결하며 실제 연구자들이 현대 문명 끝부분에서 무엇을 연구하는지, 아무것도 없는 곳에서 어떻게 나아가는지 등 수많은 고찰점과 본받을 점을 배웠습니다.

### 4. 속도와 정확도의 딜레마 해결 모델
본 프로젝트의 가장 큰 기술적 난관이자 성과입니다.
- **Neural Adjoint:** 정확하지만 로컬 미니멈(Local Minimum)에 빠지기 쉽고, 이를 128개 동시 탐색으로 해결하려니 3~5초의 시간 딜레이가 발생했습니다.
    
- **Tandem Network:** 매우 빠르게 결과를 도출하지만, 물리적 일대다(One-to-Many) 문제로 인해 디테일한 정확도가 떨어졌습니다.

이를 해결하기 위해 "Tandem 모델로 0.001초 만에 최적의 초기값(Warm Start)을 찾고, Neural Adjoint로 단 몇 번만 미세 조정(Fine-tuning)을 거치는 Hybrid Model"을 독자적으로 구축했습니다. 
그 결과, 실시간(Real-time) 응답 속도와 완벽에 가까운 물리적 정확도라는 두 마리 토끼를 모두 잡을 수 있었습니다.

+본 프로젝트의 전체 PyTorch 소스 코드와 데이터셋 생성 스크립트는 제 깃허브 리포지토리 에서 확인하실 수 있습니다.
https://github.com/pizza119/inverse_design_project_code

### 마치며...
이제 AI는 언어모델 같은 단순한 도구로만 사용되는게 아니라, 각자의 영역에서 엄청난 발전을 가져올 수 있는 매우 강력한 무기가 된 것 같습니다.
실제로 위 프로젝트는 원하는 파장대에 빛을 내는 두께조합을 찾기 위해 과거에는 모든 조합을 시험해 봐야했지만, 현재는 훌륭한 데이터만 존재한다면 AI가 매우 빠르고 정교하게 답을 찾아줍니다.
앞으로는 AI가 다른 물리학 부분에서 어떤식으로 사용될지, 어떻게 사용할 수 있을지 고민해 나가며 계속 실험해볼 생각입니다.
감사합니다.