# 비정렬 삼각형 P1 FEM 기반 Neural AMG

![Actual generated P1 triangular meshes](docs/figures/p1_unstructured_mesh.svg)

현재 **`neural-amg` 브랜치는 FEM P1 비정렬 삼각형 이산화만 사용한다.**
Q1, 정렬 삼각형 조립, 이전 geometric/CNN 실행 경로는 이 브랜치에서 제거했다.
이전 연구 코드는 [`residual-u-net-teacher`](https://github.com/slwjackie/adaptive-multigrid/tree/residual-u-net-teacher)와 Git 이력에 남아 있다.

[실행 문서](docs/neural_amg.md) · [P1 수식·검증](docs/p1_fem.md) · [영문 README](README.md)

## 실제 계산 흐름

1. Seed에 따라 내부 점 위치를 교란하고 Delaunay 삼각분할로 비정렬 mesh를 만든다.
   삼각형은 실제 좌표와 연결 정보로 저장하며, 정사각형을 고정 대각선으로 자르지 않는다.
2. 각 삼각형의 선형 형상함수 기울기를 계산해 P1 FEM 강성행렬을 조립한다.
3. 연구 데이터는 물리적 소스 `f=1`을 적분해 하중벡터를 만들고 `u=0` Dirichlet 경계를 제거한다.
4. Classical AMG가 각 level의 실제 A에서 coarse 노드 C와 interpolation support S를 정한다.
5. BAMG+EM teacher로 graph attention 모델을 지도학습하고, 실제 MG 잔차로 fine-tuning한다.
6. 생성한 P로 `R=Pᵀ`, `Ac=PᵀAP`를 구성한다. NN 추론은 setup에서만 수행한다.

| 항목 | 설정 |
| --- | --- |
| 이산화 | 비정렬 삼각형 P1 FEM만 허용 |
| 공간 적분 | 양의 가중치를 가진 7점 삼각형 적분, 5차 다항식까지 정확 |
| Attention | A 그래프 3층 + sparse fine–coarse cross-attention + signed edge head |
| P 제약 | Coarse injection, 기준 P 행합, signed row-L1 cap |
| Smoother | Weighted Jacobi pre/post 각 2회 |
| 입력 특징 | `original` 기본, `row9`·`symmetric9` 선택 실험 |
| 계산 환경 | CPU FP64 |
| 연구 데이터 | TRAIN 560 / validation 280 / final 280 + 6종 OOD |
| Mesh OOD | 별도 `ood_mesh`: jitter와 공간 grading 변화 |

격자 크기 `n`은 초기 점 밀도를 정하며 기본 내부 미지수는 `n²`개다.
점의 실제 좌표와 Delaunay 연결은 mesh seed에 따라 달라진다. Solver와 NN은
행렬을 정렬 격자 이미지로 reshape하지 않는다. `ood_topology`는 계수 분포 형상의
변화이며, mesh의 변화는 `ood_mesh`로 따로 평가한다.

## 실행과 재개

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/run_neural_amg.py train \
  --config configs/neural_amg_smoke.json --run-dir artifacts/neural_amg_p1_smoke
python scripts/run_neural_amg.py freeze \
  --config configs/neural_amg_smoke.json --run-dir artifacts/neural_amg_p1_smoke
python scripts/run_neural_amg.py evaluate \
  --config configs/neural_amg_smoke.json --run-dir artifacts/neural_amg_p1_smoke
```

**새 출력 폴더를 사용한다.** 기존 Q1 설정·label·checkpoint·run의 resume는
새 `neural-amg-p1-study-v2`에서 거부한다. 임의의 A/b NPZ 입력 경로도 제거했다.
모든 연구 행렬은 저장된 삼각형 mesh에서 P1으로 조립되며, 좌표·연결·경계·내부 DOF
배열과 해시를 함께 보존한다. Final/OOD mesh는 freeze 후 생성한다.

전체 연구·재개 명령은 [실행 문서](docs/neural_amg.md)에 있다. 단위 테스트와 smoke는
실행·수치 검증이며, 560개 연구 학습 또는 classical AMG 대비 속도 우위의 증거는 아니다.
