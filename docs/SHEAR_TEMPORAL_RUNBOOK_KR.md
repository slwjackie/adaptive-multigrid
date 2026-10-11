# 변조형 He–N₂ 혼합층과 시간 이력 기반 Neural Multigrid

주제: **변조형 He–N₂ 전단 혼합층을 주 실험으로 하고, 수소 반응유동까지
검증하는 시간 이력 기반 Neural Multigrid**.

이 브랜치는 실행 가능한 실험 경로를 제공한다. 새 혼합층의 OpenFOAM 빌드,
유동 타당성, 실제 속도 향상은 대상 WSL에서 확인해야 한다. Python 수치 테스트와
독립 C++ 수식/decoder 테스트를 CFD 실행 결과로 해석하지 않는다.

## 1. 무엇을 구현했나

| 항목 | 구현 |
|---|---|
| 주 실험 | 아래쪽 He-rich 고속 유동, 위쪽 N₂ 저속 유동의 평면 coflow 전단 혼합층 |
| 시간 변화 | 두 개의 부드러운 sin² pulse envelope와 두 시간 주파수로 속도 차이를 변조 |
| 대조군 | 같은 geometry·초기장·평균 유입 조건에서 변조만 끈 전단 혼합층 |
| 일반화 | 학습에 없던 주파수·pulse 시점·유입 조성과 조성 변조 |
| 추가 적용 | 별도 H₂/air 반응유동 학습·검증; He 모델의 수소 zero-shot 전이를 주장하지 않음 |
| 알고리즘 | P 고정, 현재 A에 대한 Galerkin coarse 행렬·수치 인수분해 갱신, NN stencil만 재사용/갱신 |
| 안전 경로 | 현재 A의 실제 잔차 검증, 거부된 trial 취소, 같은 P의 classical 복구, 실패 cache 폐기 |
| 압력 비용 | 서비스가 없는 native PCG/GAMG 진단과 서비스가 포함된 coupled 실험을 각각 측정 |

지배 방정식은 OpenFOAM Foundation 13 `multicomponentFluid`의 질량·운동량·종·
에너지 방정식과 압력 보정이다. He/N₂에서는 반응을 끄고 상수 물성 및
`unityLewisFourier`를 사용한다. 계산은 저마하 조건이지만 별도로 유도한
엄밀한 incompressible/low-Mach projection solver로 바꾼 것은 아니다.
압력 연산자 변화는 수집한 실제 A로 측정한다. 밀도비만으로 변화량이나 병목을
단정하지 않는다.

기본 격자는 31×31×1, 영역은 0.04×0.02 m, 전단 두께 매개변수는 0.002 m다.
유입의 평균 속도를 유지하며 속도 차이를 변조한다. 초기 교란은 경계에서
사라지는 streamfunction으로 주어 연속식 수준에서 divergence-free다.
유입/유출과 slip 측면 경계, empty 앞뒤 경계를 사용한다. cyclic/AMR/MPI는
이 어댑터의 지원 범위가 아니다.

기존 기본 He/N₂ counterflow는 `run_v6_7_variable_density_pipeline.py`에
그대로 남아 있다. 새 비변조 shear control과 서로 다른 대조군이다.

## 2. WSL에서 첫 실행

기존 작업 폴더에서 새 브랜치를 선택한다. 이 작업은 merge를 하지 않는다.
로컬 수정 때문에 switch가 거부되면 변경을 보존한 뒤 다시 시도한다.

```bash
cd /home/jackie/adaptive-multigrid-h2
git fetch origin
git switch --track origin/research/modulated-shear-temporal
source /opt/openfoam13/etc/bashrc
source .venv/bin/activate
python -m pip install -e '.[dev]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash integrations/openfoam13/Allwmake
python integrations/openfoam13/tests/check_wire.py --ipc
python -u scripts/run_v6_7_shear_pipeline.py \
  --output artifacts/shear_probe_v1 --stop-after probe
```

이미 로컬 브랜치가 있으면 `git switch research/modulated-shear-temporal`을 사용한다.
첫 실행은 실제 CFD 수집, H_S 학습, train/tune counterfactual 비용 측정까지다.
World Model 학습이나 validation/test 성능 평가를 아직 하지 않는다.
기존 결과를 덮어쓰지 않으므로 새 출력 폴더를 지정해야 한다.
소스·설정이 바뀌면 예전 checkpoint/실험 폴더를 새 연구에 재사용하지 않는다.

기본 설정은 CFD 400 timestep, timestep당 압력 풀이 2회, 총 7개 궤적이다.
실제 연속된 압력 행렬을 기록한다. toy elliptic 행렬로 대체하는 경로는 없다.
코드가 들어 있는 `codedFixedValue` 유입 경계는 대상 머신에서 컴파일된다.
성공한 수집 단계의 dynamicCode를 각 solver arm에 복사해 최초 경계 컴파일을
한쪽에만 부과하지 않는다. 처음 수집에 걸린 시간도 solver 속도 향상으로 세지 않는다.

## 3. 단계와 결과 파일

| `--stop-after` | 끝나는 지점 | 핵심 파일 |
|---|---|---|
| `collect` | CFD 수집·데이터 계약 검사·행렬 변화 측정 | `dataset_profile.json`, `recordings/*/physics_profile.json` |
| `probe` | H_S 학습·두 행동의 즉시/누적 비용 측정 | `study/reuse/collection.json`, `study/reuse/episodes/` |
| `validation` (기본) | 정책 학습·선택, replay, native 비용, 실제 coupled 비교 | `study/reuse/validation/report.json`, `native_*/report.json`, `coupled/*/*/report.json` |
| `test` | validation 통과 후 두 freeze → 독립 test | `study/freeze.json`, `study/reuse/freeze.json`, `study/reuse/test/report.json` |

모든 경로에 `pipeline_status.json`, `pipeline_plan.json`, 단계별 `logs/`를 남긴다.
완료된 단계의 상세 결과는 `summary.json`에서 확인한다. 실패 원인과 부분 결과를
보존하고 물리/수치 검증 실패 시 test로 진행하지 않는다. 유효한 속도 저하는
삭제하거나 성공적인 반복만 골라 평균내지 않는다.

전체 validation과 수소 검증을 새 실험으로 실행하려면:

```bash
python -m pip install cantera
python -u scripts/run_v6_7_shear_pipeline.py \
  --output artifacts/shear_h2_validation_v1 --stop-after validation --with-h2
```

수소는 `hydrogen_validation/` 아래 별도 config·데이터·expert·정책·보고서를
사용한다. 반응에 따른 양의 열방출과 T/H₂/U/p의 coupled 일치 검사를 수행한다.
H₂ 사례가 실패하면 전체 완료 표시도 실패로 남기며 허용치를 자동으로 늘리지 않는다.
이 검사는 실험 화염 데이터와의 검증이나 Soret/차등확산 모델 검증을 대신하지 않는다.

validation을 확인한 뒤, 사전에 정한 설계로 최종 독립 평가를 실행한다:

```bash
python -u scripts/run_v6_7_shear_pipeline.py \
  --output artifacts/shear_h2_final_v1 --stop-after test --with-h2
```

위 명령은 새 폴더에서 수집부터 다시 실행한다. 전체 runner의 중간 재개 기능은
없다. 개별 `run_v6_7_smoother_reuse.py`의 collect/train/evaluate/freeze 명령은
있지만 freeze 순서와 coupled 검증도 필요한 최종 실험에는 전체 runner를 사용한다.

## 4. NN 재사용과 World Model

기존 `REUSE_HS`는 P 재사용이었고 A가 바뀌면 NN stencil을 다시 생성했다.
새 `smoother_reuse` 경로는 실제로 이전 NN stencil 계수를 보존한다.
원래 대각 스케일도 포함해서 보존하므로 오래된 stencil이 악화될 가능성을
측정할 수 있다. coarse A·block/line factor·최종 잔차는 항상 현재 A를 사용한다.
다단 residual cascade의 내부 residual matvec도 현재 A로 다시 연결한다.

비교 정책은 다음과 같다. 간격의 단위는 CFD timestep이 아니라 **압력 호출**이다.

- Classical MG: 미리 선언한 classical plan, 동일한 초기 P.
- `refresh`: 매 압력 호출마다 NN stencil 재생성.
- `reuse`: 실패/복구로 cache가 폐기될 때까지 재사용.
- `periodic_8`, `periodic_32`: 일정 호출 수마다 갱신.
- `drift_0.001`, `drift_0.01`: 생성 당시 A 대비 정규화 변화와 전체 스케일 변화.
- `residual_0.001`: 행렬 변화와 직전 수렴 악화 정보를 함께 사용.
- `world`: 행동별 비용·다음 solver 관측을 예측하는 작은 GRU, 예측상 두 호출을 계획.
- `world_horizon1`: 같은 GRU로 당장의 비용만 비교.
- `world_stateless`: 매번 hidden state를 지운 같은 GRU의 ablation.
- `memoryless`: 별도로 학습한 작은 MLP, 당장의 비용 비교.

모든 비학습 정책을 tune 데이터의 실제 전체 solver 비용으로 비교해 `selected`를
정한다. 학습 정책은 불확실하거나 충분한 예상 이득이 없으면 이 기준으로 되돌아간다.
Replay에서는 모든 정책을 비교한다. 실제 coupled CFD는 classical을 짝으로
`refresh`, `selected`, 그리고 학습된 정책/ablation들을 실행한다. validation에서
가장 빨랐던 기준으로 몰래 교체하지 않는다.

World Model이 예측하는 대상은 **solver 비용과 다음 solver 관측**이다.
유동장이나 화염을 대체하는 모델이 아니다. 현재 A와 RHS/초기 잔차, cached A,
cache 나이, 직전 반복 횟수·수렴률·시간·복구 여부를 관측한다. 절대 시각,
case ID, forcing phase/schedule, 다음 압력 행렬은 정책 입력으로 주지 않는다.
MLP도 동일한 현재 관측을 쓰므로 일부 직전 풀이 요약을 볼 수 있다. 이 비교는
추가적인 recurrent hidden history의 가치를 확인한다.

학습 gate는 현재 snapshot에서 refresh/reuse를 각각 실제 실행하고, 이후 최대
4회 압력 호출까지 guarded reuse를 계속하는 두 branch의 누적 비용을 비교한다.
각 branch는 기본 3회 반복한다. 반복 비용 구간이 5% 이상 분리되는 refresh 우세와
reuse 우세가 각각 3개 이상이고 실패가 없어야 모델 학습을 시작한다.
즉시 갱신 비용이 크더라도 나중에 회수하는 경우를 놓치지 않기 위한 probe다.
이 lookahead는 train/tune 라벨 생성에만 사용한다. 겹치는 window는 독립 표본이
아니며, 이 gate 자체가 통계적 유의성·최적 정책의 상한·WM 이득을 증명하지 않는다.
4회보다 긴 회수 기간은 이 probe가 놓칠 수 있다.

Gate가 닫히면 휴리스틱 비교를 계속하고 WM은 `skipped...`로 남긴다.
“World Model이 원천적으로 쓸모없다”는 결론이 아니다. 설정과 cache age 범위에서
측정된 상충관계가 부족하다는 뜻이다. source/config/설계를 바꾸려면 새 run을 쓴다.

GRU/MLP는 train으로만 fitting하고 오차 margin·기준 정책은 tune으로 정한다.
validation에서는 수정 판단을 하고 test 전에는 expert와 temporal artifact를
별도로 freeze한다. 기본 split은 train 2 / tune 1 / validation 2 / test 2개이며,
같은 초기장을 가진 forced/control은 같은 validation 그룹에만 둔다.
수집·provenance 검사를 위해 test CFD도 미리 생성하지만 fitting/정책 선택에는
test 배열이나 test 성능을 쓰지 않는다.

현재 크기 지원 검사는 학습한 cell count 범위에 한정된다. 다른 관측값의 OOD에
대해 인증된 불확실성 보장은 없다. 현재 잔차를 이용한 복구와 held-out 성능
검증이 필요하며, tune margin도 경험적 장치다.

## 5. 어떤 숫자로 연구의 가치를 판단하나

`coupled/*/*/report.json`의 `pairs[].timing`에서 setup, 실제 solve, policy,
service, admission, bridge startup, pressure, 전체 wall 시간을 확인한다.
`smoother_refreshes/reuses`는 실제 cache 사용 횟수다. service와 MG total처럼
포함 관계인 시간을 더해서 총시간을 만들면 중복 집계가 된다.

`native_*/report.json`의 pressure fraction에는 Python IPC/JSON이 없다.
raw residual 검사와 native solver 호출에 든 시간을 포함한다. 설정된 PCG/GAMG는
동일 raw L2 기준을 보수적인 normalized L1 기준으로 변환하므로 과수렴할 수
있다. native 최적 tuning 또는 “보통 수소연소의 압력 비중”을 대표하지 않는다.
Native field sanity는 C/H_S 물리장의 paired equivalence와 별개다.

WM의 추가 이득은 전체 validation/test 궤적에서 좋은 주기/규칙 정책 및 MLP보다
낮은 **실측 총비용**으로 판단한다. replay에는 생성자·특징·예측·setup·solve·검증·
복구 비용을, coupled에는 서비스 시작/전송을 포함한 전체 비용을 사용한다.
학습 비용은 별도로 보고한다. 다음 순서로 확인한다:

1. 모든 방법이 동일 잔차 기준을 만족하고 coupled 물리장 허용치를 통과하는가.
2. 기본 He/N₂와 달리 실제 압력 행렬/반복 비용이 변하고 행동 간 상충관계가 있는가.
3. `world`가 tune-selected 규칙과 별도 MLP보다 빠른가.
4. history/horizon ablation에서 시간 이력과 계획의 기여를 구분할 수 있는가.
5. native 압력 비중과 전체 CFD wall에서 가속할 여지가 충분한가.
6. 수소에 같은 절차를 적용해도 물리장 검증을 통과하고 이득이 남는가.

## 6. 격자·시간과 물리적 범위

기본 0.04 s는 기능/비용 조사용이다. 평균 속도 기준 약 2.5회의 영역 통과에
해당하며 충분히 발달한 난류를 뜻하지 않는다. 장시간 성장, 격자/시간 수렴과
3D 검증은 후속 단계다. 기본 CFL 사전 검사는 유입/초기 추정이므로 계산 중
실제 Courant number와 역류도 로그에서 확인한다.

| 용도 | 예시 추가 인자 |
|---|---|
| 기본 작동 확인 | `--n 31 --end-time 0.04 --delta-t 1e-4` |
| 더 긴 63×63 조사 | `--n 63 --end-time 0.08 --delta-t 1e-4` |
| 127×127 조사 | `--n 127 --end-time 0.08 --delta-t 5e-5` |

이 표는 안정성/수렴이 검증된 권장 해상도라는 뜻이 아니다. 각 격자는 별도 run에서
새로 학습·검증한다. 격자/시간 수렴을 비교할 때 물리적 forcing schedule과 end-time을
같게 유지한다. end-time을 바꾸면 정규화된 pulse 위치도 바뀐다.
각 위치의 조성 폭, vorticity thickness, He 분산, enstrophy는 실제 저장된
장에서 계산하지만, 이것만으로 Kelvin–Helmholtz 성장률·DNS 정확도를 검증하지 않는다.

유학 포트폴리오에서는 현재 결과를 **가변밀도 혼합유동의 수치 해석과 학습 기반
압력 solver 제어**로 제시한다. 수소 검증을 통과한 뒤 계산 연소 적용을 추가하고,
수소터빈 자체의 설계나 화염 안정성을 풀었다고 설명하지 않는다.

## 7. 모델 구조를 최소한으로 읽는 순서

1. `integrations/openfoam13/case/generate_shear_layer.py`: 유동·초기장·변조 경계.
2. `src/adaptive_mg/v67/h2_fixed_p/backend.py`: 고정 P, 현재 A 갱신, 실제 stencil 재사용과 복구.
3. `src/adaptive_mg/v67/smoother_reuse/policy.py`: 관측과 재사용/갱신 판단.
4. `src/adaptive_mg/v67/smoother_reuse/world.py`: GRU/MLP, 예측과 두 단계 계획.
5. `src/adaptive_mg/v67/smoother_reuse/study.py`: 행동 비용 라벨, 데이터 분리, 평가·freeze.
6. `scripts/run_v6_7_shear_pipeline.py`: 주 실험 → 대조군 → 수소 적용의 실행 순서.

기반 API: [Foundation 13 codedFixedValue](https://cpp.openfoam.org/v13/classFoam_1_1codedFixedValueFvPatchField.html),
[Foundation 13 GAMGSolver](https://cpp.openfoam.org/v13/classFoam_1_1GAMGSolver.html).
