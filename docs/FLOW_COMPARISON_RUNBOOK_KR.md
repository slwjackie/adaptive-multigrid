# 세 유동·다중 level CNN·시간 이력 정책 비교

이 실험은 세 유동에서 **CNN smoother의 성능**과 **World Model이 갱신 시점을 선택해서 얻는 추가 이득**을 분리한다. 구현이나 단위 테스트 통과는 실제 CFD 수렴, 물리 검증 또는 가속 효과의 증거가 아니다. 새 CFD 기록으로 검증을 실행해야 한다.

## 물리 문제와 적용 범위

| 유동 | 구현 대상 | 이번 모델의 범위 |
|---|---|---|
| 조성 변조 He–N₂ 혼합층 | 학습·튜닝에도 조성 변조를 적용한 평면 전단 혼합층 | 2D 비반응 기체 혼합; 원래 조성 변조가 없는 케이스를 대조군으로 보존 |
| LOX–He 슬로싱 | 가속하는 밀폐 탱크의 액체산소–헬륨 계면 운동 | 고정 물성·등온·상변화 없는 VOF; 상부 기체는 순수 He 근사 |
| H₂ 횡류 제트 | N₂ 횡류에 맥동하는 H₂ 슬롯 제트 분사 | 2D 비반응 혼합; 3D 원형 분사기, 수소 화염 또는 터빈 실증을 뜻하지 않음 |

LOX–He는 헬륨으로 가압된 산소 탱크의 **계면 유체역학** 적용 사례다. 산소 증기, 기체 혼합, 벽면 열전달, 증발·응축, 능동 가압계통에 의한 탱크 압력 변화는 포함하지 않는다. 밀폐 비압축성 계산의 압력 기준점은 선형계의 영공간을 없애기 위한 것으로, 실제 탱크의 가압 제어 모델이 아니다.

세 경우 모두 고정 Cartesian 격자와 현재 행렬의 잔차 검증을 사용한다. `p_rgh`도 기준점 적용 후의 대칭 양정치 행렬만 받는다. 원래 H₂ 반응유동 파이프라인은 별도의 적용 사례로 유지한다.

## 비교 방법

1. 먼저 조성 변조 혼합층에서 실제 행렬을 기록하고 짧은 갱신·재사용 probe를 실행한다.
2. 공유 CNN과 level별 독립 CNN을 **동일한 level 0, 1**에서 비교한다. 독립 CNN은 각 level의 현재 Galerkin 행렬을 입력받고, 고정 P의 전체 V-cycle 목적함수로 함께 학습한다. 각 CNN의 실제 gradient와 가중치 갱신을 기록한다.
3. 같은 CNN과 level 범위 안에서 Classical, 매번 갱신, 안전장치가 있는 재사용, 일정 간격, 행렬 변화·잔차 규칙, World Model을 비교한다.
4. World Model은 공유 GRU의 hidden 16/32/64와 기존 선형 출력/역할별 작은 비선형 MLP 출력을 비교한다. 역할별 독립 GRU도 별도 옵션으로 비교한다. 파라미터 수와 실제 정책 실행 비용을 함께 기록한다.
5. 튜닝 자료로 선택하고 validation을 통과한 구성을 고정한 뒤 test를 연다. test에서 가장 빠른 구성을 다시 골라 최종 성능으로 보고하지 않는다.

최대 반복 수는 600회이고 수렴하면 즉시 종료한다. 모든 정책은 같은 상대·절대 잔차 기준을 사용한다. CNN 가중치는 온라인에서 재학습하지 않는다. 갱신은 현재 행렬에서 smoother 연산자를 다시 생성하는 것이며, P는 고정한다. 현재 World Model의 한 행동은 선택된 level들의 smoother를 함께 갱신/재사용한다. level마다 다른 갱신 행동을 선택하는 정책은 후속 연구 범위다.

## 비용과 결과 해석

주된 비교 대상은 반복 횟수가 아닌 **전체 풀이 시간**이다. 초기 solver 구성, 현재 행렬의 설정, CNN 연산자 생성, 실제 V-cycle 적용, 특징 추출, World Model 예측·행동 선택, 실제 잔차 검사, 실패 복구를 포함한다. 오프라인 학습 및 기록 I/O는 별도로 보고한다. coupled 계산에서는 서비스 통신을 포함한 전체 CFD 시간도 확인한다.

- 공유 CNN과 독립 CNN의 비교는 smoother 구조의 효과다.
- World Model과 해당 CNN에서 튜닝한 저렴한 갱신 규칙의 비교가 World Model의 추가 효과다.
- 압력 replay 가속만으로 전체 CFD 가속을 주장할 수 없다. native 압력 처리 비율과 coupled 벽시계 시간을 함께 확인한다.
- probe는 양쪽 행동이 각각 유리한 **완전한 길이의 서로 겹치지 않는 구간**을 사용한다. 겹치지 않더라도 시간 상관성은 남으므로 통계적 독립 표본으로 단정하지 않는다.
- 시간 차이가 있어도 각 구간의 반복 수·생성·적용·복구 내역이 같은지 확인한다. 측정 잡음에 의한 갱신 승리를 물리적 예측 가능성으로 해석하지 않는다.
- 실패, 느려진 구성, 유의미한 갱신 이득이 없는 유동도 결과에 남긴다. probe 기준을 충족하지 않으면 World Model 학습을 건너뛰고 이유를 기록한다.

## 재현과 검증 상태

### WSL에서 시작

기존 연구 브랜치를 사용하며 merge하지 않는다. 작업 트리에 직접 수정한 파일이 있다면 먼저 보존하고, 아래 `pull --ff-only`가 충돌을 알리면 강제로 덮어쓰지 않는다.

```bash
cd /home/jackie/adaptive-multigrid-h2
git switch research/modulated-shear-temporal
git pull --ff-only origin research/modulated-shear-temporal
source /opt/openfoam13/etc/bashrc
source .venv/bin/activate
python -m pip install -e '.[dev]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash integrations/openfoam13/Allwmake
```

우선 조성 변조의 짧은 probe만 실행한다. 기본 비교 arm은 공유 CNN `[0]`, 공유 CNN `[0,1]`, 독립 CNN `[0,1]`이다. quick은 기능 확인용 학습 횟수와 짧은 유동 시간을 사용한다.

```bash
python -u scripts/run_v6_7_flow_comparison.py \
  --output artifacts/composition_probe_v1 \
  --families shear_composition --profile quick --stop-after probe
```

결과는 `summary.json`과 `families/shear_composition/probe_summary.json`에 모인다. 각 arm의 `studies/<arm>/reuse/collection.json`과 `reuse/episodes/`에서 갱신·재사용의 전체 시간과 구간별 반복·복구 내역을 확인한다. probe에서 멈추면 World Model 학습은 아직 수행하지 않는다.

세 유동과 기존 혼합층 대조군, 모든 CNN×World Model 조합까지 계획에 포함하려면 새 디렉터리에서 실행한다. 이 조합은 계산량이 크므로 먼저 `--stop-after generate`로 케이스만 만들고 계획을 확인할 수 있다. `generate`까지는 OpenFOAM 설치가 필요 없다.

```bash
python -u scripts/run_v6_7_flow_comparison.py \
  --output artifacts/three_flow_v1 --profile full \
  --include-legacy-control --independent-grus --full-factorial \
  --stop-after generate

python -u scripts/run_v6_7_flow_comparison.py \
  --output artifacts/three_flow_v1 --resume --stop-after probe

python -u scripts/run_v6_7_flow_comparison.py \
  --output artifacts/three_flow_v1 --resume --stop-after validation

python -u scripts/run_v6_7_flow_comparison.py \
  --output artifacts/three_flow_v1 --resume --stop-after test
```

`--full-factorial`이 없으면 World Model 구조 비교는 tune에서 선택한 두-level CNN에 수행한다. `--independent-grus`가 없으면 공유 GRU의 폭과 출력부만 비교한다. 옵션은 최초 계획에서 정하며 resume 중 실험군을 늘리지 않는다. World Model 후보 조건을 통과하지 못한 경우 학습을 건너뛴 사실도 결과다.

`selection.json`에는 test 이전의 선택을, `comparison_freeze.json`에는 고정한 구성을 기록한다. 각 study의 `comparison_validation.json`, `comparison_test.json`은 replay·coupled·native 압력 비율을 연결한다. 전체 `summary.json`은 level 추가, CNN 분리, World Model 추가 효과를 따로 제시한다. 비용 세부항목 중 `nn_generation_seconds`는 setup의 일부이고 `neural_apply_seconds`는 수치 풀이의 일부이므로 전체 시간에 다시 더하지 않는다.

이전 `shear_probe_v1`의 체크포인트·결과는 새 소스와 섞지 않는다. 소스·설정·입력·체크포인트를 해시로 묶으므로 코드 변경 후에는 새 실험 디렉터리를 사용한다. `--resume`은 같은 소스와 같은 계획에서 중단된 실험을 이어갈 때 사용한다.

완료된 단계는 재사용하지만, CFD나 학습 도중 실패해 불완전한 출력이 생긴 디렉터리를 자동 삭제·덮어쓰기하지 않는다. 해당 오류를 확인한 뒤 새 출력 디렉터리에서 재실행한다. `summary.json` 또는 `status.json`의 `status: failed`와 로그를 함께 확인한다.

새 유동의 mesh/time-step 수렴성, 슬로싱 고유진동수·감쇠, 제트 궤적·혼합률의 실험 대조는 별도로 검증해야 한다. 생성된 케이스와 Python 테스트는 OpenFOAM Foundation 13에서의 실제 빌드·실행 검증을 대체하지 않는다.

이번 구현 검증: 전체 Python 회귀시험 951개 통과, Cantera 미설치로 기존 반응유동 생성 시험 1개 건너뜀. 독립 C++ wire 검사 및 27개 케이스 생성·계약 검증 통과. 상세 범위는 `FLOW_COMPARISON_VALIDATION.json`에 기록한다. OpenFOAM 플러그인 빌드·실제 CFD·속도 이득은 아직 검증하지 않았다.

격자를 늘릴 때는 설정 파일의 시간 간격도 CFL·모세관 시간 제한에 맞게 조정한 새 계획을 만든다. `--n`만 변경해서 모든 기존 시간 간격을 그대로 사용할 수 있다고 가정하지 않는다.
