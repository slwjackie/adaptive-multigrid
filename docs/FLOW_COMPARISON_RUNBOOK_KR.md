# 세 유동·다중 level CNN·시간 이력 정책 비교

이 실험은 세 유동에서 **CNN smoother의 성능**과 **World Model이 갱신 시점을 선택해서 얻는 추가 이득**을 분리한다. 구현이나 단위 테스트 통과는 실제 CFD 수렴, 물리 검증 또는 가속 효과의 증거가 아니다. 새 CFD 기록으로 검증을 실행해야 한다.

## 물리 문제와 적용 범위

| 유동 | 구현 대상 | 이번 모델의 범위 |
|---|---|---|
| 조성 변조 He–N₂ 혼합층 | 학습·튜닝에도 조성 변조를 적용한 평면 전단 혼합층 | 2D 비반응 기체 혼합; 원래 조성 변조가 없는 케이스를 대조군으로 보존 |
| LOX–He 슬로싱 | 가속하는 밀폐 탱크의 액체산소–헬륨 계면 운동 | 고정 물성·등온·상변화 없는 VOF; 상부 기체는 순수 He 근사 |
| H₂ 횡류 제트 | N₂ 횡류에 맥동하는 H₂ 슬롯 제트 분사 | 2D 비반응 혼합; 3D 원형 분사기, 수소 화염 또는 터빈 실증을 뜻하지 않음 |

LOX–He는 헬륨으로 가압된 산소 탱크의 **계면 유체역학** 적용 사례다. 산소 증기, 기체 혼합, 벽면 열전달, 증발·응축, 능동 가압계통에 의한 탱크 압력 변화는 포함하지 않는다. 밀폐 비압축성 계산의 압력 기준점은 선형계의 영공간을 없애기 위한 것으로, 실제 탱크의 가압 제어 모델이 아니다.

세 경우 모두 고정 Cartesian 격자와 현재 행렬의 잔차 검증을 사용한다. `p_rgh`도 기준점 적용 후의 대칭 양정치 행렬만 받는다. 원래 H₂ 반응유동 파이프라인은 유지하며, 아래 `--with-h2`로 새 비교 실험에 반응유동 검증을 연결할 수 있다.

세 유동은 같은 CNN·World Model 구조 후보군을 사용하지만 **유동별로 별도 학습·튜닝**한다. 따라서 가중치가 다르며 선택되는 GRU 폭·출력부도 달라질 수 있다. 현재 비교는 유동별 갱신 정책의 효과를 보는 것이며, 하나의 공통 가중치가 모든 유동에 일반화함을 입증하지 않는다. ‘공유 GRU’는 한 모델 안의 예측 역할 간 기억 공유이며, 유동 간 가중치 공유가 아니다.

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

### 슬로싱 수집의 `exactly one bridge control required: maxIter` 오류

초기 구현은 `fvSolution` 파일 전체에서 `maxIter`가 하나라고 가정했다. 슬로싱에는 `pcorr`와 `p_rgh`에 각각 이 설정이 있어, 실제 CFD 실행 전에 bridge 설정 변경이 실패했다. 수정본은 `solver AdaptiveFixedP`가 선언된 단일 dictionary의 직접 항목만 바꾸며 다른 solver·주석·문자열·하위 dictionary의 설정은 보존한다. 중복/누락된 bridge 설정은 파일을 쓰기 전에 거부한다. 생성된 세 비반응 케이스에 실제 `_rewrite_bridge`를 적용하는 회귀시험도 추가했다.

이 수정은 Python 코드이며 OpenFOAM C++ 플러그인 재빌드는 필요 없다. 소스 해시가 바뀌고 실패한 collection 복사본이 남아 있으므로 **기존 `three_flow_merge_check_v1`에 `--resume`하지 않는다**. 원본 폴더·로그를 보존하고 새 계획으로 수집한다. 기록의 소스 해시를 수동 변경하거나 실패 폴더를 삭제해서 재개하지 않는다.

```bash
cd /home/jackie/adaptive-multigrid-h2
git pull --ff-only origin research/modulated-shear-temporal
source /opt/openfoam13/etc/bashrc
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -o pipefail
mkdir -p artifacts/merge_verify_logs_v1

python -m pytest tests/test_h2_fixed_p_fullrun.py tests/test_propulsion_flow_generators.py -ra

python -u scripts/run_v6_7_flow_comparison.py \
  --output artifacts/three_flow_merge_check_v2 \
  --profile quick --n 31 --repeats 3 \
  --independent-grus --with-h2 --h2-source-family h2_crossflow \
  --stop-after collect \
  2>&1 | tee artifacts/merge_verify_logs_v1/collect_v2.log
```

`collect`는 새 케이스 생성도 수행한다. 세 유동이 모두 수집되면 이후 `--resume --stop-after probe`, `validation`, `test`도 같은 `artifacts/three_flow_merge_check_v2`를 사용한다. 이미 성공한 혼합층의 기존 결과는 v1에 그대로 보존되며 v2에서는 수정된 소스로 다시 수집한다.

새 유동의 mesh/time-step 수렴성, 슬로싱 고유진동수·감쇠, 제트 궤적·혼합률의 실험 대조는 별도로 검증해야 한다. 생성된 케이스와 Python 테스트는 OpenFOAM Foundation 13에서의 실제 빌드·실행 검증을 대체하지 않는다.

구현 검증의 최신 시험 수와 범위는 `FLOW_COMPARISON_VALIDATION.json`에 기록한다. 독립 C++ wire 검사 및 27개 비반응 케이스 생성·계약 검증을 통과했다. OpenFOAM 플러그인 빌드·실제 CFD·속도 이득은 아직 검증하지 않았다.

격자를 늘릴 때는 설정 파일의 시간 간격도 CFL·모세관 시간 제한에 맞게 조정한 새 계획을 만든다. `--n`만 변경해서 모든 기존 시간 간격을 그대로 사용할 수 있다고 가정하지 않는다.

### `tune_slosh`의 작은 체적분율 초과 재검증

WSL의 v3 수집에서 `alpha_maximum=1.000000015491555`로 기존 `alpha_bounds_tolerance=1e-8` 검사가 실패했다. 액체 체적 상대오차 최댓값은 약 `4.55e-9`였고, 계면 운동과 유한한 압력·속도 검사도 통과했다. 이는 작은 수치적 overshoot이며, 이 출력만으로 발산이나 원인을 단정하지 않는다. 기계 정밀도 수준의 반올림 오차라고 부르지도 않는다.

우선 생성기의 alpha 선형 풀이 잔차 기준을 `1e-10 → 1e-12`로 강화했다. **체적분율 합격 기준 `1e-8`, 압력 허용오차, MULES 제어값은 유지한다.** alpha 잔차와 체적분율 최대오차는 같은 양이 아니므로 이 변경이 통과를 보장하지 않는다. alpha를 잘라내거나 이전 결과를 재분류하지 않는다. 강화된 alpha 풀이 비용도 이후 모든 solver arm의 전체 CFD 시간에 동일하게 포함된다.

아래 명령은 새 v4 계획에 세 유동·H₂ 후속 검증을 모두 포함하되, 먼저 실패했던 train/tune 케이스 하나만 실제 native bridge로 수집한다. C++ 플러그인은 이전 `p_rgh` 대응 버전으로 이미 다시 빌드했다면 재빌드할 필요 없다.

```bash
cd /home/jackie/adaptive-multigrid-h2
git pull --ff-only origin research/modulated-shear-temporal
source /opt/openfoam13/etc/bashrc
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
set -o pipefail
mkdir -p artifacts/merge_verify_logs_v4

sha256sum --quiet -c FLOW_COMPARISON_SOURCE.sha256 &&
sha256sum --quiet -c PACKAGE_MANIFEST.sha256 &&
python -m pytest tests/test_propulsion_flow_generators.py tests/test_new_flow_physics.py tests/test_flow_comparison_pipeline.py -ra &&
python -u scripts/run_v6_7_flow_comparison.py \
  --output artifacts/three_flow_merge_check_v4 \
  --families lox_he_sloshing shear_composition h2_crossflow \
  --profile quick --n 31 --repeats 3 \
  --independent-grus --with-h2 --h2-source-family h2_crossflow \
  --stop-after collect --collect-one lox_he_sloshing tune_slosh \
  2>&1 | tee artifacts/merge_verify_logs_v4/tune_slosh.log
```

단독 수집 성공 시 `status: partial`, `completed_through: collect_one`으로 표시한다. 전체 수집을 완료했다는 뜻이 아니다. 통과한 케이스의 원본 필드·서비스 증거를 재검사하여 다음 전체 수집에서 재사용한다.

```bash
python -u scripts/run_v6_7_flow_comparison.py \
  --output artifacts/three_flow_merge_check_v4 --resume --stop-after collect \
  2>&1 | tee artifacts/merge_verify_logs_v4/collect.log
```

다시 실패하면 전체 수집으로 넘어가지 않는다. `families/lox_he_sloshing/record_cases/tune_slosh/native_field_checks.json`에 실패한 수치와 필드 해시를 저장한다. `log.foamRun`의 alpha residual·극값·Courant 수·연속성 및 필요하면 시간 간격 민감도를 확인한다. 이 기록은 성공한 `physics_profile.json`을 만들지 않으며 실패한 디렉터리는 그대로 보존한다. v3는 이전 설정의 증거로 보존하고 새 코드로 resume하지 않는다. Python 회귀시험은 실제 WSL CFD 통과를 대신하지 않는다.

## 비반응 비교 → 실제 H₂ 반응유동 검증

기존 `run_v6_7_shear_pipeline.py --with-h2`는 별도 H₂ 연구를 실행했으며 새 3종 비교에서 선택한 구조를 전달하지 않았다. 이제 `run_v6_7_flow_comparison.py --with-h2`가 다음 단계를 연결한다.

1. 세 비반응 유동에서 CNN·World Model 구조를 tune 데이터로 선택하고 validation에서 수렴·물리장 허용치를 확인한다.
2. **실험 전에 지정한 source 유동**의 선택된 구조를 고정한다. 기본은 수소 연결 사례인 `h2_crossflow`이며 `--h2-source-family shear_composition` 또는 `lox_he_sloshing`으로 바꿀 수 있다. 서로 다른 유동의 절대 계산시간을 비교해 ‘최고 유동’을 고르거나 test 결과로 source를 바꾸지 않는다.
3. 실제 H₂–공기 반응 대향류의 별도 train/tune 데이터로 CNN·World Model을 새로 학습한다. CNN 공유/level별 구조, 적용 level, GRU 크기·출력부·학습 설정은 source에서 이어받지만 가중치는 복사하지 않는다. 이는 **구조/방법 전이이며 무학습 가중치 전이가 아니다**. source에서 World Model이 학습되지 않았다면 target에서도 억지로 활성화하지 않는다. source에 모델이 있어도 H₂ 자체의 갱신·재사용 probe가 후보 조건을 통과해야 target World Model을 학습한다.
4. Classical, 매번 갱신, H₂ tune에서 선택한 저렴한 규칙, 학습 가능했던 World Model을 실제 coupled CFD로 검증한다. `T, H2, U, p` 오차와 연속성, 유한한 양의 `Qdot`를 확인한다. 화학반응·수송·압력·I/O가 포함된 **전체 CFD 프로세스 벽시계 시간**을 측정하며 화학반응만의 시간을 별도로 계측한다고 주장하지 않는다.
5. 모든 비반응·반응 validation을 통과한 후 모델을 freeze하고 held-out test를 평가한다. 느려진 결과도 남기며 속도 이득 입증 여부는 자동으로 참으로 바꾸지 않는다.

최초 계획부터 옵션을 넣고 새 디렉터리에서 시작한다. 기존 소스로 만든 출력에 사후 추가하지 않는다. 아래 코드는 변경된 연구 브랜치를 로컬에 반영한 뒤 사용한다.

```bash
python -u scripts/run_v6_7_flow_comparison.py \
  --output artifacts/three_flow_reacting_v1 --profile quick \
  --with-h2 --h2-source-family h2_crossflow --stop-after probe

# 이 단계부터 Cantera와 OpenFOAM chemkinToFoam도 필요
python -u scripts/run_v6_7_flow_comparison.py \
  --output artifacts/three_flow_reacting_v1 --resume --stop-after validation

python -u scripts/run_v6_7_flow_comparison.py \
  --output artifacts/three_flow_reacting_v1 --resume --stop-after test
```

`probe`까지는 비반응 실험만 실행하고 H₂는 pending으로 남긴다. `validation`부터 반응 단계가 실행된다. `--dry-run --with-h2`는 Cantera/OpenFOAM 없이 계획만 작성한다. H₂ 기본 시간은 `2e-4 s`, 시간 간격은 `1e-6 s`인 짧은 기능 확인용 설정이다. 충분히 발달한 화염·통계 수렴을 보장하지 않으므로 본 실험은 최초 계획의 `--h2-end-time`, `--h2-delta-t`와 격자·초기조건을 물리적으로 검토해야 한다.

확인할 출력:

- `hydrogen_transfer.json`: source 선택·설정·체크포인트·validation 해시, 무학습 전이가 아님을 명시.
- `families/hydrogen_reacting/summary.json`: probe, 학습 여부, validation/test, 전체 CFD 시간 범위.
- `families/hydrogen_reacting/studies/selected_method/comparison_validation.json` 및 `comparison_test.json`: replay와 native/coupled 측정 결과.
- 최상위 `hydrogen_combustion_validated`: 실제 반응 validation의 수치·물리장 비교 통과 여부. 실험 대조, 터빈 검증 또는 World Model의 속도 향상 입증을 뜻하지 않는다. held-out test 완료 여부는 별도 기록한다.
