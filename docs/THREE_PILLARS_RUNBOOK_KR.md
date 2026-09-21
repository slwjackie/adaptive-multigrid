# Three-pillars 변경: 학습·실행·검증 안내

## 1. 적용 범위

기준 소스는 `0d20cb26762b7af1add697dfa4a41c7e4c760e0a`이다. 새 주 실행 경로는
`scripts/run_v6_7_three_pillars.py`이며, **기존 checkpoint나 개인 컴퓨터의 artifacts가 없어도 시작할 수 있다.**
기존 v6.6/v6.7 스크립트는 legacy 비교용으로 남긴다. 새로운 성능 결과는 아래 경로로 생성한다.

### Classical baseline

- 같은 zebra 색 line 간 실제 coupling이 없을 때만 색별 block-diagonal solve로 묶는다.
  coupling이 있으면 기존 순차 line solve로 돌아간다. reverse 순서와 rectangular level을 보존한다.
- 원래 numerical-work proxy를 보존하고 LU 호출 수와 처리된 line 수를 구분한다.
  부분 H_S가 추가 line LU를 lazy 생성하는 비용도 기록한다.
- selector v2는 방향과 heterogeneity를 함께 분류한다. v1 rules는 조용히 재사용하지 않고 거부한다.
- 새 calibrator는 독립적인 multi-size train/tune, 서로 다른 2개 RHS, 반복 측정 전부의 수렴과
  cycle headroom을 요구한다. train에서 고른 후보를 tune에서 승인/거부한다.
- 표본 또는 격자 범위가 부족한 leaf는 **동일한 fixed parent 전체 조합**으로 돌아간다.
  입력은 A와 grid shape뿐이다. PDE family/해/RHS/측정 시간은 온라인 selector 입력이 아니다.
- `C`, `H_S`, `H_P`, `H_SP`는 동일한 frozen `C*(A)`를 classical parent와 recovery로 공유한다.
  이 구조는 보존·검증했고, 별도의 online oracle/shadow solve는 추가하지 않았다.

### H_S: replacement 비용 최적화

- primary config는 `application="replace"`, `lazy=true`, `nn_levels=1`,
  `replace_pre=1`, `replace_post=0`이다. classical 2 pre/2 post 중 pre 1회를 대체하며 나머지는 유지한다.
- 이미 root gate가 있으면 버려지던 residual matvec를 실행하지 않는다.
- 모델 copy/device/dtype 준비는 setup transaction 내 모델당 한 번이다. checkpoint를 eval/dtype 변경으로 오염시키지 않는다.
- actual level A의 feature/digest를 cache한다. H_SP에서 learned P 때문에 바뀐 coarse A를 원래 A와 혼동하지 않는다.
- stencil CSR topology를 cache하되 수치 data는 bank별 독립 소유한다. native 경로에서는 CSR을 필요할 때까지 만들지 않는다.
- opt-in `Components.frozen_inference_copy()`는 불변 inference snapshot의 signature를 cache한다.
  정상적인 Torch in-place 변경은 version으로 감지한다. snapshot의 `.data`/NumPy alias 변경은 지원하지 않는다.
  기존 mutable 모델 API의 전체 hash 검사는 유지한다.
- hardware pool의 비싼 탐색은 cache하지만 실제 thread limit은 계속 확인한다.
  새 BLAS/OpenMP library를 로드한 경우 `hardware_environment(refresh=True)`로 갱신한다.
- `auto`는 `auto_device_min_cells` 미만을 CPU로 보낸다. 4096은 조정 가능한 보수적 시작값이지
  모든 기기에 대해 검증한 crossover가 아니다. 제공 primary config는 CPU를 명시한다.
- 1.05~1.15 cycle-cost ratio는 **목표**이지 보장 조건이 아니다. 실제 시간을 benchmark로 측정한다.

### H_P: support/complexity

- 새 학습 모델의 `support="support_preserving"`은 classical P의 실제 nonzero support 안에서만 가중치를 바꾼다.
  softmax 자체를 이 support에서 수행하여 외부 edge의 gradient도 0이다.
- training과 deployment가 같은 projection을 쓴다. baseline row sum(경계에서는 1이 아닐 수 있음),
  coarse injection, `R=P.T`, `Ac=P.T@A@P`를 보존한다.
- zero-delta는 classical P와 Ac를 그대로 재사용한다. 고정 support의 명백한 불가능 조건은 NN/Galerkin 전에 거부한다.
- Boolean symbolic Galerkin은 fill의 **상계**이다. 같은 support여도 numerical cancellation 때문에 실제 nnz는 다를 수 있어
  numeric Galerkin 뒤의 검사도 유지한다.
- 새 primary cap은 `complexity_reference="parent"`이다. local actual-A classical 비교와 전체
  selected parent hierarchy 대비 aggregate ratio를 구분한다. 기존 absolute 3.0에 의해 동일한 P가 거부되는 모순을 제거한다.
  별도 hard absolute ceiling은 선택 사항이다.
- 기존 standard/expanded 모델은 명시적인 legacy ablation으로 남긴다. 새 projection schema와 섞지 않는다.
  **기존 P checkpoint의 추론 설정만 바꾸지 말고 새 경로에서 재학습한다.**

## 2. 검증과 주장 범위

로컬 Linux CPU 검증: 회귀 테스트 546개 통과. 여기에는 batch/순차 수치·work 동등성,
rectangular/reverse/fallback, zero/perturbed learned P, 학습-실행 일치, cache mutation,
calibration 중단 후 재개, from-scratch smoke, 축소된 별도 final 계획의 동결/단일 소비 테스트가 포함된다.

기존 source에서도 실패하던 `semi_x` 행 합 검사 한 곳은 4e-16 고정 threshold 대신
`candidate_count * machine_epsilon * max(1,row_L1)`의 합산 오차 허용치를 사용하도록 고쳤다.
solver tolerance를 완화한 것이 아니며 support/injection/SPD 검사는 유지한다.

Smoke와 축소 final 경로 테스트는 **성능 우월성이나 전체 연구 final/OOD 평가가 아니다.**
전체 1120-update 학습, macOS/MPS/CUDA 성능, 255-grid OOD 속도 개선을 완료했다고 간주하지 않는다.
기존 fixed_C 대비 2배 결과는 과거 구현의 관측값으로 보존하되 새 구현의 성능 근거로 재사용하지 않는다.

이번 변경은 세 축과 재현 workflow에 한정된다. 새로운 `strong_C+guard`, size-binned adaptive policy,
대형 teacher/KD 재학습이나 자동 production 인증은 추가하지 않았다. 동적 production 정책의 인증은 별도 과제다.

## 3. 저장소와 환경 준비

새로 설치할 때:

```bash
git clone https://github.com/slwjackie/adaptive-multigrid.git
cd adaptive-multigrid
git switch main
```

이미 clone한 폴더라면 먼저 `git status --short`로 로컬 변경을 확인한다. 변경이 있으면 별도 commit/보관한 뒤:

```bash
cd /실제/저장소/경로/adaptive-multigrid
git switch main
git pull --ff-only origin main
```

Python 3.12 환경을 권장한다. 코드 요구사항은 Python >=3.10, Torch >=2.10이다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

C++ compiler가 있으면 native stencil도 준비한다. macOS에서는 `xcode-select --install`로 command-line tools를 설치할 수 있다.
OpenMP 설치가 없어도 실행되는 경로:

```bash
python scripts/build_native_stencil.py --no-openmp
```

OpenMP를 이미 정상 설치한 Linux 환경에서는 옵션 없이 빌드해도 된다. native는 선택 사항이며 CSR 경로도 지원한다.
**빌드와 thread 설정은 calibration 전에 마치고 해당 run 중에는 바꾸지 않는다.**

## 4. 테스트

```bash
python -m pytest -ra

# 이번 변경을 집중 확인
python -m pytest tests/test_line_smoother_batched.py tests/test_three_pillars.py -ra

# JUnit 결과 저장
mkdir -p artifacts/test_logs
python -m pytest -ra --junitxml=artifacts/test_logs/pytest.xml
```

native가 없는 환경에서 일부 native 테스트는 skip될 수 있다. skip을 pass로 바꿔 보고하지 않는다.

## 5. 먼저 실행할 smoke: 기존 모델 필요 없음

새 output 이름을 사용한다. 아래 `RUN`은 한 번 정한 뒤 모든 단계에서 동일하게 사용한다.

```bash
RUN="artifacts/three_pillars_smoke_$(date +%Y%m%d_%H%M%S)"
echo "$RUN"

# 1) n=7,15 / train 14개 + tune 14개 / classical 16개 후보 / 서로 다른 RHS 2개
python scripts/run_v6_7_three_pillars.py calibrate \
  --config configs/v6_7_three_pillars_smoke.json \
  --run-dir "$RUN"

# 2) 분리된 NN train 14개, validation 14개 생성. H_S/H_P 각 8 optimizer updates
python scripts/run_v6_7_three_pillars.py train \
  --run-dir "$RUN" --branches H_S H_P

# 3) 동일 C*(A)를 기준으로 cold/warm 및 실제 여러 RHS를 측정
python scripts/run_v6_7_three_pillars.py benchmark \
  --run-dir "$RUN" --branches H_S H_P \
  --tag validation --repeats 3 --warmups 1 \
  --rhs-counts 1 4 --regimes cold warm multiple

# 4) 별도의 channel operator를 실행. n=31은 이 smoke 학습보다 큰 격자 진단
python scripts/run_v6_7_three_pillars.py demo \
  --run-dir "$RUN" --branch H_S --family channel --n 31
python scripts/run_v6_7_three_pillars.py demo \
  --run-dir "$RUN" --branch H_P --family channel --n 31
```

`demo` family: `near_isotropic`, `aligned_anisotropy`, `rotated_anisotropy`, `local_patch`,
`checkerboard`, `channel`, `variable_coefficient`. `--branch C`로 같은 selector의 순수 classical을 실행한다.
`demo`는 진단용으로 반복 실행할 수 있지만 final 데이터는 아니다.

## 6. 본 학습

```bash
RUN="artifacts/three_pillars_research_$(date +%Y%m%d_%H%M%S)"
echo "$RUN"

python scripts/run_v6_7_three_pillars.py calibrate \
  --config configs/v6_7_three_pillars_research.json \
  --run-dir "$RUN"

python scripts/run_v6_7_three_pillars.py train \
  --run-dir "$RUN" --branches H_S H_P

python scripts/run_v6_7_three_pillars.py benchmark \
  --run-dir "$RUN" --branches H_S H_P \
  --tag validation --repeats 5 --warmups 1 \
  --rhs-counts 1 4 16 64 --regimes cold warm multiple
```

제공 research 설정:

| 항목 | 설정 |
|---|---|
| Calibration | n=15,31,63; family별 3개; train 63개 + tune 63개 |
| Calibration RHS/반복 | 독립 RHS 2개; 반복 3회; 16개 classical 조합 |
| Leaf coverage | 각 train/tune에 모든 calibration size 필요; 최소 독립 operator 6개 |
| 수렴 여유 | 모든 관측 RHS/반복이 150 cycles의 80% 이내 |
| NN training | n=15,31,63; family별 8개, 총 168개 |
| NN validation | n=15,31,63; family별 2개, 총 42개 |
| H_S/H_P | 각 1120 optimizer updates; hidden width 16 |
| Loss trajectory | learned prefix 4 V-cycles + classical tail 2 cycles |
| Inference/solver | NN FP32 CPU; solver와 true residual FP64 CPU |
| Primary replacement | fine level만; pre sweep 1회 교체; post classical |
| P cap | P nnz ratio <=1; Ac와 aggregate parent-complexity ratio <=1.15 |

1120은 epochs가 아니라 **optimizer updates**이다. warm/cold 모두 실제 수렴 여부를 확인하며 실패나
Neural work 부재를 속도 향상으로 해석하지 않는다. validation 결과에 따라 설정을 바꾸려면 새 RUN을 만든다.

### 선택: H_SP joint fine-tuning

H_S와 H_P 학습이 완료된 같은 RUN에서만:

```bash
python scripts/run_v6_7_three_pillars.py train --run-dir "$RUN" --branches H_SP
python scripts/run_v6_7_three_pillars.py benchmark \
  --run-dir "$RUN" --branches H_S H_P H_SP \
  --tag validation_joint --repeats 5 --warmups 1 \
  --rhs-counts 1 4 16 64 --regimes cold warm multiple
```

joint는 독립 H_S/H_P checkpoint에서 시작한다. 단순 H_SP 조립이 아니라 추가 학습이다.
H_SP가 H_S보다 빠르다는 가정은 하지 않는다.

## 7. 중단 후 재개

소스, 설정, thread/library/hardware, output을 그대로 유지한다.

```bash
# 연구 run 예시. smoke run이면 config도 smoke 파일로 바꾼다.
python scripts/run_v6_7_three_pillars.py calibrate \
  --config configs/v6_7_three_pillars_research.json --run-dir "$RUN" --resume
python scripts/run_v6_7_three_pillars.py train --run-dir "$RUN" --branches H_S H_P --resume
python scripts/run_v6_7_three_pillars.py benchmark \
  --run-dir "$RUN" --branches H_S H_P --tag validation \
  --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 \
  --regimes cold warm multiple --resume
```

source/settings/hardware가 달라지면 기존 timing/cache/certificate를 재사용하지 않도록 오류를 낸다.
Calibration은 이미 기록된 operator를 그대로 복구하므로 다른 run의 결과가 늘어도 dataset이 바뀌지 않는다.
checkpoint나 JSON을 수동 수정해 오류를 우회하지 않는다.

## 8. 독립 final/OOD: 개발이 끝난 뒤 한 번

smoke에서는 freeze/final을 의도적으로 거부한다. 본 연구 run에서만:

```bash
# 검증할 branch, repeats, RHS regime을 먼저 고정. 이후 학습/재보정 금지.
python scripts/run_v6_7_three_pillars.py freeze \
  --run-dir "$RUN" --branches H_S H_P \
  --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 \
  --regimes cold warm multiple

# 동결된 checkpoint/rules/source/config로만 final operators 생성 및 평가
python scripts/run_v6_7_three_pillars.py final --run-dir "$RUN"

# final 실행이 중단된 경우에만 동일 입력으로 재개
python scripts/run_v6_7_three_pillars.py final --run-dir "$RUN" --resume
```

H_SP도 평가하려면 freeze 시 `--branches H_S H_P H_SP`로 선언한다. 이후 변경할 수 없다.
final grid에는 n=127, grid OOD에는 n=255가 포함된다. 이 크기는 calibration 범위 밖이므로
검증되지 않은 selector leaf/size에서는 fixed parent로 abstain할 수 있다. 이것은 강제로 외삽하지 않는 설계다.
freeze는 성능 인증이 아니고, final 완료 역시 Neural speedup을 자동으로 주장하거나 adaptive production certificate를 발급하지 않는다.

## 9. 결과 위치와 읽는 법

```text
RUN/
  configuration.json
  three_pillars_manifest.json
  calibration_manifest.json
  calibration_records/selector_train/*.json
  calibration_records/selector_tune/*.json
  selector_rules.json
  selector_evidence.json
  data/development_manifest.json
  data/dataset_plan.json
  checkpoints/H_S/candidate.pt
  checkpoints/H_P/candidate.pt
  checkpoints/H_S/status.json
  checkpoints/H_P/status.json
  benchmarks/validation/raw_results.json
  benchmarks/validation/comparison.json
  benchmarks/validation/failure_inclusive_wall.json
  demos/channel_31_H_S.json
  final/<split>/raw_results.json
```

`comparison.json`의 success/실제 Neural 사용을 먼저 확인하고 `geometric_speedup`을 읽는다.
`>1`은 strong_C보다 빠르고, `<1`은 느리다는 뜻이다. `fixed_C→strong_C`는 classical 선택 효과,
`strong_C→H_*`는 같은 parent 위 Neural의 추가 효과다. 서로 다른 성공 cohort의 비율을 나누어 CI를 만들지 않는다.
`failure_inclusive_wall.json`은 실패까지 소비한 elapsed time이다. 실패를 성공한 time-to-solution으로 처리한 순위가 아니다.

`cold`는 **모델이 로드된 process에서 새 A를 받아 setup+solve**하는 비용이다. A feature/selector,
classical hierarchy/factors, NN inference와 bank 생성, 실패·rollback, true-residual 검사가 포함된다.
Python 시작/import/모델 파일 읽기/불변 모델 session 준비와 offline 학습은 별도이다.
완전한 process cold-start 지연은 별도 측정해야 한다. `warm`은 동일 A의 bank 재사용,
`multiple`은 같은 A에서 실제 서로 다른 RHS 묶음의 전체시간이다.

## 10. 설정 변경 원칙

새 config를 복사해 **calibration 전에** 수정한다. 제공 기본 설정은 CPU에서 검증한 출발점이다.

```bash
cp configs/v6_7_three_pillars_research.json configs/my_three_pillars.json
# 편집 후 항상 새 RUN 사용
python scripts/run_v6_7_three_pillars.py calibrate \
  --config configs/my_three_pillars.json --run-dir artifacts/my_new_run
```

`solver.mg.nn_levels`, `solver.replace_pre`, `solver.replace_post`를 바꾸는 실험은
실행 schedule 자체를 바꾸므로 해당 schedule로 재학습/재검증한다.
FP32 inference model과 FP64 numerical solver 계약을 유지한다.
`inference_device=auto`의 threshold는 hardware 측정 후 조정한다. MPS/CUDA 성능은 이 변경에서 별도로 인증하지 않았다.

과거 top-level source/package manifest는 릴리스 스냅샷의 기록이며, 현재 git 작업트리 검증에는
새 `THREE_PILLARS_SOURCE.sha256`를 사용한다. run 자체도 source digest를 고정한다.
