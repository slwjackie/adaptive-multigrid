# Adaptive Neural Multigrid v6.7 limited fix

이번 변경은 ① stage 상대 승격, ④ 저장 후보의 classical 직접 비교,
⑤ abstention 이후 반복 controller 비용 제거, ⑦ C/H_S/H_P/H_SP bank에 한정한다.
식별자는 `v67-limited-fix-1`이다. 기존 package/API 이름을 유지한다.

## 위치와 원본 보존

실제 프로젝트는 `/Users/kimjiin/Desktop/adaptive_neural_multigrid_v6_7`이다.
Git checkout이 아닌 배포 디렉터리이며, 프로젝트 및 관련 상위/하위 경로에서
AGENTS.md는 발견되지 않았다. 기존 소스를 먼저 hash하고 작업 사본에서 수정했다.
적용 전 같은 파일이 사용자가 변경하지 않았는지 다시 검사한다.
Python은 해당 프로젝트의 `.venv/bin/python`이며 설치 후 실제 import 경로도 확인한다.
의존성 설치·upgrade, 기존 연구 재학습, 기존 프로세스 종료는 하지 않았다.

원본 `artifacts/my_v6_7_research`는 읽기 전용으로 취급한다. 원본 35개 중요
checkpoint/status/raw/config/certificate 파일의 SHA-256을 수정 전후 비교한다.
새 결과는 `artifacts/my_v6_7_limited_fix`에만 둔다. `.venv`와 기존 artifacts 전체는 복제하지 않았다.
이번 변경의 diff와 수정 전 변경대상 소스는 새 결과의 `patch/`에 있다.

## 확인한 원래 결과

| Stage | 실제 거부 조건 | 공통 성공 | 이전 incumbent 대비 GM | 새 상대 정책의 보존 원자료 재판정 |
|---|---|---:|---:|---|
| smoother | 모든 사례 공통 성공 조건 | 13/14 | 1.1477883624 | 승격 |
| transfer | 시간 기준 미달 | 14/14 | 0.9834638713 | 거부 |
| joint | 모든 사례 공통 성공 조건 | 13/14 | 1.1059459203 | 승격 |
| hierarchical | 시간 기준 미달, neural 사용 없음 | 14/14 | 1.0108131543 | 거부 |

smoother와 joint에서 `checkerboard_31_4_seed20360914`는 두 모델 모두 실패했다.
각 stage의 신규 실패와 추가 해결은 0이다. S/J는 incumbent와 candidate 각각
13/14 성공이다. 기준은 원래 config의 3% 시간 감소, 즉 `1/(1-0.03) = 1.0309278351`을 유지했다.
**1.148×/1.106×는 classical 대비 speedup이 아니다.**

candidate.pt 4개는 모두 보존되어 있었다. 첫 세 stage의 entry snapshot은 없었다.
원래 seed와 hidden으로 초기 모델을 재구성한 뒤, archived previous-arm 실행
126개의 전체 모델 signature와 일치함을 확인했다. 이는 복원된 entry이며 원본
checkpoint 파일이나 원래 metadata 전체가 남아 있다는 뜻은 아니다.
hierarchical entry는 stage 순서와 raw signature가 일치하는 `controller/controller.pt`이다.
최종 checkpoint를 rejected candidate 또는 entry로 대신 사용하지 않는다.

원래 audit는 classical/adaptive 모두 33/35 성공, GM 0.9711885162,
CI [0.9555686607, 0.9857015072], final certificate=False였다.
원래 neural_used_cases는 accepted neural cycles를 기준으로 한다.
실제 175개 adaptive repeat를 확인하니 NN setup/attempt/accepted/rejected/application,
detector 호출 모두 0이었고 controller 호출은 6,695회였다.

## 구현과 의미

Stage 설정은 연구 JSON의 최상위 training 설정 또는 재평가 CLI에서 지정한다.

```json
{
  "stage_acceptance_policy": "stage_relative_v2",
  "stage_min_common_success": 5,
  "stage_min_common_fraction": 0.5
}
```

최소 표본은 `max(5, ceil(0.5 * total))`이다. stage가 학습한 parameter의 실제 값,
성공 case ID 집합, 모든 repeat의 tolerance/attempt budget 충족, 공통 성공의 시간,
실제 numerical neural application을 검사한다. 메타데이터나 signature 문자열만
달라져도 weight 변경으로 인정하지 않는다. 실패 시간은 successful time-to-tolerance로
쓰지 않는다. `strict_v1`은 원래 full-model-signature 및 all-common 조건을 재현한다.
새 status에는 개별 checks/failed_conditions, 신규 실패/추가 해결/동시 실패 ID,
GM/CI/기준, component 변화와 호출 통계가 있다.

`stage_promoted`, `direct_classical_speedup_observed`,
`final_time_improvement_certified`는 독립된 필드다.
archived 재판정은 새 시간 측정이 아니다. 새 selected 모델은 원본 후보의 S/P를 그대로
보존하고 옛 detector/controller를 stale로 표시한다. 기존 final certificate 파일과
`certify()`의 성공·시간 조건은 변경하지 않는다.

| Branch | 실행 | 준비하지 않는 것 |
|---|---|---|
| C | 원래 classical hierarchy/smoother | S/P forward, learned bank |
| H_S | C의 P/R/Ac/factor 공유 + cached S sweep 교체 | P forward, learned-P hierarchy build |
| H_P | learned P/R/Ac + 해당 Ac의 classical factors | S forward, smoother detector |
| H_SP | learned P/R/Ac + 그 Ac에서 생성한 cached S | 추가 coarse cycle |

`AdaptiveConfig(branch='H_S')` 등으로 명시한다. forced 진단은 `mode='research',
spatial=False, gate_mode='open'`을 쓴다. 기존 rollback/safety와 near-tolerance lock,
attempt budget은 유지한다. setup 및 rejected trial/recovery 비용도 모두 포함한다.
S/P forward는 bank 준비 때만 발생하며 residual만 변한다고 재생성하지 않는다.
같은 PreparedAdaptiveMG에서 `ensure_branch()` 또는 `solve_many()`로 cache를 공유한다.
A/config/component weight/device/dtype/actual coarse operator 의존성에 따라 관련 bank를 갱신한다.
S 생성 시 필요한 dtype/device 변환은 checkpoint 모듈 원본을 바꾸지 않는다.

auto는 설정된 component 조합에 기존 controller를 적용하는 작은 확장이다.
H_S/H_P는 일치하는 branch calibration이 없으면 `uncalibrated_branch`로 classical에
머문다. H_SP의 기존 policy는 원래 provenance가 유효할 때만 사용한다.
여러 branch를 온라인에서 전부 풀어 공짜로 가장 빠른 것을 선택하지 않는다.

시작 때 C-only로 확정되면 원래 classical_cycle을 호출하는 전용 루프로 바로 들어간다.
LOCK이면 마지막 accepted x와 남은 attempt budget을 그대로 이어간다.
temporary CLASSICAL은 다시 H로 들어갈 수 있다. LOCK 이후 과거 trajectory가 처음부터
C였던 것과 같다고 주장하지 않는다. 초기 행렬 검증/hash, hierarchy 준비, 필요 시
policy/certificate 검사와 결과 요약 비용은 남는다. 반복 temporal feature/MLP,
detector, neural hash, learned setup 및 adaptive 상태 갱신은 제거한다.
정확한 표현은 **per-cycle learned-controller overhead removed after abstention**이다.

## 이번에 실행한 작은 측정

새 진단 seed의 n=15 두 case, 8 arms, arm당 5 repeats: 총 80회 측정.
각 case/arm당 warmup 1회는 별도이며 측정 수에 포함하지 않는다.
모든 arm은 2/2 case, 10/10 repeats 성공, 공통 성공 2, 신규 실패/추가 해결 0이었다.

| Arm | classical 대비 GM |
|---|---:|
| original final research policy | 0.8765× |
| original final production policy | 0.9830× |
| smoother candidate / H_S | 0.9154× |
| transfer candidate / H_P | 0.6784× |
| joint candidate / H_SP | 0.6902× |
| joint candidate / H_S | 0.9096× |
| joint candidate / H_P | 0.6660× |

이 작은 grid에서는 모든 neural arm이 classical보다 느렸다. H_S/H_P/H_SP는 두
case 모두 실제 numerical application이 있었으며 H_S의 P forward/build=0,
H_P의 S/detector forward=0을 확인했다. 원래 final policy는 neural application 0이다.
2개 case의 CI는 파일에 제공하지만 일반화/성능 인증 근거가 아니다.

별도 stage smoke는 기존 tune의 첫 2개 case에서 smoother/joint 각 3 repeats,
2 arms로 총 24회 측정했다. incumbent 대비 각각 약 1.0113×, 0.9764×였다.
둘 다 최소 표본 5 및 시간 기준을 충족하지 못해 승격하지 않았다.
기존 archived 14-case 재판정과 이 fresh 2-case 측정을 구분해야 한다.

전체 회귀와 신규 테스트의 정확한 통과 수는 `logs/all_tests_installed.log` 및
`completion_report.json`에 저장한다. policy refit은 synthetic n=7/1epoch 통합 테스트로
연결만 검사했다. 실제 연구 candidate의 full policy refit, 35-case full direct benchmark,
224-operator S/P 재학습은 **미실행**이다. 설치 후 추가 최소 smoke는 별도로 표시한다.

## 시간·재현 조건

baseline은 원래 `configuration.json`을 읽는다: `line_alt_bilinear_full`, pre/post 2/2,
150 attempts, relative 1e-8, absolute 1e-12, initial residual normalization,
CPU FP32 generation/FP64 solve, torch/native threads 1. 이를 코드에 새 상수로 고정하지 않는다.
config, 실행 환경, 실제 runtime pools, seed, A/b/x0 digest와 checkpoint hash를 저장한다.
native library는 이미 존재할 때만 프로세스 초기화에서 로드한다. 컴파일/autotune은 없다.
추가 OpenMP library의 기본 pool 크기가 기록되더라도 실제 stencil kernel의 thread 수는
원래 config의 native_threads를 사용한다. 모든 arm에 같은 환경을 적용한다.

측정은 이미 import된 process 안의 cold constructor+solve wall time이다.
process 시작/import/checkpoint load/공통 native-library load는 제외한다.
계층/factor 준비, NN 생성, 판단, 반복, rejected attempt/recovery는 포함한다.
기존 `setup_seconds`는 classical constructor 시간이고 lazy NN setup은 `solve_seconds`에
들어간다. `nn_setup_seconds`/`branch_setup_seconds`를 total에 다시 더하지 않는다.
`nn_forward`는 heavy S/P 합, `total_nn_forward`는 tiny controller/detector까지 포함한다.
`transfer_bank_builds`가 실제 learned-P build 수이며 기존 호환 필드
`learned_hierarchy_builds`에는 S overlay 준비도 포함된다.

case별 실행 순서는 repeat마다 seed로 무작위화한다. 저장된 audit를 재구성할 때 전체
manifest의 exact specification과 A digest를 확인한다. b는 변경하지 않은 원래
deterministic generator/seed로 재생성한다. 원본에는 독립 b digest가 없어 과거 b digest와
직접 대조했다는 주장은 하지 않는다. 새 측정 b/x0/exact digest는 저장한다.
기존 audit 재사용은 항상 `diagnostic re-evaluation of previously inspected audit`이다.

## 복사해서 실행할 명령

아래의 manual_01은 새 결과 디렉터리다. 이미 실행했으면 다른 이름을 사용한다.
완료되지 않은 자기 작업만 동일 input/config/version으로 `--resume`할 수 있다.
원래 완료된 research에 pipeline --resume을 걸어 단계를 건너뛰지 않는다.

### 1. 프로젝트와 Python

```bash
cd /Users/kimjiin/Desktop/adaptive_neural_multigrid_v6_7
PY=.venv/bin/python
RUN=artifacts/my_v6_7_limited_fix/manual_01
export PYTHONDONTWRITEBYTECODE=1
$PY -c 'import sys, adaptive_mg; print(sys.executable); print(adaptive_mg.__file__)'
```

### 2. 테스트

```bash
$PY -m pytest
```

### 3. 원본 후보 검색·진단

```bash
$PY scripts/inspect_v6_7_candidates.py --research artifacts/my_v6_7_research --output "$RUN/inspection"
$PY scripts/inspect_v6_7_candidates.py --verify-hashes artifacts/my_v6_7_limited_fix/inspection/original_preservation_hashes.json
```

수정 전 진단은 이번 세션의 `inspection/original_acceptance_diagnosis.json`에 보존된다.
다시 실행한 검사는 현재 코드 위치와 archived 실제 판정 근거를 함께 기록한다.

### 4. 재학습 없이 stage 재평가와 classical 직접 비교

```bash
# 원래 raw timings의 정책 재판정: 학습·새 측정 없음
$PY scripts/reevaluate_v6_7_candidates.py --measurement archived --output-dir "$RUN/stages_archived"
# 원래 14개 stage 사례를 실제로 다시 측정: 원래 validation repeats 사용
$PY scripts/reevaluate_v6_7_candidates.py --measurement remeasure --output-dir "$RUN/stages_fresh"
# 원래 audit 35개 직접 비교. 오래 걸릴 수 있으며 이 세션에서는 실행하지 않음
$PY scripts/benchmark_v6_7_limited.py --split audit --repeats 5 --output-dir "$RUN/direct_audit"
```

원래 strict 판정 재현은 `--policy strict_v1`과 새 output-dir를 사용한다.
빠른 점검은 benchmark `--split smoke`, stage `--limit 2`로 한다.

### 5. 동일 checkpoint의 explicit four-branch 비교

```bash
$PY scripts/benchmark_v6_7_limited.py --checkpoint artifacts/my_v6_7_research/joint/candidate.pt --branches C,H_S,H_P,H_SP --split audit --repeats 5 --output-dir "$RUN/joint_four_branches"
```

### 6. 필요한 경우에만 branch별 label 재생성 / policy refit

```bash
$PY scripts/refit_v6_7_limited_policy.py --checkpoint artifacts/my_v6_7_limited_fix/stages_archived/joint/selected.pt --branch H_SP --output-dir "$RUN/policy_joint"
```

이는 기존 train split의 detector/controller labels와 기존 loss를 사용한다.
S/P 전체 학습을 호출하지 않는다. 실제 비용이 큰 optional 명령이며 이 세션에서는
실행하지 않았다. H_S는 smoother selected + `--branch H_S`, H_P는 transfer candidate +
`--branch H_P`로 별도 보정한다. 다른 후보의 S/P를 섞지 않는다.

### 7. 새 policy로 adaptive 실행

```bash
$PY scripts/benchmark_v6_7_limited.py --checkpoint "$RUN/policy_joint/calibrated.pt" --branches C --adaptive-checkpoint "$RUN/policy_joint/calibrated.pt" --split audit --repeats 5 --output-dir "$RUN/refit_adaptive"
```

연구 adaptive와 production을 구분해 저장한다. calibration은 final certificate가 아니다.
독립 final audit를 통과하지 않은 production은 원래 안전 정책에 따라 C로 동작한다.

### 8. 상세 결과 확인

```bash
$PY -m json.tool artifacts/my_v6_7_limited_fix/completion_report.json
$PY -m json.tool "$RUN/stages_archived/reason_summary.json"
$PY -m json.tool "$RUN/direct_audit/comparison.json"
open "$RUN/direct_audit/comparison.csv"
open "$RUN/direct_audit/branch_statistics.json"
$PY scripts/benchmark_v6_7_limited.py --help
$PY scripts/refit_v6_7_limited_policy.py --help
```

`comparison.csv/json`에는 case ID/n/checkpoint/branch, successful repeats,
성공 여부, C/candidate cycles, residual/error, setup/solve/total, 별도 두 speedup 필드,
NN/application/controller/detector, attempts/accepted/rejected/fallback,
classical strategy/P bank/실제 branch가 있다. 원자료는 `cases/*.json`,
`raw_results.json`, 실행 조건은 `run_manifest.json`에 있다.

## 핵심 산출물

- `inspection/checkpoint_registry.json`, `original_acceptance_diagnosis.json`: 원본 구조와 판정 근거
- `inspection/original_preservation_hashes.json`: 원본 중요 파일 보존 검사
- `stages_archived/`: 원래 raw 재판정, reason 요약, 새 selected 2개
- `stages_smoke/`: fresh stage 소규모 원자료와 최소 표본 거부
- `direct_smoke/`: case별 CSV/JSON, branch 통계와 raw timings
- `installed_smoke/`: 실제 설치된 소스의 네 branch 연결 확인
- `logs/`: 원래/설치 후 테스트, CLI --help, smoke 로그
- `patch/limited_fix.patch`, `patch/changed_files.json`, `patch/source_before/`: 이번 변경과 수정 전 소스
- `completion_report.json`: 실행/미실행 범위 및 보존 결과
