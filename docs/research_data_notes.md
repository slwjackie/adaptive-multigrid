# Strong-aware 연구 데이터와 final gate

기존 selector는 수정하지 않는다. `research_data.py`는 기존 `select_strong_strategy(A,n,rules)`를 호출하고, 각 sample에 전략 이름·rule·operator feature·rules digest를 저장한다. 학습 시 `Example.strong_selection` 또는 development manifest의 `selection`을 사용한다. selector가 RHS·해답·family 이름을 읽는 경로는 없다.

## 기본 규모

| split | 크기 | operators | 사용 |
|---|---|---:|---|
| train | 15,31,63,127 | 280 | 학습 |
| validation | 15,31,63,127 | 56 | architecture/checkpoint/KD/controller 선택 |
| final | 15,31,63,127 | 56 | freeze 뒤 한 번 |
| grid OOD | 255 | 14 | freeze 뒤 한 번 |
| rotation OOD | 31,63 | 20 | freeze 뒤 한 번 |
| anisotropy OOD | 31,63 | 12 | freeze 뒤 한 번 |
| contrast OOD | 31,63 | 8 | freeze 뒤 한 번 |
| topology OOD | 31,63 | 16 | freeze 뒤 한 번 |

기본 smoke는 train 7개(n15), validation 14개(n15/31)이다. smoke plan은 final freeze/materialization을 거부한다. 이 수는 작은 plumbing 검증용이며, final 성능 근거가 아니다. 전체 280개 생성·학습은 명시적인 full 명령에서만 실행한다.

Training angle은 15,30,45,60,75도(정렬 family는 0/90도), rotation OOD는 7.5,22.5,37.5,52.5,67.5,82.5도이다. Training anisotropy epsilon은 1e-3~1e-1, OOD는 1e-5~10^-3.1이다. Training heterogeneous contrast는 10~1000, OOD는 10^3.2~10^5이다. Topology OOD는 두 타원 patch, 엇갈린 checkerboard, 굽은 이중 channel, 두 spatial scale의 smooth coefficient를 사용한다. 균일 PDE에는 무의미한 topology OOD label을 붙이지 않는다. 모든 OOD에서도 동일한 structured P1 mesh와 Dirichlet discretization을 사용한다.

## 누출 검사

`historical_roots`에는 기존 프로젝트 `artifacts`와 별도로 보관한 이전 실행의 artifact root를 모두 명시한다. 대상 파일은 JSON manifest와 `raw_results.json`이며, checkpoint·array·log·per-repeat 기록은 읽지 않는다. 원본 train/tune/audit, 이미 본 strong 14-case validation, smoke, 새로 추가된 사용자 실행도 이름을 hard-code하지 않고 검색한다. 잘못된 provenance JSON은 조용히 무시하지 않고 실패한다.

모든 split에서 기존 scale-invariant `operator_digest`를 사용한다. 동일 A를 scalar 배율만 바꿔서 새 sample로 세지 않는다. history와 train/validation 내부·사이의 normalized duplicate는 결정적인 retry로 제외한다. Final 생성 때 history를 다시 읽어 development commit 이후 사용자가 관찰한 operator도 제외한다. Final/OOD끼리의 중복도 제외한다. 이는 fingerprint에 의한 operator-disjointness이고, 통계적 IID나 discretization별 independence를 증명하지는 않는다.

`make_research_plan()`은 seed/config commitment만 만든다. `prepare_development_data()`는 train/validation만 조립한다. final/OOD의 A·feature·C*·해답은 freeze와 explicit final claim 전에는 생성하지 않는다. 이것은 로컬 코드의 실수 방지와 reproducibility 규약이며, 사람이 Python/JSON을 의도적으로 고치는 공격까지 차단하는 보안 장치가 아니다.

## API

```python
from adaptive_mg.v67.research_data import (
    make_research_plan, prepare_development_data, load_development_data,
    freeze_research, claim_final_evaluation, materialize_final_data,
    complete_final_evaluation,
)

train, validation, manifest = prepare_development_data(
    run_root, rules, plan=make_research_plan(smoke=True),
    historical_roots=[project_root / 'artifacts'],
)
# 재실행은 overwrite가 아니라 검증된 reconstruct
train, validation, manifest = load_development_data(run_root, rules)
```

`run_root`는 이번 연구 전체 output directory이다. 이번 실행의 모든 하위 결과를 그 안에 두어 history scan이 자기 자신을 이전 연구로 오해하지 않게 한다. 이전 연구 root를 `run_root`로 지정하면 안 된다.

Full 연구가 끝나면 checkpoint·controller·selector JSON·관련 source 파일·최종 measurement config를 다음과 같이 고정한다. `checkpoint_paths`, `controller_paths`는 이름→파일 경로 dictionary이고, `source_paths`는 모든 관련 source 경로 목록이다. controller weight가 없으면 빈 dictionary로 명시하고 그 설정을 config에 포함한다.

```python
freeze = freeze_research(run_root, checkpoint_paths, rules_path,
                         controller_paths, source_paths, final_config)
with claim_final_evaluation(run_root, freeze, invocation, resume=False) as claim:
    final_splits, final_manifest = materialize_final_data(run_root, claim, rules)
    # freeze된 model/config로 final_splits 전체를 평가하고 결과를 저장한다.
    complete_final_evaluation(run_root, claim, result_paths)
```

Claim은 파일 lock과 exclusive ledger 생성으로 final invocation을 먼저 소비한다. 동일 checkpoint/config/command의 중단된 실행만 `resume=True`로 재개할 수 있고, 동시 실행과 완료 뒤 재사용은 거부한다. freeze 후 파일 변경, 다른 output, 다른 반복수·branch invocation은 거부한다. 결과가 완성되기 전에 `complete_final_evaluation`을 호출하지 않는 것은 평가 CLI의 책임이다. 이 함수는 결과 파일의 존재/hash를 고정하며 연구의 성공 여부를 대신 판정하지 않는다.

Final/OOD 결과는 architecture·teacher/KD 채택·controller label fitting에 넘기지 않는다. 최종 결과가 좋지 않더라도 같은 final을 다시 최적화 대상으로 쓰지 않는다. 새로운 연구를 시작한다면 새로운 plan과 development/final splits를 별도로 만들고 이전 final도 history에서 제외한다.

## 실제 시간 측정과 최종 판정

`research_evaluation.evaluate_research(examples, arm_specs, cfg, rules, out, ...)`는 실제 cold constructor+solve, 별도로 기록된 cache-prime 뒤 warm solve, fresh constructor+`solve_many`를 각각 실행한다. Multiple RHS는 실제로 서로 다른 manufactured RHS를 생성하며 1/4/16/64 batch가 같은 prefix를 공유한다. cold 시간을 단순 나눠서 multi-RHS라고 보고하지 않는다. 검증용 residual·L2 error·energy error·hierarchy nnz 계산은 timer 밖에서 수행한다. 성공 여부는 모든 반복의 모든 RHS에 대해 tolerance와 attempt budget을 다시 검사한다.

`arm_specs`에는 `strong_C: {model: None, branch: 'C'}`가 반드시 있어야 한다. Student는 `model=Components`, `branch='H_S'/'H_P'/'H_SP'`로 지정한다. Adaptive는 `policy=ResearchPolicy`, `branch='auto'`로 지정한다. Teacher에는 명시적인 `training_only=True`가 필요하고 offline context에서만 측정한다. Teacher를 final/OOD arm에 넣으면 거부한다.

출력은 `run_manifest.json`, per-repeat hash 기록, `raw_results.json`, `comparison.json`, `comparison.csv`이다. 반환값의 `rows[i]['runs'][arm][regime][str(rhs_count)]`에는 반복 측정들이 들어 있다. `summary[regime][str(rhs_count)][arm]`에는 successes/total, geometric_speedup/ci95, new_failure_case_ids, actual_neural_case_ids가 있다. 속도비는 Strong C 시간 / 비교 arm 시간이다. 실패 runtime은 raw에 남지만 속도비에는 포함하지 않는다. 신뢰구간은 반복 solve 수가 아니라 operator별 median ratio를 bootstrap한다.

`research_certification.certify_research_final(...)`는 기존 `evaluation.certify()`와 별도이다. active final claim, freeze 파일 hash, policy/model/controller signature, source/hardware, 저장된 raw 파일, 사전 확정된 전체 final cohort를 모두 확인한다. OOD를 섞거나 좋은 case만 고르면 인증하지 않는다. 최소 20개의 서로 다른 final operator와 공통 성공 operator, 각 5회 반복, strong success 손실 0, 실제 accepted neural work, GM ≥ `max(1.03, 1/(1-win_margin))`, CI 하한 >1을 요구한다. 기본 설정은 discrete L2/energy error 악화도 검사한다.

최종 freeze의 `config`에는 `solver`, `rules_digest`, `policy_digest`와 `final_evaluation={repeats, regimes, rhs_counts, ...}`를 넣는다. 선택적인 판정값은 `min_cases=20`, `require_solution_error_not_worse=True`, `minimum_speedup=max(1.03, 1/(1-win_margin))`가 기본이다. final을 본 뒤 판정 기준을 바꾸면 거부한다. `final_assessment.json`에 raw/manifest/comparison hash와 판정 근거를 기록하고 certificate를 반환한다. 증거가 부족하거나 성능 기준이 미달이면 `validated=False`이며, 기존 checkpoint·production policy를 수정하지 않는다. 통과한 경우에만 호출자가 `policy.attach_final_certificate()`로 별도 copy를 만들어 저장한다.
