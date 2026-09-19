# v6.7 — A-only Classical Selector와 Neural MG 비교

실제 프로젝트는 `/Users/kimjiin/Desktop/adaptive_neural_multigrid_v6_7`이다. Python은 이 프로젝트의 `.venv/bin/python`을 사용한다. 기존 fixed Classical, research 결과, limited-fix 결과, checkpoint, final certificate는 보존한다. 새 실험은 `artifacts/my_v6_7_strong_baseline/`에 저장한다.

이 확장은 **PDE operator A와 구조 격자 크기만으로 classical strategy를 하나 선택**하고, 그 선택에서 출발하는 C/H_S/H_P/H_SP를 동일한 조건에서 비교한다. 코드의 `strong_C`는 검증할 baseline의 이름이다. 모든 문제에서 최적이거나 기존 bank보다 강하다는 보증을 뜻하지 않는다.

## 1. 실행되는 비교

| Arm | Classical 기반 | Neural component |
|---|---|---|
| `fixed_C` | 기존 research config의 strategy | 없음 |
| `strong_C` | A-only selector가 고른 strategy | 없음 |
| `fixed_H_S` | 기존 fixed Classical | 보존된 smoother candidate |
| `strong_H_S` | strong_C와 같은 P/R/A_coarse 및 선택 strategy | smoother candidate |
| `strong_H_P` | 같은 선택 strategy의 smoother와 coarsening | transfer candidate의 P/R/A_coarse |
| `strong_H_SP` | 같은 선택 strategy와 coarsening | joint candidate의 S와 P |
| `oracle::...` | 지정한 bank의 모든 strategy를 실제 측정 | 없음; offline 참고용 |

H_P의 classical coarse smoother는 learned coarse A에 맞춰 준비된다. 모든 neural arm은 기존 rollback/safety를 유지하는 **명시적 branch 실행**이다. NN을 쓰려다 거절된 시도와 복구 비용도 포함한다. 기존 fixed-baseline controller/certificate를 새 baseline에 유효한 것으로 재사용하지 않는다. `auto`를 쓰면 calibration 부재가 명시되므로, 이 비교에는 explicit branch를 사용한다.

기존 후보는 fixed baseline에서 학습된 가중치 그대로다. 이번 결과는 **기반 classical strategy를 바꿨을 때 기존 NN 효과가 유지되는지**에 대한 평가다. 새 기반에서 재학습한 Neural MG의 최종 성능과 구분해야 한다.

## 2. Deterministic rule

구현: `src/adaptive_mg/v67/strong.py`와 `configs/v6_7_strong_rules.json`.

`operator_features(A, n)`은 sparse A의 내부 격자 signed second moments, 방향별 coupling, 대각 성분의 contrast 및 log variation, local anisotropy와 방향 변화량을 계산한다. 비용은 sparse 변환 이후 O(nnz(A)+N)이다. constant-coefficient P1 diffusion의 내부 second moment는 확산 tensor의 방향/비율을 복원하며, 일반 heterogeneous operator에서는 **proxy**로 해석한다. 제거된 Dirichlet coupling의 편향을 줄이기 위해 최외곽 노드는 moment 평균에서 제외한다. FE off-diagonal의 양수 부호도 유지한다.

선택 함수는 case 이름, PDE label, b, x0, exact solution, residual, checkpoint, 측정 시간에 접근하지 않는다. hierarchy 생성이나 trial solve 없이 하나를 고른다. 동일한 A와 rules에서 C/H_S/H_P/H_SP는 같은 결정을 사용한다.

규칙은 다음 순서로 검사한다. 숫자와 leaf mapping은 JSON에 고정하고 hash를 기록한다.

| Rule | 기본 전략 |
|---|---|
| 지원 stencil 밖 coupling이 크거나 tensor 추정이 불확실 | `line_alt_operator_full` |
| local 방향 변화와 이방성이 큼 | `line_alt_operator_alternating` |
| contrast/heterogeneity가 큼 | `line_alt_operator_full` |
| x 정렬 강한 이방성 | `line_x_operator_semi_y` |
| y 정렬 강한 이방성 | `line_y_operator_semi_x` |
| +45도 부근의 이방성 | `line_diag45_operator_full` |
| 기타 회전 이방성 | `line_alt_operator_full` |
| 거의 등방성 | `jacobi_bilinear_full` |
| 나머지 | `chebyshev_operator_full` |

기본 이방성 ratio threshold는 4, 등방성 상한은 1.5, 정렬/대각 허용각은 12도다. 나머지 threshold와 전체 우선순위는 JSON/코드에 명시되어 있다. 기본값은 이론에서 착안한 검증 전 후보이며 smoke나 audit 결과를 보고 바꾸지 않았다.

기존 classical bank의 `controlled` 16개 또는 `all` 48개를 사용한다. Jacobi/Chebyshev/x-line/y-line/alternating-line/+45-degree-line, bilinear/operator-dependent P, full/x-first/y-first/alternating coarsening의 기존 구현이다. 현재 `semi_*`는 한 축의 coarsening을 우선하다가 다른 축도 줄이며, 영구적인 한 방향 semicoarsening과 다르다. -45-degree line이나 외부 AMG/ILU는 포함하지 않는다. operator-P도 기존 fixed-support approximate-ideal interpolation 구현이며 hypre SMG와 동일하다고 주장하지 않는다. 이 bank의 최선이 전체 classical solver의 최선이라는 의미는 아니다.

점 relaxation과 더 강건한 line/plane relaxation, semicoarsening을 구분하는 배경은 [hypre 공식 SMG/PFMG 설명](https://hypre.readthedocs.io/en/latest/solvers-smg-pfmg.html)을 참고할 수 있다. 실제 이 프로젝트에서의 우열은 아래 측정으로 결정한다.

## 3. Offline 보정과 정보 유출 방지

`scripts/calibrate_v6_7_strong_rules.py`는 **완료된 train/tune classical bank 측정**만 읽는다. 새 solve나 NN 학습은 하지 않는다. feature threshold/분기 조건을 유지하면서 각 leaf가 고를 classical strategy만 보정한다. 실행 시에는 보정된 JSON만 읽으므로 여전히 deterministic A-only selector다.

- original train/tune manifest와 case가 일치해야 한다. audit/smoke/fresh를 보정 입력으로 주면 거부한다.
- A/RHS/x0 digest, 원래 fixed Classical strategy, stopping/work config, 반복 수, 실제 residual 성공을 검사한다.
- leaf당 기본 3개 미만이면 기본 선택을 유지한다. 관측되지 않은 leaf도 유지한다.
- fixed_C 또는 기존 selector가 성공한 case를 하나라도 잃는 후보는 배제한다. 단순 성공 개수만 비교하는 success swap을 허용하지 않는다.
- 나머지는 성공 개수를 우선하고, 동일한 fixed-success cohort의 geometric relative time으로 선택한다. 1% timing tie에서는 기존 전략을 유지한다.
- fixed_C 성공이 없는 leaf에서는 실패 시간을 speedup으로 쓰지 않는다. rescue 개수만 비교한다.
- 출력에는 입력 manifest/raw hash, 사용한 case, rules hash, leaf별 표본 수와 선택 근거가 남는다. 새 디렉터리를 요구하며 기존 규칙을 덮어쓰지 않는다.

보정한 뒤 별도 validation으로 강도를 확인해야 한다. 반복해서 validation을 보고 규칙을 고치면 그 집합도 tune 데이터가 된다. 최종 논문 실험에는 충분한 train/tune coverage와 사전에 고정한 별도의 test 설계가 필요하다.

## 4. 시간·정확도 평가

모든 arm이 같은 A/b/zero-x0, tolerance, residual normalization, maximum attempted cycles, pre/post sweeps, learned replacement budget, precision/device/native thread 설정을 사용한다. 해당 값은 원래 `artifacts/my_v6_7_research/configuration.json`에서 읽는다. 현재 원본은 line_alt_bilinear_full, 2/2 sweeps, 150 attempts, relative 1e-8, absolute 1e-12이며 코드에 이 값을 baseline으로 하드코딩하지 않는다.

주요 시간은 **cold per-system selector + constructor/setup + solve의 wall-clock**이다. strong_C와 strong neural arm 모두 실제 selector 비용을 낸다. NN 생성, rejected attempt와 fallback도 포함된다. 프로세스 import, checkpoint 파일 로딩, 기존 native library 로딩, warmup 및 측정 후 진단 추출은 제외한다. 실행 전 native 초기화를 완료하며 benchmark 도중 autotune/compile을 하지 않는다. 같은 A로 준비된 solver를 multiple RHS에 재사용하면 selector/hierarchy를 재사용하지만 기본 benchmark는 매번 cold 준비 비용을 낸다.

기본 5 repeats, warmup 1회이며 case/repeat마다 arm 순서를 무작위화한다. **모든 repeat가 독립 재계산한 true residual tolerance와 attempt budget을 만족**해야 그 case를 성공으로 센다. 실패 시간은 speedup에 사용하지 않는다. 성공을 잃은 case, 구한 case, 각 쌍의 common-success 집합과 전체 deployment arm 공통 성공 집합을 모두 기록한다.

`speedup = baseline wall time / candidate wall time`이므로 1보다 크면 candidate가 빠르다. operator별 median으로 geometric mean을 계산하고 operator를 단위로 bootstrap CI를 산출한다. 반복 solve를 독립 PDE 표본으로 세지 않는다. 서로 다른 성공 cohort의 속도를 하나의 동일 조건 순위로 혼동하지 않는다.

manufactured exact solution에 대한 relative L2 및 energy error도 기록한다. 같은 residual tolerance가 같은 solution error를 보장하지 않으므로, 속도와 정확도를 따로 확인한다. `diagnostic_superiority_eligible`은 각 case의 최대 error가 strong_C보다 5%+1e-12 이상 나빠지지 않는지도 검사한다. 이 slack은 **진단 비교 기준**이며 solver stopping rule이나 기존 certificate를 완화하지 않는다. 기본으로 20개 이상 unique operator, 5 repeats, 모든 deployment arm 성공, 실제 accepted NN work, CI 하한 > 1도 요구한다. 이것이 True여도 final certificate는 아니다.

`--oracle-bank`는 각 case에서 실제로 16개/48개를 전부 실행한다. 모든 repeat가 성공한 전략 중 median time 최소값을 **empirical offline reference**로 보고한다. 탐색의 measured/warmup/합계 시간은 별도로 기록한다. oracle은 A뿐 아니라 해당 RHS/x0/budget/hardware와 bank에 의존하며, deployable selector나 이론적 상한으로 취급하지 않는다. selector/oracle time ratio와 selector 실패·oracle 성공 case로 baseline 자체의 약점도 확인할 수 있다.

## 5. 실행 명령

아래 명령은 실제 프로젝트에서 실행한다. 이미 사용한 output을 덮어쓰지 않도록 `manual_01`을 새 이름으로 바꿀 수 있다.

```bash
cd /Users/kimjiin/Desktop/adaptive_neural_multigrid_v6_7
PY=.venv/bin/python
OUT=artifacts/my_v6_7_strong_baseline/manual_01
```

회귀 테스트:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -p no:cacheprovider
```

작은 새 smoke: seven families, n=15, 4-branch + fixed continuity + 16개 bank.

```bash
"$PY" scripts/benchmark_v6_7_strong.py \
  --split smoke --rules configs/v6_7_strong_rules.json \
  --branches H_S,H_P,H_SP --oracle-bank controlled \
  --repeats 5 --output-dir "$OUT/smoke"
```

전체 tune bank를 측정하고 rules를 고정한다. 큰 격자와 전체 bank 때문에 오래 걸릴 수 있는 **수동 실행 명령**이다. 더 넓은 bank를 원하면 이 측정부터 `--oracle-bank all`을 사용한다.

```bash
"$PY" scripts/benchmark_v6_7_strong.py \
  --split tune --classical-only --oracle-bank controlled \
  --rules configs/v6_7_strong_rules.json --repeats 5 \
  --output-dir "$OUT/tune_bank"

"$PY" scripts/calibrate_v6_7_strong_rules.py \
  --input-run "$OUT/tune_bank" --rules configs/v6_7_strong_rules.json \
  --minimum-leaf-cases 3 --output-dir "$OUT/calibration"
```

보정한 rules로 새 seed의 35개 문제에서 비교한다. 이 명령도 자동 실행하지 않았다. seed/분포/평가 기준을 실행 전에 고정하고 결과를 본 뒤 같은 집합으로 보정하지 않는다. 기본 코드는 original train/tune/audit와 normalized operator 중복을 제거한다. 다른 별도 실행의 fresh/smoke 데이터까지 자동 탐색해 중복 검사하지는 않는다.

```bash
"$PY" scripts/benchmark_v6_7_strong.py \
  --split fresh --sizes 15,31,63,127,255 --per-size 7 --seed 20270915 \
  --rules "$OUT/calibration/selector_rules.json" \
  --branches H_S,H_P,H_SP --repeats 5 \
  --output-dir "$OUT/fresh_validation"
```

기존 audit 35개 재평가도 지원한다. 정확히 저장된 specification/seed로 재구성하며 **previously inspected audit의 diagnostic 재평가**로 표시한다. 아래에는 비용이 큰 offline oracle도 포함된다.

```bash
"$PY" scripts/benchmark_v6_7_strong.py \
  --split audit --rules "$OUT/calibration/selector_rules.json" \
  --branches H_S --oracle-bank controlled --repeats 5 \
  --output-dir "$OUT/audit_diagnostic"
```

중단된 benchmark는 **모든 원래 인자에 `--resume`을 추가**한다. config/checkpoint/rules/source/environment/입력 digest가 달라지면 재개를 거부한다. 완료 repeat를 hash와 함께 저장하며 실패 repeat만 골라 제거하거나 재측정하지 않는다. 보정은 완료된 bank run만 받는다. `--checkpoint`는 모든 요청 branch에 명시한 단일 checkpoint를 사용하는 추가 ablation용이며 기본은 smoother/transfer/joint stage별 보존 후보다.

## 6. Python에서 사용

기존 `AdaptiveConfig`와 `Components`를 준비한 뒤 다음 entry point를 사용한다.

```python
from dataclasses import replace
from adaptive_mg.v67.strong import PreparedStrongMG, load_strong_rules

rules = load_strong_rules("artifacts/my_v6_7_strong_baseline/calibration/selector_rules.json")
classical = PreparedStrongMG(A, n, config=replace(cfg, mode="classical", branch="C"), rules=rules)
result_c = classical.solve(b)

neural = PreparedStrongMG(
    A, n, components=smoother_components,
    config=replace(cfg, mode="research", branch="H_S", spatial=False, gate_mode="open"),
    rules=rules,
)
result_s = neural.solve(b)
print(neural.selection.to_dict())
```

`PreparedStrongMG`가 기존 solver를 확장하므로 A/model/config 변경 시 기존 bank invalidation을 따르며 rules 변경도 검출한다. `selection`에는 features, rule evidence, 선택 strategy, selector 시간, rules digest가 있다. same-A multiple RHS의 추가 solve에서는 selector를 반복 실행하지 않는다.

## 7. 이번에 실행한 작은 검증

원래 35-case direct audit의 H_S 1.26449×는 fixed C 대비였으며 원본 그대로 보존했다. 아래 작은 집합과 표본·seed가 다르므로 숫자만 빼서 baseline 효과를 추정하면 안 된다.

보정 전 기본 rules로 새 n=15 seven-family smoke를 5 repeats 실행했다. 6개 deployment arm은 모두 7/7 성공했다. setup/selector 포함 결과는 다음과 같다.

| 비교 | Geometric speedup | 95% CI |
|---|---:|---|
| strong_C vs fixed_C | 1.226× | 상세 JSON 참조 |
| strong_H_S vs strong_C | 0.854× | [0.612, 1.067] |
| strong_H_P vs strong_C | 0.678× | [0.568, 0.788] |
| strong_H_SP vs strong_C | 0.674× | [0.468, 0.890] |

따라서 이 smoke는 neural 우위를 보여주지 않는다. 모두 tolerance를 만족했지만 L2/energy error까지 항상 strong_C보다 작지는 않았다. 기본 selector의 oracle 대비 시간 비율도 case별 약 1.23~7.78이어서 bank 내 개선 여지가 있다.

이후 사전 기록한 계획대로 original tune 첫 7개(n=15), classical-only, controlled bank, 3 repeats를 사용한 **제한적 보정**을 수행하고, 새 seed의 n=15/31 총 14개에서 5 repeats로 C와 H_S를 검증한다. 작은 tune에서는 heterogeneous leaf 4개만 최소 표본 수를 채웠고 다른 leaf는 기본값을 유지했다. 최종 적용한 코드의 결과와 원본 보존 hash 검증은 `artifacts/my_v6_7_strong_baseline/completion_report.json`, 새 문제 결과는 `fresh_validation/comparison.json`에 기록한다. 이 단계도 전체 tune 기반 강한 baseline 검증이나 독립 최종 인증을 대체하지 않는다.

## 8. 결과 파일과 변경 범위

- `run_manifest.json`: 전체 config, source/checkpoint/rules/environment hash, A/b/x0/exact digest, seed, 비용 범위.
- `raw_results.json`, `repeats/`: 실제 모든 repeat와 순서, residual/error, timing, counters, selector evidence, branch/P bank/hierarchy.
- `comparison.json` / `.csv`: arm별 성공/실패, pair/common cohort speedup/CI, 정확도, oracle 및 selector 품질.
- `calibration/selector_rules.json`, `calibration.json`: 고정된 runtime rules와 offline 선택 근거.
- `inspection/`: 기존 연구 결과 진단 및 변경 전 보존 hash.
- `patch/`: 새 파일 patch와 변경 파일 목록.
- `logs/`, `completion_report.json`: 테스트/실행 로그와 완료 결과.

추가 파일은 strong selector/evaluation/calibration 모듈 3개, CLI 2개, rule config, 관련 테스트 4개, 이 설명서다. 기존 학습 코드, stage acceptance, final certificate와 fixed solver 구현은 수정하지 않는다. NN 재학습, architecture/loss/precision/thread 변경, dependency upgrade, full audit 자동 실행은 하지 않았다.
