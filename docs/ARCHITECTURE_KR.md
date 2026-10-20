# 현재 P-only 구현 위치와 데이터 흐름

브랜치 `residual-u-net-teacher`의 현재 기본 경로다.
[설계](learned_p_study.md) · [실행 안내](learned_p_runbook.md) · [문서 목록](README.md)

## 먼저 읽을 파일

아래 경로는 저장소 루트 기준이다.

| 기능 | 파일 |
| --- | --- |
| 현재 CLI 진입점 | [scripts/run_v6_7_learned_p_study.py](../scripts/run_v6_7_learned_p_study.py) |
| 설정 검증·prepare/train/diagnose·선택적 search/distill | [src/adaptive_mg/v67/joint_transfer_study.py](../src/adaptive_mg/v67/joint_transfer_study.py) |
| P-only 모델·Frobenius 손실·고정 probe 진단 | [src/adaptive_mg/v67/learned_p.py](../src/adaptive_mg/v67/learned_p.py) |
| CNN 본체·출력층 / support 계약 | [models.py](../src/adaptive_mg/v67/models.py), [research_controls.py](../src/adaptive_mg/v67/research_controls.py) |
| 합성 PDE·분할·중복 검사·final 봉인 | [research_data.py](../src/adaptive_mg/v67/research_data.py), [pde.py](../src/adaptive_mg/pde.py) |
| P projection·복잡도 검사 | [research_transfer.py](../src/adaptive_mg/v67/research_transfer.py), [transfer.py](../src/adaptive_mg/transfer.py) |
| 실제 Galerkin 계층을 통과하는 자동미분 | [unroll.py](../src/adaptive_mg/v67/unroll.py), [autograd_sparse.py](../src/adaptive_mg/v67/autograd_sparse.py) |
| 수치 P/R/Ac·line/LU 생성 및 캐시 | [banks.py](../src/adaptive_mg/v67/banks.py) |
| solve/solve_many·잔차 검사·rollback·classical 마무리 | [solver.py](../src/adaptive_mg/v67/solver.py) |
| 고정 classical 정의·수치 smoothing | [fixed_classical.py](../src/adaptive_mg/v67/fixed_classical.py), [smoothers.py](../src/adaptive_mg/smoothers.py) |
| 반복 시간 측정·제조해·CSV·측정 재개 | [research_evaluation.py](../src/adaptive_mg/v67/research_evaluation.py) |
| fallback 없는 fixed-cycle 검사·cycle 비교 보고 | [joint_transfer_evaluation.py](../src/adaptive_mg/v67/joint_transfer_evaluation.py) |

`joint_transfer_study.py`라는 파일명은 공유 workflow의 이름이다. P-only version 설정에서
`train`이 실행하는 stage는 `learned_P` 하나이며, `distill`도 P만 학습한다.

## 오프라인 학습

각 학습 A에 대해 CNN이 실제 A_l의 정규화된 특징에서 각 level의 P_l를 생성한다.
Projection 뒤의 P_l로 A_(l+1)=P_lᵀ A_l P_l를 구성하고, 다음 level에도 실제 A_(l+1)를
입력한다. `b=0`, 초기 오차 z에서 raw V-cycle을 한 번 실행해 E z를 계산한다.
P와 Galerkin 구성, 하위 line solve 및 최하위 solve를 통과해 P 생성기로 역전파한다.

Line smoother의 알고리즘과 sweep 수는 고정이다. 다만 하위 A_l가 P에 따라 달라져
그 A_l에 대한 line 연산도 달라지므로, 이를 고정된 옛 coarse 행렬로 대체하지 않는다.
학습과 raw 진단에는 fallback이 없고, 실패한 learned 연산을 classical 성능으로 대체하지 않는다.

## 온라인 setup과 재사용

`PreparedAdaptiveMG`는 현재 A의 FP64 복사본과 bilinear classical 계층을 먼저 준비한다.
P-only branch가 필요해지면 각 nonterminal level에서 CNN을 한 번 실행하고 learned
P/R/Ac 및 해당 수치 분해를 준비한다. 실제 smoothing은 classical `line_alt`다.

같은 prepared 객체와 같은 A/model/config에서는 모든 cycle과 RHS가 이 계층을
재사용한다. Freshness 검사는 `solve`당 한 번, `solve_many` batch당 한 번 수행한다.
매 cycle에 전체 모델 hash를 다시 계산하거나 CNN을 실행하지 않는다. A 변경 시에는
P와 수치 계층을 재생성한다. Cold 측정은 매번 새 prepared 객체를 구성한다.

초기 zero-head CNN이 정확히 bilinear P를 내면 `generated_p`는 기존 P 객체를 반환한다.
이때 기존 Ac와 수치 분해도 재사용할 수 있다. 학습 후 P가 달라지면 새 Ac와 필요한
분해를 계산한다. 따라서 초기/학습 모델의 setup 비용이 항상 같지는 않다.

## 현재 실행되는 제어

설정은 `mode=research`, `branch=H_P`, `gate_mode=open`, `spatial=false`,
`use_smoother=false`, `use_learned_controller=false`다. 범용 컨테이너에는 비활성
controller/detector가 남지만 이 경로에서 학습하거나 실행하지 않는다.

- Learned cycle 후보의 잔차 비율이 `1 + 1e-12` 이상이거나 잔차가 비정상이면 거부한다.
- 처리 가능한 learned 생성/실행 오류도 거부·복구 경로로 들어간다.
- 거부된 후보를 채택하지 않고 이전 승인 해에서 classical cycle을 시도한다. 잔여 cycle 예산이 없으면 실패로 종료한다.
- 잔차가 stopping threshold의 10배 이하이면 classical로 마무리한다.
- `CLASSICAL_LOCK` 이후 같은 RHS에서는 learned branch로 돌아오지 않는다. 다음 RHS는 새 정책 상태로 시작한다.

`controller.py`의 일반 auto-policy와 설정의 break-even/stagnation 파라미터는 현재
명시적 H_P 경로의 전환 판정에 사용되지 않는다. 매 cycle classical shadow solve도 없다.
최대 150 attempts에서 성공하지 못하면 실패를 기록한다. Fallback은 시간 우위나
무조건적인 수렴 보장이 아니다.
