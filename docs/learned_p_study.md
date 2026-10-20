# Line_alt 고정 + learned P

현재 작업 브랜치는 **`residual-u-net-teacher`**다. 이름은 이전 U-net 실험에서
이어졌지만 현재 주 경로에서는 U-net teacher를 사용하지 않는다.
[실행 안내](learned_p_runbook.md) · [구현 위치](ARCHITECTURE_KR.md) · [문서 목록](README.md)

이 실험은 NN smoother를 제거하고 `line_alt` pre/post smoothing을 고정한다.
새 기본 설정과 `run_v6_7_learned_p_study.py`가 이 경로를 사용한다. 이전 joint
실험은 명시적인 이전 설정 파일을 사용할 때만 재현할 수 있다. 기존 run을
새 설정으로 바꾸거나 freeze/hash를 수정하지 않는다.

## 실행하는 연산

| 항목 | 현재 설정 |
| --- | --- |
| Coarsening | full 고정 |
| Pre/post smoother | classical line_alt, 각각 2회 |
| NN smoother | 없음: 파라미터·저장된 smoother 가중치·추론 모두 0 |
| 학습 대상 | width8 residual CNN이 생성하는 P 가중치만 |
| 초기 P·fallback | bilinear |
| Restriction | R=P.T |
| Coarse 행렬 | 실제 learned P로 Ac=P.T@A@P |
| 최하위 풀이 | LU |
| 학습 level | 최하위 직접 풀이를 제외한 모든 level |
| 독립 reference | line_alt + EM5 + full |

CNN은 3,208개 파라미터이며 P의 허용된 16개 후보 슬롯을 출력한다. 활성 연결은
bilinear support로 제한하고, 실제 경계 행합과 coarse 대표점 injection을 보존한다.
Galerkin으로 내려간 실제 Ac를 다음 level CNN에 입력하며 이를 미분한다.
NN smoother 객체 대신 zero-parameter marker를 저장하므로 P-only checkpoint에는
CNN smoother 가중치가 없다. 기존 Components 컨테이너의 비활성 detector/controller는
호환성을 위해 남지만 학습·추론하지 않는다.

## PDE와 학습 데이터

대상은 단위 정사각형에서 homogeneous Dirichlet 경계를 갖는 합성 확산 문제다.

\[
-\nabla\cdot(K(x,y)\nabla u)=f,\qquad u|_{\partial\Omega}=0.
\]

양의 확산 tensor를 사용하는 P1 삼각형 유한요소법으로 SPD 행렬 A를 조립한다.
n은 방향별 내부 자유도 수이고 미지수 수 N=n²다. 데이터 family는 near_isotropic,
aligned_anisotropy, rotated_anisotropy, local_patch, checkerboard, channel,
variable_coefficient의 7개다. 이방성·대비·각도·위치 등의 매개변수로 서로 다른 A를 만든다.
`channel`은 계수장의 띠 모양을 뜻하며 유체 채널 시간 적분이 아니다.

직접 학습에는 정답 P label이 없다. A에서 생성한 계층을 무작위 오차 probe에 적용해
아래 손실로 P 생성기를 갱신한다. 평가에서는 이산 벡터 u_*와 b=A u_*를 만들어
정확해를 아는 선형계를 푼다. RHS count 1/4/16은 **같은 A로 서로 다른 b를 몇 개
푸는가**이며 b의 스칼라 값이 아니다. 분할 간 A 중복은 정규화된 연산자 digest로
검사한다. Final/OOD는 개발 학습·선택에서 제외한다.

| 구성 | 학습 격자 n / A 수 | Validation n / A 수 | 직접 P updates |
| --- | --- | --- | ---: |
| 저장소 smoke JSON | 7 / 7개 | 7 / 7개 | 14 |
| 저장소 research JSON | 15,31,63 / 84개 | 15,31,63,127 / 28개 | 336 |
| 확대 실행안: 별도 로컬 JSON | 15,31,63 / 420개 | 15,31,63,127 / 140개 | 4,200 |

확대 실행안은 family·격자당 train 20개, validation 5개다. 저장소 research 기본값을
바꾸지 않는다. 현재 요청은 직접 P 학습의 데이터를 늘려 먼저 평가하는 것이므로
[runbook](learned_p_runbook.md)은 search/distill 없이도 가능한 평가 경로를 제공한다.
격자 OOD n=255는 최종 평가용으로 남긴다. 이 예산이 충분하거나 빨라질지는 아직 미확인이다.

## Spectral radius를 목표로 Frobenius 손실을 쓰는 이유

고정 A와 setup에서 생성한 P에 대해 raw V-cycle은 선형이다. 따라서 오차는

\[
e_{k+1}=E_\theta e_k
\]

로 전달된다. 이 선형 반복의 점근 수렴을 지배하는 것은 spectral radius이고,
정확한 행렬 norm 사이에는 다음 관계가 있다.

\[
\rho(E_\theta)\leq\|E_\theta\|_2\leq\|E_\theta\|_F.
\]

Greenfeld et al.과 Luz et al.은 two-grid의 spectral radius를 목표로 하되
실제 학습에는 squared Frobenius norm을 사용했다. 본 구현은 그 목적을 **실제 전체
multilevel V-cycle**의 오차 연산자로 옮긴다. 논문과 동일한 아키텍처/성능이라는
의미는 아니다.

- https://proceedings.mlr.press/v97/greenfeld19a/greenfeld19a.pdf
- https://proceedings.mlr.press/v119/luz20a/luz20a.pdf

하지만 큰 E 행렬을 구성하면 N×N 메모리가 필요하다. 이 구현은 독립적인 ±1
Rademacher 벡터 z의 공분산이 I인 성질을 사용한다.

\[
\mathbb E_z\|E_\theta z\|_2^2=\|E_\theta\|_F^2,\qquad
\widehat L_F=\frac1{qN}\sum_{j=1}^{q}\|E_\theta z_j\|_2^2.
\]

N은 미지수 수이며, /N은 격자마다 손실 크기를 정규화한다. 기본 q는 8이다.
`b=0`, 초기 오차 `x=z`에서 **한 번의 전체 V-cycle**을 실행한 출력이 E z다.
이 경로에서 line_alt, P, Ac, 하위 line solve, 최하위 직접 풀이까지 미분한다.
학습은 항상 한 cycle의 E를 사용한다. 이전 residual 학습의 3→6 cycle curriculum과
시간/SPSA 손실을 이 기본 경로에 섞지 않는다.

Frobenius는 spectral radius의 대리 목표다. Norm bound는 정확한 Frobenius norm에
대한 것이며 유한 표본의 추정값이 매번 spectral radius의 상한이라는 뜻은 아니다.
Frobenius가 줄어도 실제 cycle 수·총시간이 반드시 개선되지는 않는다. 특히 fallback과
classical tail을 포함하는 배포 solver 전체가 하나의 고정 E인 것은 아니다.

## 직접 학습, P 탐색, 증류

`train`은 `learned_P` 하나만 학습한다. `learned_S`, `joint` 단계는 실행하지 않는다.
직접 학습에는 smoother 또는 residual loss가 없고 위 Frobenius 손실만 사용한다.

선택적 `search`도 P의 허용 원소만 추가 최적화한다. 초기값은 직접 학습 P,
bilinear P, 그리고 직접 학습 출력에 작은 변화를 준 후보들이다. Smoother는
모든 후보에서 line_alt 그대로다. 학습 A마다 기본 4 starts×200 updates를 사용한다.
P 갱신 때 실제 Ac와 하위 계층을 다시 구성한다. 후보의 초기/최적화 상태를
같은 held-out probe와 fallback 없는 실제 시간으로 비교한다.

`distill`은 선택된 P oracle을 작은 CNN에 옮긴다. Frobenius task loss에 P 원소
matching과 전체 V-cycle 보정 action loss를 더하고, 이후 P CNN만 task 미세조정한다.
Teacher와 student 어느 쪽에도 NN smoother는 없다. 이는 NN smoother와 P의 공동
학습이 아니다. 유효하지 않은 탐색 후보는 기록하고 label로 쓰지 않는다.

| Arm | P | Smoother |
| --- | --- | --- |
| fixed_C | bilinear | line_alt |
| em5_C | EM5 | line_alt |
| learned_P | Frobenius 직접 학습 | line_alt |
| distilled_P | P-only 탐색 후 증류 | line_alt |

EM은 em5_C에서만 계산한다. 학습 setup이나 fallback 준비에는 EM을 호출하지 않는다.
최종 후보는 동일 tolerance에서 실측 setup+solve 시간으로 선택하고, 실패한 문제를
버려서 성공 cohort만으로 순위를 매기지 않는다.

일반 `benchmark` CLI는 위 네 비교군의 checkpoint를 요구하므로 `train`만 끝낸
run에서는 아직 실행할 수 없다. 직접 학습 직후의 세 비교군 평가는
[runbook의 기존 API 호출](learned_p_runbook.md#직접-p의-세-비교군-평가)을 사용한다.
그 결과는 development validation이고 기존 네 비교군 `freeze/final` 증거를 대신하지 않는다.

## 현재 fallback 조건

현재 H_P 경로는 learned controller나 classical shadow trajectory를 사용하지 않는다.
Stopping threshold는 τ=max(1e-12, 1e-8·||r₀||₂)다.

- Learned 후보의 ||r_new||₂ / ||r_old||₂가 `1 + 1e-12` 이상이거나 비정상 값이면 거부한다.
- 처리 가능한 learned setup/실행 오류도 같은 복구 경로로 들어간다.
- 거부된 후보를 버리고 이전 승인 해에서 bilinear classical cycle을 시도한다.
- 현재 잔차가 `10τ` 이하이면 classical로 마무리한다.
- `CLASSICAL_LOCK` 이후 같은 RHS에는 learned branch로 돌아오지 않는다.

느리지만 계속 감소하는 후보를 classical보다 느리다는 이유로 교체하지는 않는다.
현재 명시적 H_P 실행에서는 일반 auto-policy의 stagnation/break-even 설정을 쓰지 않는다.
최대 150 attempts 내에 못 풀면 실패이며, 거부된 시도와 복구 시간도 비용에 포함한다.
따라서 fallback이 있다는 사실만으로 수렴이나 총시간 우위가 보장되지는 않는다.
`rollback=0`이어도 near-tolerance classical tail이 있을 수 있으므로 accepted neural
cycle과 classical 사용도 함께 읽는다. 학습·raw 고정-cycle 진단에는 fallback이 없다.

## 학습 전후 증거와 재사용

`train` 뒤에 `diagnose`를 실행하면 search/distill 없이도 직접 학습의 효과를 확인한다.
TRAIN과 별도 validation의 같은 A·동일한 고정 error tensor·한 cycle에서 초기 P와
학습 P의 Frobenius 추정값을 비교한다. 초기 P는 bilinear이고 smoother도 동일하다.
진단 기본 probe 수는 16이다. 초기/학습 모델의 같은 RHS에서 실제 tolerance까지
setup+solve 시간, cycle, rollback, accepted neural cycle, setup failure도 별도로 측정한다.
`diagnostics/p_frobenius.json`에 저장하며 이 진단으로 checkpoint를 선택하지 않는다.
최종/OOD를 이 과정에서 사용하지 않는다. 콘솔의 `[P DIAG]` improved 개수는
validation만 집계하며, 통계적 유의성 판정 없이 수치가 감소한 사례 수다.

여기서 before의 시간은 **초기 CNN이 bilinear P를 생성하는 P-only 경로**의 시간이다.
순수 classical보다 빨라졌는지는 이 before/after 시간만으로 판단하지 않는다.
순수 classical과의 공정한 총시간 비교는 benchmark의 fixed_C와 수행한다.
초기 zero-head P가 정확히 bilinear이면 기존 Ac/factor를 재사용하는 빠른 경로를
타지만 학습 P가 달라지면 새 Ac/factor가 필요하다. 진단 시간 차이가 모두 CNN
추론 때문이라는 뜻은 아니다. 진단은 before 다음 after 순서로 측정하므로 작은
차이의 의미는 반복 평가와 별도 benchmark로 확인한다.

같은 A에서는 이미 적극적인 정확 재사용을 한다.

1. 각 level의 P를 한 번 생성한다.
2. R, Ac, line factors와 LU를 준비한다.
3. 같은 A의 모든 cycle과 RHS에서 저장한 연산자를 재사용한다.

따라서 같은 PreparedAdaptiveMG에 여러 RHS를 전달하면 매 cycle/RHS CNN을
재실행하지 않는다. `solve_many`는 batch당 freshness 검사도 한 번 수행한다.
한편 benchmark의 cold repeat는 이와 별도로 새 setup부터 다시 측정한다.

A가 바뀌면 현재는 P를 다시 생성하고 모든 수치 계층을 갱신한다. 새 A에 대해
이전 P를 계속 쓰는 방법은 가능하지만 별도 approximate-reuse 정책이 필요하다.
그때도 Ac=P.T@A_new@P, line factor, LU는 새 A에 맞게 갱신해야 한다.
P와 함께 오래된 Ac/factor까지 재사용하면 다른 선형계를 푸는 오류가 된다.
현재 구현은 A 변경을 감지하면 정확하게 재생성하며, stale P 재사용은 추가하지 않았다.

Smoother와 P를 각각 학습한 뒤 공동 미세조정하는 방법은 향후 별도 ablation이 될
수 있다. 다만 먼저 P-only가 line_alt와 잘 결합하는지 확인하고, NN smoother가
line_alt 대비 이득을 보인 뒤에 검토하는 것이 실험 원인을 분리하기 좋다.
현재 기본 경로에 NN smoother나 joint fine-tuning을 다시 넣지 않는다.

## 실행과 결과 확인

전체 명령은 [Mac 실행 안내](learned_p_runbook.md)에 모았다.

1. 새 run을 prepare하고 `train → diagnose`로 직접 P를 확인한다.
2. `fixed_C / em5_C / learned_P` 세 비교군을 같은 A·RHS·tolerance에서 평가한다.
3. 손실·cycle·fallback 없는 raw contraction·총시간을 구분해서 읽는다.
4. 직접 학습의 효과가 확인되면 선택적으로 P-only `search → distill`과 네 비교군 평가를 한다.
5. 연구 run의 정식 validation 증거와 모델을 freeze한 후 final/OOD를 연다.

같은 checkpoint/설정/환경의 중단된 작업만 resume한다. 기존 run의 설정이나 manifest
hash를 수정해 새 예산을 끼워 넣지 않는다. Smoke 성공은 연구 규모의 속도 개선 증거가 아니다.
