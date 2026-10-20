# Residual U-net teacher → 작은 CNN student로 P 학습

> **과거 실험·구현 기록.** 아래의 기본값·명령·결과는 이 문서가 다루는 경로에 한정된다.
> 현재 `residual-u-net-teacher`의 주 실험은 [line_alt 고정 + P-only 학습](learned_p_study.md)이다.
> 현재 실행은 [Mac runbook](learned_p_runbook.md), 경로별 구분은 [문서 목록](README.md)을 따른다.

이 독립 U-net 실험은 **희소 prolongation을 생성하는 NN**을 비교하는 독립 연구 경로다.
Teacher가 validation에서 실제 수렴 개선을 보인 경우에만 증류한다.
최종 목표는 student의 생성·Galerkin 구성까지 포함한 총 풀이 시간이 고정 classical
MG보다 짧아지는 것이다. 구현이나 smoke 통과가 속도 향상 증거는 아니다.

## Application 범위와 학습 데이터

Sloshing·전단 혼합층·Navier–Stokes 시간 적분기는 이 브랜치에 없다.
외부 CFD finalized-LDU recorder/importer, native 순서 변환, GAMG 결과 비교와
해당 CLI 옵션도 제거했다. 합성 PDE용 H_S/world-model solver 실험은 유지한다.
기존 validation JSON은 과거 검증 이력이며 삭제된 기능의 현재 제공을 뜻하지 않는다.

현재 U-net 경로는 `research_data.py`의 합성 자료를 사용한다. PDE는

\[
-\nabla\cdot(K(x,y)\nabla u)=f\quad\text{in }(0,1)^2,
\qquad u|_{\partial\Omega}=0.
\]

정사각형 격자의 각 cell을 두 삼각형으로 나눈 **P1 유한요소법**으로 stiffness
matrix A를 조립하고 Dirichlet 경계 자유도를 제거한다. n은 방향별 내부 자유도 수이며
A의 크기는 n²×n²다. 양의 확산 tensor를 사용하므로 이 문제군의 A는 SPD다.

| 계수장 family | 의미 |
| --- | --- |
| near_isotropic | 거의 등방성인 일정 계수 |
| aligned_anisotropy | 격자 방향으로 정렬된 이방성 |
| rotated_anisotropy | 격자와 비스듬한 이방성 |
| local_patch | 국소 고대비 사각형 영역 |
| checkerboard | 체커보드 고대비 영역 |
| channel | 대각선 띠 모양의 고전도 영역; 유체 채널 해석이 아님 |
| variable_coefficient | 공간적으로 매끄럽게 변하는 양의 계수 |

기본 이방성 tensor는 Q(θ) diag(1, ε) Q(θ)^T다. 난수 seed로 이방성 강도,
각도, 대비, 패치 위치·폭, 패턴 규모 등을 정하고 A에 양의 전체 scale도 곱한다.

각 A에 대해 sin(πx)sin(πy)에 난수 계수의 사인 모드 4개를 더한 벡터 u_*를 만든다.
그 후 **b=A u_***로 우변을 생성한다. 따라서 u_*는 만들어진 **이산 선형계의 정확해**다.
연속 PDE의 f를 해석적으로 적분한 load vector라고 해석하면 안 된다.
유동 해석이나 별도 고정밀 PDE solve로 정답 snapshot을 수집하는 방식이 아니다.
Research 설정의 `rhs_variants=4`는 A마다 이 기본 우변과 추가 사인 모드 조합 3개를
사용하게 한다. 추가 해 벡터는 각각 사인 모드 5개를 더하며 역시 b=A u_*로 생성한다.
학습 반복에서 우변을 순환 선택하므로 A 개수가 4배 늘어나는 것은 아니다.

NN 입력은 A에서 추출한 10채널: 중앙값으로 정규화한 대각 성분의 로그 1채널과
각 행의 절대 대각 성분으로 나눈 국소 3×3 stencil 9채널이다. b나 u_*는 NN 입력이 아니다.
NN 출력으로 제약을 만족하는 희소 P를 만들고 R=P^T, A_c=P^TAP로 실제 MG를 구성한다.
현재 설정은 0 초기값에서 3회 V-cycle 후 각 cycle의 정규화 잔차 로그를 가중합한다.
여기에 안정성·복잡도·제약·classical 대비 악화 억제 항을 더해 역전파한다.
u_*는 해 오차 진단에 사용할 수 있으며, 주 학습 목표는 u_*를 회귀하는 MSE가 아니다.
EM으로 만든 P는 동일 제약의 **기준 P**이지 지도학습용 최적 P 정답 label이 아니다.
Teacher가 gate를 통과하면 student에는 별도로 teacher V-cycle 보정 동작을 맞추는 KD loss가 붙는다.

`configs/v6_7_unet_transfer_research.json` 기준:

| 분할 | 격자 | family당/격자당 A | A 개수 |
| --- | --- | --- | --- |
| train | 15², 31², 63² | 8 | 168 |
| validation | 15², 31², 63² | 2 | 42 |

`prepare`가 `dataset_plan.json`과 `development_manifest.json`에 계수·seed·hash를
기록하고 같은 정의로 A/b/u_*를 재구성한다. Train/validation 및 final/OOD 분리는
정규화한 operator digest와 독립 seed로 검사한다. Final/OOD는 freeze 후 별도로 생성한다.
위 수치는 연구 **설정**이며 연구 규모 학습을 완료했다는 뜻은 아니다.

## 모델과 수치 연산

| 항목 | Direct / distilled student | Teacher |
| --- | --- | --- |
| 입력 | A의 10채널 특징 | 동일 |
| 구조 | 기존 TransferNet의 residual5 CNN, hidden 8 | Residual U-net, depth 2 |
| 기본 폭 | 8 | 본 연구 16 → 32 → 64, smoke 4 → 8 → 16 |
| 출력 | fine 점별 coarse 후보 16개의 delta | 동일 |
| 사용 시점 | 새 A에 대한 P 생성 후 캐시 | 학습과 명시적 offline 평가만 |
| Attention | 없음 | 없음 |

Teacher의 각 블록은 `shortcut(x) + Conv(GELU(Conv(x)))`이며 채널이 바뀌면
shortcut에 1×1 convolution을 사용한다. Encoder에서는 stride-2 convolution으로
축소하고 decoder에서는 skip 특징맵의 정확한 크기로 bilinear resize한 뒤 채널을
연결한다. 따라서 7/15/31처럼 홀수 크기나 semi-coarsening의 직사각형을 처리한다.
U-net 내부의 축소·복원은 **P 가중치를 생성하기 위한 특징 처리**다.
수치 MG의 restriction/prolongation은 아래의 희소행렬 연산으로 수행한다.

각 numerical level에서 frozen classical parent의 P를 기준으로 사용한다.
NN 입력은 실제 learned hierarchy의 A_l이고, 출력 delta는 공통 projection을 거친다.

\[
P_l=P_{C,l}+\Delta P_\theta(A_l),\quad
R_l=P_l^T,\quad A_{l+1}=P_l^T A_lP_l.
\]

Parent에 존재하는 연결만 허용하고, parent 행합과 coarse-node injection을
보존한다. 경계의 행합을 무조건 1로 만들지 않는다. Teacher/direct/KD 모두 같은
support·projection·complexity cap을 쓴다. Zero-initialized head는 정확히 classical
hierarchy에서 시작한다. Default는 기존 support-preserving softmax delta이며,
새 run의 `parameterization: affine`으로 동일 제약의 affine delta도 선택 가능하다.
P와 coarse A의 nonzero 수, hierarchy complexity 제한은 실제 sparse 값으로 검사한다.

## 학습과 teacher 판정

이 연구 경로에서는 classical portfolio 검색·selector calibration·A별 전략 선택을
제거했다. 모든 A와 모든 단계에서 아래의 같은 numerical contract를 사용한다.

| 설정 | 고정값 |
| --- | --- |
| Strategy ID | `line_alt_energymin_full__em5__v22` |
| Smoother | x/y alternating line smoother |
| Pre / post smoothing | 각 2회 |
| Coarsening | full |
| P 생성 | 같은 희소 연결·행합·injection 제약의 energy minimization |
| EM 반복 | level별 projected CG 예산 5회, `em_fixed_iterations: true` |
| EM 수렴 진단 허용오차 | `em_rtol: 1e-8` |
| R / coarse A | `R=P.T`, `Ac=P.T @ A @ P` |

고정 반복 모드는 상대 허용오차로 조기 종료하지 않는다. 시작점이 이미 정지점이거나
CG 잔차가 FP64 기계 정밀도에 도달하면 0으로 나누는 계산을 피하기 위해 종료한다.
이때 실제 반복 횟수와 `stop_reason`을 기록하며 실행하지 않은 반복을 세지 않는다.
5회는 완전히 수렴한 energy minimum을 보장하는 값이 아니라 **고정 setup 예산**이다.

`prepare`가 `fixed_classical.json`에 전체 MG numerical contract를 저장하고,
normalized-A digest가 겹치지 않는 train/validation 자료를 새로 생성한다.
A가 다르면 이 고정 알고리즘으로 계산한 P도 달라진다. NN의 baseline P는 해당 A에서
생성한 frozen classical hierarchy의 P이며, teacher/direct/KD 모두 동일하다.
기존 strong 기준 데이터·체크포인트·gate를 이 v2 run에 재사용할 수 없다.

기존 H_S/world-model 등 별도 solver 실험이 사용하는 legacy strong 모듈은 호환성을 위해 남아 있다.
이 U-net 연구의 import·prepare·학습·평가·fallback 경로에서는 해당 모듈을 사용하지
않으며, strong baseline arm·selector rules·calibration 설정/명령도 제공하지 않는다.

1. `direct`: 작은 CNN을 실제 V-cycle residual loss로 학습한다.
2. `teacher`: residual U-net을 같은 numerical constraints와 residual loss로 학습한다.
3. `teacher-check`: 동일 validation A/RHS/초기값에서 3회 **raw numerical V-cycle**
   후 진짜 residual 비율을 비교한다. 이 경로는 fallback으로 teacher를 대체하지 않는다.
   별도 safeguarded solve에서 실제 tolerance와 teacher 사용 여부도 확인한다.
4. Teacher의 residual 비율이 **direct CNN과 고정 classical C 모두 대비 기하평균 5% 이상
   감소**하고, 최소 사례 수·feasibility·새 실패 없음·실제 neural 사용 조건을 만족해야
   `teacher_gate.json`의 `adopted`가 true다. Teacher 추론 시간은 이 판정의 기준이 아니다.
5. `distill`: gate가 false면 사유만 기록하고 checkpoint를 만들지 않는다. True면
   direct와 **동일한 초기 가중치·작은 CNN 구조·train 순서·총 update 수**로 학습한다.
   앞부분은 MG loss + KD, 뒷부분은 task-only fine-tuning이다.

본 연구 direct/KD는 각 1,120 updates(앞 840 KD, 뒤 280 task-only), teacher는 2,240
updates다. Smoke는 각각 8 updates이며 teacher capacity나 최종 성능을 판단할 수 없다.
Student가 teacher와 같은 긴 범위의 의존성을 표현할 수 있다는 보장도 없다.

MG loss는 3-cycle의 가중 log residual에 stability·feasibility·no-harm·sparse-cost
항을 더한다. Teacher와 student의 실제 P, Galerkin A와 coarse solve를 통해 미분한다.
KD는 P 원소 MSE 대신 deterministic random error probe에서 만든 r=Ae에 대해
전체 V-cycle의 보정값을 맞춘다.

\[
\mathcal L_{KD}=\mathbb E_r
\frac{\|V_S(0,r)-\operatorname{stopgrad}(V_T(0,r))\|_A^2}
{\|V_T(0,r)\|_A^2}.
\]

Teacher 출력은 detach하며 probes/KD/teacher는 배포 시 실행하지 않는다.
이 loss는 유한 probe에 대한 보조 목적함수이며 수렴 보증이나 spectral-radius 계산이 아니다.
기존 연구 경로의 기본 `weights` KD는 유지되고 이 새 연구에서만 `cycle_action`을 선택한다.

## 설치와 smoke 실행

저장소에서 실행한다. 기존 작업 변경분은 보존하고 이 브랜치를 checkout한다.

```bash
git fetch origin
git switch residual-u-net-teacher
git pull --ff-only origin residual-u-net-teacher
python3 -m pip install -e '.[dev]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
RUN="artifacts/unet_transfer_smoke_$(date +%Y%m%d_%H%M%S)"
python3 scripts/run_v6_7_unet_transfer_study.py prepare --config configs/v6_7_unet_transfer_smoke.json --run-dir "$RUN"
python3 scripts/run_v6_7_unet_transfer_study.py train --run-dir "$RUN"
python3 scripts/run_v6_7_unet_transfer_study.py teacher-check --run-dir "$RUN"
python3 scripts/run_v6_7_unet_transfer_study.py distill --run-dir "$RUN"
python3 scripts/run_v6_7_unet_transfer_study.py benchmark --run-dir "$RUN" --repeats 3 --warmups 1 --rhs-counts 1 4 --regimes cold warm multiple
```

Teacher가 거부되어도 마지막 benchmark는 direct CNN과 고정 classical C를 비교한다.
Teacher가 통과한 경우에는 증류가 완료되어야 benchmark가 가능하다.
`train --stage direct` / `train --stage teacher`로 개별 실행할 수 있다.
중단한 단계에는 `--resume`을 사용한다. 소스·설정·환경·가중치·판정 근거가 달라지면
새 RUN을 만들어야 한다. Teacher gate 이후에는 같은 RUN에서 전문가를 다시 학습하지 않는다.

## 본 연구와 최종 평가

새 RUN에서 위 prepare의 config를 `configs/v6_7_unet_transfer_research.json`으로
바꾼 뒤 같은 train → teacher-check → distill → benchmark 순서로 실행한다.
Benchmark는 `--repeats 5 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple`
을 권장한다. Offline 학습 시간은 따로 기록하며 online 시간에 합산하지 않는다.
Online cold/multiple 시간은 타이머를 시작한 뒤 classical hierarchy를 생성하므로
**EM 계산·CNN 추론·P의 sparse assembly·Galerkin/coarse factorization·실제 solve·
fallback**을 모두 포함한다. 최종 판정의 `wall_seconds`는 이 전체 구간의 실측값이다.
Warm은 이미 생성한 hierarchy를 재사용하는 진단이며 최종 성공 판정에 사용할 수 없다.
Offline teacher/direct/KD 학습 비용은 따로 기록한다.

NN 생성 실패, nonfinite 결과 또는 잔차 증가 시 실패한 cycle 직전의 해로 되돌리고
같은 고정 EM classical hierarchy로 복구한다. 같은 RHS를 푸는 동안에는 classical로
고정된다. 종료 허용치 근처의 classical 마무리와 기존 cycle 예산도 유지한다.
Teacher의 raw 3-cycle 잔차 검사에서는 이 safeguarded solver를 호출하지 않는다.
실패한 teacher 연산자는 실패로 기록하며 classical 잔차로 대신 채우지 않는다.

독립 final/OOD는 모든 설정과 checkpoint를 고정한 다음 한 번만 소비한다.

```bash
python3 scripts/run_v6_7_unet_transfer_study.py freeze --run-dir "$RUN" --repeats 5 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple
python3 scripts/run_v6_7_unet_transfer_study.py final --run-dir "$RUN"
```

Smoke run에서는 freeze/final이 허용되지 않는다. Final을 보고 teacher/gate/모델을
다시 선택할 수 없다. Student 성능 판단은 named workload와 공통 성공 cohort에 한정된다.

## 확인할 결과

- `checkpoints/{direct,teacher,distilled}/training.json`: 실제 loss, feasibility, KD 적용,
  skipped update, 학습 시간. Distilled는 gate 통과 시에만 존재한다.
- `fixed_classical.json`: EM 예산을 포함한 불변 classical 설정.
- `teacher_raw_contractions.json`: fallback 없는 동일 cycle residual 곡선.
- `teacher_validation/raw_results.json`: teacher/direct/고정 classical C의 실제 tolerance 검증.
- `teacher_gate.json`, `distillation_decision.json`: 판정 근거와 checkpoint/evidence hash.
- `benchmarks/validation/comparison.json` 및 `comparison.csv`: 시간, cycle, 실패,
  solution/energy error, 실제 neural 사용, setup/fill 비용.
- `deployment_decisions.json`: cold/multiple 각각 고정 classical C 대비 1.03× 이상,
  CI 하한 >1, 새 실패 없음, 실제 neural 사용, accuracy 진단을 모두 충족하는지.
  Warm-only 개선은 이 판정을 통과시키지 않는다. 증류의 추가 가치는 direct CNN 대비
  속도비도 함께 보고 판단한다. 어떤 경우에도 자동 production 승격이나 수렴 인증은 없다.

Validation의 선택 결과와 independent final/OOD 결과를 구분해야 한다.
실험이 느리거나 teacher gate가 거부되는 결과도 그대로 기록한다.

## 이번 변경의 범위

요청한 baseline·학습 parent·teacher gate·setup 포함 평가·fallback 외에 필요한 변경:

- CLI의 `calibrate`를 `prepare`로 교체하고, 사용하지 않는 portfolio/selector 설정을 삭제했다.
- 데이터/체크포인트/teacher gate/final freeze를 `baseline_digest`로 결합했다.
  이전 run과 설정이 다른 결과를 섞는 resume는 거부한다.
- 결과의 `strong_C`, `*_vs_strong` 이름을 `fixed_C`, `*_vs_fixed`로 변경했다.
- 공유 데이터·학습·평가 모듈은 fixed baseline을 지원하고, legacy selector는 필요한
  이전 경로에서만 import한다. 수치·통계 보고 함수도 selector 의존성에서 분리했다.
- EM 고정 반복 옵션을 runtime/differentiable graph에 동일하게 전달하고 캐시 키에
  포함했다. 기존 다른 실험은 기본값 `false`로 종전 EM 동작을 유지한다.
- raw gate는 `fallback_allowed: false` 근거가 없는 결과도 통과시키지 않는다.
- 모델 구조, attention 없음, 학습 횟수, KD 손실, 5% teacher 기준, 1.03× 시간 기준,
  종료 허용오차와 통계·정확성 조건은 유지했다.
