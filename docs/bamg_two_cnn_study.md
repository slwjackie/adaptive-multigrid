# BAMG teacher와 두 CNN을 사용하는 직접 P 실험

이 경로는 `residual-u-net-teacher`의 기존 실험을 보존하면서 추가한
P-only 실험이다. 수치 알고리즘이 만든 연결 mask와 보간값으로 독립적인
Mask CNN과 Weight CNN을 지도학습한다. 이후 기준 모델과 문제·level별 mask를
고정하고 Weight CNN만 여러 V-cycle의 잔차로 미세조정한다.

Full coarsening, coarse 점 위치, classical `line_alt`, pre/post sweep,
`R=P.T`, 실제 Galerkin `Ac=P.T@A@P`, 최하위 직접 풀이는 유지한다.
학습하는 것은 CNN 파라미터다. `P`의 보간 가중치는 CNN이 만들어 내는
출력이며, smoother 자체를 학습하거나 cycle마다 모델을 갱신하지 않는다.

## Teacher가 만드는 label

Teacher는 별도의 신경망이 아니다. 무작위 초기 오차에 동차 문제의
line smoothing을 적용하여 쉽게 제거되지 않는 오차 표본을 만든다.

\[
z_j=S_{\mathrm{line}}^{\nu_j}\xi_j.
\]

SPD 행렬의 동차 문제 `Az=0`의 정확해는 0이다. 남아 있는 비영 `z_j`는
정확해가 아니라 아직 제거되지 않은 오차다. 기본 smoothing 예산은
`{2,4,8}`의 혼합이고, 정규화한 표본을 coarse 값으로 복원하도록 국소
연결과 signed 가중치를 찾는다.

\[
\min_{P\in\mathcal C}\frac1m\sum_{j=1}^{m}
\|z_j-Pz_{c,j}\|_2^2+\mu\|P\|_F^2.
\]

`mu`는 **teacher P 원소의 제곱합**에 대한 ridge 계수다. Bilinear에
가까워지게 하는 벌점도, CNN 파라미터의 weight decay도 아니다. 표본의
가중합을 1로 정규화하여 표본 수를 바꿔도 ridge의 상대 크기가 의도치 않게
달라지지 않도록 한다. `mu=0`도 비교할 수 있지만, 많은 표본이 서로 비슷하면
국소 최소제곱의 rank가 부족할 수 있다. 표본 수, 자유변수에 대한 design의
rank·조건수, held-out 오차를 함께 확인해야 한다.

예를 들어 fit 표본의 coarse 값이 항상 `(1,1)`이면 `(0.5,0.5)`와 `(3,-2)`가
모두 1을 복원한다. 새로운 값 `(1,1.01)`에서는 각각 1.005와 0.98을 출력한다.
두 가중치의 제곱합은 0.5와 13이다. Ridge는 이런 큰 상쇄 가중치를 억제하지만
전체 multigrid의 수렴을 보장하지는 않는다.

연결 탐색은 제한된 국소 후보에서의 greedy 탐색이다. 전역 최적 P를
구했다고 해석하지 않는다. Label은 해당 coarse column에 대응하는
**이진 mask와 P 값**이며, scalar reward를 P의 정답으로 사용하지 않는다.

초기 LS 구성 뒤 계층을 만들고 오차 표본을 개선하여 다시 구성하는 것이
여기서의 bootstrap이다. `setup_passes=2`는 초기 구성 1회와 개선 구성 1회를
뜻한다. Smoothing sweep 수나 학습 V-cycle 수와는 별개다. 다중 level의
낮은 모드는 다음 generalized eigenproblem을 사용한다.

\[
H_0=I,\quad H_{l+1}=P_l^TH_lP_l,\quad
A_{l+1}=P_l^TA_lP_l,\quad A_Lv=\lambda H_Lv.
\]

모드를 위로 보간하고 eigen-residual을 개선한다. 별도로 현재 MG가 남기는
hard error를 추가할 수 있다. Fit, teacher 후보 선택, 독립 audit의 표본을
구분한다. Bootstrap 횟수가 늘었다는 이유만으로 개선을 주장하지 않고
선택용 TRAIN probe에서 가장 좋은 유효 후보를 보존한다. Eigen enrichment를
끄면 adaptive LS 변형으로 표기한다. 이는 fixed-coarsening/P-only 실험이며
BAMG 논문 전체의 재현이나 최단시간 보장은 아니다.

## 두 CNN과 직접 출력

두 CNN은 파라미터와 가변 상태를 공유하지 않으며 각각 여러 level에서
재사용한다. Mask CNN은 행렬 특징에서 후보별 logits를 출력한다. 유효한
non-injection 후보의 teacher mask에 분류 손실을 주므로 hard top-k를 통과하는
미분은 필요 없다. Weight CNN은 행렬 특징과 결정된 mask를 입력받아 직접
signed 보간값을 생성한다.

기본 구조는 각 모델에 hidden width 8의 3×3 convolution 5개와 별도의
1×1 출력 head 1개다. 따라서 설정의 `depth=5`는 총 convolution 6개를
뜻한다. 행렬 특징 10개에 x/y coarse phase와 경계 행합 3개를 추가하여
Mask CNN 입력은 13채널이다. 기본 `mask_mode=early`의 Weight CNN은
16개 mask 채널을 더해 29채널을 받는다. 출력은 각각 후보 16슬롯이다.
Mask CNN은 3,424개, Weight CNN은 4,576개 파라미터다. 1단계에서는
합계 8,000개를, 2단계에서는 4,576개만 학습한다. 고정 reference Weight CNN을
포함한 배포 파라미터 복사본은 합계 12,576개다.

\[
P_l=\operatorname{Project}_{\mathcal C(S_l)}
\bigl(S_l\odot g_\theta(X(A_l),S_l)\bigr).
\]

국소 16개 후보 중 bilinear 값이 0이던 유효 연결도 선택할 수 있다.
기본 연결 예산은 `k=4`, 비교 옵션은 `k=8`이다. 경계에서는 유효 후보 수로
제한하고 coarse 대표점은 정확한 identity injection으로 고정한다.
Bilinear에 delta를 더하지 않는다. 경계 행합은 geometry 제약이며,
eliminated Dirichlet 경계의 모든 행에 합 1을 강제하지 않는다. 최종 가중치는
row-L1 cap을 만족시킨다. 선택된 값이 0이 되어도 연결을 삭제하지 않는다.
`Project`는 행합을 보존하면서 균등한 feasible 행을 중심으로 zero-sum
성분을 줄이는 radial feasibility retraction이다. Euclidean 최근접 투영이라고
주장하지 않는다. Teacher의 제약 LS와 CNN의 이 decoder는 같은 feasible
제약을 사용하지만 최적화 알고리즘 자체는 다르다.

기존 행렬 특징은 local stencil의 요약이다. 실제 coarse 행렬의 모든 먼
결합을 CNN 입력으로 보존한다는 의미는 아니다. Solver의 residual,
Galerkin 및 classical smoothing에는 실제 전체 sparse 행렬을 사용한다.
`hierarchy.feature_coverage`에는 level별 9점 stencil 밖의 저장/실제 비영
연결 수와 그 결합의 상대 Frobenius 크기를 기록한다.

## 기준 계층과 실제 계층

1단계가 끝나면 Mask CNN `phi_ref`와 Weight CNN `theta_ref`의 완전한
고정 복사본을 만든다. 같은 root A에 대해 다음 기준 계층에서 mask를
한 번 결정한다.

\[
S_l=\operatorname{topk}(f_{\phi_{ref}}(X(A_{ref,l}))),\quad
P_{ref,l}=\operatorname{Project}(S_l\odot
g_{\theta_{ref}}(X(A_{ref,l}),S_l)),\quad
A_{ref,l+1}=P_{ref,l}^TA_{ref,l}P_{ref,l}.
\]

2단계의 실제 계층은 저장한 `S_l`과 현재 학습용 `theta`로 새로 만든다.

\[
A_0=A,\qquad
P_l=\operatorname{Project}(S_l\odot g_\theta(X(A_l),S_l)),\qquad
A_{l+1}=P_l^TA_lP_l.
\]

따라서 `theta`가 변하면 실제 coarse A와 line/coarse 분해도 바뀐다.
하지만 같은 root A·같은 level의 mask는 변하지 않는다. 서로 다른 문제나
서로 다른 level의 mask가 다른 것은 정상이다. 한 K-cycle rollout과 실제
풀이의 여러 cycle/RHS에서는 만들어 놓은 계층을 재사용한다.

새로운 A에서도 동일한 고정 reference pass로 mask를 만들고 최종 Weight CNN으로
실제 계층을 만든다. 새 A의 teacher label은 필요 없다. 배포에는
`phi_ref`, `theta_ref`, 최종 `theta`와 생성 규칙이 모두 필요하다.
`theta_ref`는 재현용 고정 복사본이며 세 번째 학습 모델이 아니다.

## 잔차 학습의 의미

2단계는 정답 P 없이 실제 solver를 미분한다. P, 실제 Galerkin,
하위 level Weight CNN, classical line 연산, 최하위 solve를 통한 gradient를
유지한다. Mask 생성 경로만 고정한다. `b=0, x0=e`를 사용하는 probe의
초기 잔차는 `r0=-Ae`이며 오차 norm으로 대신 정규화하지 않는다.

\[
q_k=\frac{\|r_k\|_2}{\|r_0\|_2},\qquad
\mathcal L_{res}=\mathbb E\left[\sum_{k=1}^K
\omega_k\log(\max(q_k,\epsilon))\right].
\]

기본값 `uniform`은 `omega_k=1/K`다. `K=3`이면 각각 1/3,
`K=6`이면 각각 1/6이다. 필수 비교 `terminal`은 마지막 항만 1이며,
마지막 잔차까지 수행한 모든 cycle을 통한 역전파는 유지한다. `omega`는
사람이 정한 cycle 손실 설정으로, teacher 표본 가중치나 P 원소와 다르다.
두 방식은 동일한 초기 모델·reference mask·데이터·학습 예산에서 비교하고
손실 숫자의 크기가 아니라 실측 validation warm 수렴시간으로 선택한다.

학습의 `K=3→6`은 시작 예산이다. 더 긴 rollout과 tolerance solve에서
정체·발산을 확인해야 한다. 처음부터 수렴한 probe는 나누기 전에 분기하고,
NaN·발산·실패를 결과에서 제외하지 않는다. 고정 mask의 연결 수는 `theta`와
무관하므로 연결 수 비용항은 기본 손실에서 사용하지 않는다.

## 시간 측정과 해석

주 선택 목표는 준비된 계층에서
`||r|| <= max(1e-12, 1e-8*||r0||)`에 도달하는 **warm RHS=1 시간**이다.
Warm에는 실제 cycle, RHS-dependent 작업, residual과 종료 검사가 들어간다.
Reference pass, CNN, P/Ac assembly, line/coarse 분해는 setup에 들어간다.
Cold는 A의 mask·계층 cache가 없는 상태에서 setup과 solve를 합한다.
모델 로드와 프로세스 최초 실행 비용의 포함 여부는 실행 보고서에 명시한다.

같은 A의 순차 RHS 재사용은 `setup + sum(solve_j)`다. 동시에 푸는
batched RHS=4/16과 구분한다. Numerical teacher를 solver로 비교할 때는
teacher 구축 비용도 해당 방법의 cold/setup에 포함한다. Offline 학습과
label 생성 비용을 learned online setup과 섞지 않는다.

\[
\rho_k=\frac{\|r_{k+1}\|}{\|r_k\|},\qquad
v_k=\frac{1-\rho_k}{\Delta t_k},\qquad
G_k=\frac{-\log\rho_k}{\Delta t_k}.
\]

`rho`는 인접 cycle의 잔차 배율이고 `q`는 초기 잔차 대비 값이다.
`rho>1`이면 progress rate는 음수다. Trajectory log rate는
`log(||r0||/||rK||)/sum(dt_k)`다. Rate를 기록했다는 이유만으로 해당 rate로
학습했다고 표시하지 않는다. 측정 시간을 상수로 두는 시간 가중 손실도
wall clock 자체를 미분하는 것은 아니다.

예를 들어 매 cycle `rho=0.8`, 1 ms이면 상대 감소율은 0.2/ms지만
상대 tolerance 1e-8에 83 cycle, 83 ms가 필요하다. `rho=0.25`, 4 ms이면
0.1875/ms여도 14 cycle, 56 ms다. Cycle별 rate 순위는 실제 수렴시간
순위를 보장하지 않는다. 실패를 포함한 전체 cohort와 성공률을 함께 보고하며,
성공 case만 모아 속도 우위를 주장하지 않는다.

## 검증 범위와 추가 실험

비교군은 bilinear, EM5, one-pass LS, bootstrap teacher, 두 CNN 지도학습,
동일 mask에서의 uniform 미세조정과 terminal 미세조정이다. Final/OOD는
모델 선택에 사용하지 않는다. Raw 수렴과 fallback을 쓴 배포 수렴을
구분하여 fallback의 성공을 learned P 자체의 성공으로 세지 않는다.

CNN 크기는 Mask와 Weight 각각 독립 설정한다. Mask만 축소, Weight만 축소,
둘 다 축소를 별도 실험으로 비교한다. 기본 Weight CNN은 mask를 입력받으므로
같은 level의 Mask→Weight 실행에는 순서 의존성이 있다. 병렬 추론은
architecture와 backend를 바꾸는 별도 실험이며, 두 CNN의 시간이 하나와
같다고 가정하지 않는다. 이 변경은 주로 setup에 영향을 준다.

작은 수치 테스트와 smoke는 수학적 계약과 실행 경로를 확인한다.
모든 A에서 수렴하거나 EM5/bilinear보다 빠르다는 증명이 아니다.
대규모 teacher 생성·장시간 학습, final/OOD, architecture sweep,
병렬 추론의 성능 결과는 실제로 실행한 경우에만 보고한다.

## 실행

명령은 저장소 루트에서 실행한다. 기본 CLI는 CPU CSR/FP64 경로이며
`coarsest_n=3`을 사용한다. GPU 및 다른 terminal 크기의 성능을 검증한
경로로 해석하지 않는다. 아래 thread 설정은 BLAS와 Torch의 CPU 경쟁을
제한한다. Torch thread 수는 JSON에도 명시한다.

```bash
python -m pip install -e '.[dev]'
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python scripts/build_native_stencil.py --no-openmp

python scripts/run_v6_7_bamg_study.py run \
  --config configs/v6_7_bamg_smoke.json --output artifacts/bamg_smoke
```

`run`은 prepare→teacher→pretrain→freeze-masks→uniform/terminal
finetune→diagnose→validation benchmark를 실행한다. Final/OOD를 자동으로
열지 않는다. 단계별 실행은 다음과 같다.

```bash
python scripts/run_v6_7_bamg_study.py prepare \
  --config configs/v6_7_bamg_smoke.json --output artifacts/bamg_smoke
python scripts/run_v6_7_bamg_study.py teacher --output artifacts/bamg_smoke
python scripts/run_v6_7_bamg_study.py pretrain --output artifacts/bamg_smoke
python scripts/run_v6_7_bamg_study.py freeze-masks --output artifacts/bamg_smoke
python scripts/run_v6_7_bamg_study.py finetune --output artifacts/bamg_smoke --weighting both
python scripts/run_v6_7_bamg_study.py diagnose --output artifacts/bamg_smoke
python scripts/run_v6_7_bamg_study.py benchmark --output artifacts/bamg_smoke
```

한 run에 전체 실행과 단계별 실행을 중복 적용하지 않는다. 중단한 단계에는
`--resume`을 붙인다. `diagnose`와 최종 선택을 잠그는 `freeze`에는 이 옵션이
없다. 소스·환경·설정·label·reference가 달라졌으면 같은 run을 재개하지 않고
새 output 디렉터리를 사용한다. 실패 checkpoint를 성공 run처럼 자동 재개하지
않는다. 학습 checkpoint에는 모델, optimizer, 진행 횟수, RNG와 provenance가
들어가며 fine-tune checkpoint에는 기준 모델과 고정 mask도 포함된다.

| 설정 | Smoke | Research P1 / Q1 |
| --- | --- | --- |
| TRAIN | n=7, 7 families × 1 | n=15/31/63/127, 7 families × 10 = 280 A |
| Validation | n=7, 7 families × 1 | n=31/63, 7 families × 4 = 56 A |
| Fit / selection / audit 벡터 | 각각 16 | 각각 32 |
| Teacher setup pass / μ | 2 / 1e-6 | 2 / 1e-6 |
| 지도학습 updates | 2 | 2,000 |
| 각 미세조정 updates | 2 | 1,000 |
| K 전환 | 첫 update K=3, 이후 K=6 | 첫 500 update K=3, 이후 K=6 |
| 지도 손실 α_mask / α_weight / α_action | 1 / 1 / 0.1 | 1 / 1 / 0.1 |
| Benchmark 반복 | 1 | 3 |
| solve timeout / 실패 시간 벌점 | 30 s / 30 s | 120 s / 120 s |
| 모델 선택 허용 실패율 | 0 | 0 |

지도 손실은 level별 동일 비중, 각 level의 learnable row별 동일 비중이다.
BCE는 행의 valid 후보를 평균하고, P MSE는 선택된 후보를 평균하며
teacher 행 L1 norm과 1 중 큰 값으로 오차 크기를 정규화한다.
Action 항은 teacher test vector에 대한 보간 차이를 평가한다.
미세조정은 random/smoothed/detached hard error probe를 각각 1개씩 사용하며
Adam learning rate는 1e-4다. 지도학습 Adam learning rate는 1e-3이다.

```bash
python scripts/run_v6_7_bamg_study.py run \
  --config configs/v6_7_bamg_research_p1.json --output artifacts/bamg_research_p1
python scripts/run_v6_7_bamg_study.py run \
  --config configs/v6_7_bamg_research_q1.json --output artifacts/bamg_research_q1
```

P1과 Q1은 별도 run이며 기존 7개 family를 유지한다. Research를 실행하기
전에 작은 격자에서 teacher의 case당 시간·sparse 저장량, 각 K의 update당
시간을 측정한다. 예상 offline 시간은 크기별 case 수×teacher 평균 시간과
2,000개의 지도 update 및 각 1,000개의 미세조정 update 시간을 합해 산정한다.
Validation도 56 A×7방법×3반복이며 numerical teacher를 다시 구성하므로
그 비용을 포함해야 한다. 작은 smoke 시간만으로 n=127/255의 비용을
선형 외삽하지 않는다. Sparse nnz·operator complexity·저장 바이트 예산은
명시적으로 검사한다. 저장 바이트는 전체 프로세스 peak memory와 다르며,
process memory 관측치는 별도로 읽어야 한다.

Validation에서 선택을 완료한 뒤 final/OOD가 필요할 때만 실행한다.

```bash
python scripts/run_v6_7_bamg_study.py freeze --output artifacts/bamg_research_p1
python scripts/run_v6_7_bamg_study.py evaluate-final --output artifacts/bamg_research_p1
```

기존 연구의 holdout claim/freeze 규칙을 그대로 사용한다. Final/OOD 결과로
다시 학습·선택하지 않는다. 허용 실패율을 만족한 learned 후보가 없으면
freeze를 성공시킨 것처럼 처리하지 않는다.

## 결과 파일과 구현 범위

| 파일 | 용도 |
| --- | --- |
| `configuration.json`, `study_manifest.json`, `fixed_classical.json` | 설정·소스·환경·고정 비교 기준 |
| `data/development_manifest.json` | operator 기준 TRAIN/validation 분리 |
| `teacher/train/*.pt`, `teacher/index.json` | 수치 teacher 계층·label·진단 및 digest |
| `pretrain.pt`, `supervised_models.pt` | 재개용 지도학습 상태와 두 CNN |
| `reference.pt`, `frozen_masks.json` | 고정 기준 모델·문제별 mask 및 digest |
| `finetune_uniform.pt`, `finetune_terminal.pt` | 같은 mask에서의 재개용 미세조정 상태 |
| `deployment_uniform.pt`, `deployment_terminal.pt` | teacher 없이 새로운 A를 처리하는 배포 패키지 |
| `benchmark.json` | raw/deployed, setup/warm/cold, 순차 RHS 및 validation 선택 |
| `diagnostics.json` | 모델 크기, μ/ω, synthetic 손실/rate 비교 |

Benchmark는 모델이 메모리에 로드된 상태를 가정한다. Cold setup에는
새 reference pass를 포함하지만 모델 파일 로드는 포함하지 않는다.
RHS=1/4/16 결과는 **순차 재사용**이고 simultaneous batched 측정은 아니다.
실패에는 사전 설정한 시간 벌점을 적용하며 case별 반복 median 뒤 전체
cohort 평균을 계산한다. Raw 성공률을 먼저 충족한 후보 사이에서 warm RHS=1
시간으로 선택한다. Deployed 경로는 별도 bilinear fallback 준비 비용과
실제 growth rejection/stagnation tail 비용을 기록한다.

`mask_mode=late|none`, 각 CNN의 depth/hidden 변경, k=8, μ=0,
벡터 예산 변경은 명시적인 새 config/run으로 비교할 수 있다. Low-level API는
고정 mask LS label 재적합과 Weight CNN warmup, custom cycle 가중,
detached deployment 시간으로 나눈 상대/log progress 손실을 지원한다.
주 CLI 실험은 uniform/terminal 비교다. 시간 가중 옵션은 `time_profile`에
snapshot ID, `measurement=deployment_cycle`, `backend=scipy_fp64_cpu`,
`dtype=torch.float64`, k, 두 모델 spec, 측정 hardware, cycle seconds와
유효 update 예산을 요구하며 불일치를 거부한다. 본 기본 run은 rate 학습을
수행하지 않는다.

`support_mode=fixed_bilinear`는 기하학적 bilinear 연결만 유지하면서 Weight
CNN의 직접 출력값을 학습하는 별도 대조 실험이다. Teacher 지도학습은 동일하게
수행하고 reference 생성에서 Mask CNN의 선택을 사용하지 않는다. 기본값은
`learned`이며 두 조건은 각각 새 run으로 비교한다. Fixed-bilinear 모드도
bilinear 값에 보정량을 더하지 않는다.

Stage1 student 분포 teacher refresh, GPU 병렬 실행 및 architecture 자동 sweep은 기본 CLI가 실행한 비교군으로
표시하지 않는다. `late|none` 모델이 존재한다는 것이 병렬 성능 검증을
완료했다는 뜻은 아니다. 이 문서의 research 명령은 재현 예산이며
장시간 학습 또는 final/OOD의 성능 결과를 대신하지 않는다.

## 회귀 검증

```bash
python -m pytest -ra
sha256sum -c THREE_PILLARS_SOURCE.sha256
sha256sum -c P_HEADROOM_SOURCE.sha256
sha256sum -c WORLD_MODEL_SOURCE.sha256
sha256sum -c HS_WORLD_SOURCE.sha256
```

Source provenance를 검사하는 기존 테스트는 새 Python 파일의 추가도
변경으로 감지한다. 전체 회귀 및 smoke 실행 중에는 소스를 수정하지 않는다.
검증에서 관측한 실제 결과는 별도 실행 보고서에 기록한다.

## 관련 1차 자료

- Brandt, Brannick, Kahl, Livshits, *Bootstrap AMG*: https://doi.org/10.1137/090752973
- *Bootstrap Algebraic Multigrid: Status Report, Open Problems, and Outlook*: https://arxiv.org/abs/1406.1819
- Kahl & Rottmann, *Least Angle Regression Coarsening in Bootstrap Algebraic Multigrid*: https://arxiv.org/abs/1802.00595
- Greenfeld et al., *Learning to Optimize Multigrid PDE Solvers*: https://proceedings.mlr.press/v97/greenfeld19a.html

이 구현은 위 접근들의 일부 구성 원리를 사용한다. Coarsening 변경이나
논문의 전체 실험·성능을 재현한 것으로 표시하지 않는다.
