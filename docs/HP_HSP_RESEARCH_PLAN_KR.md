# H_P / H_SP 후속 연구: 근거와 신규 제안의 구분

이 문서는 warm-study 구현 이후의 **제안**이다. 아래 P 재설계나 새 joint 학습법을
이번 main 변경에서 이미 구현/학습했다고 해석하지 않는다. 기존 support-preserving
H_P/H_SP 경로는 유지했다. 목표는 standalone FP64 MG이고, 선행연구의 Krylov나
non-Galerkin 결과와 구분한다.

## 확인한 primary sources

1. Greenfeld et al., ICML 2019, *Learning to Optimize Multigrid PDE Solvers*.
   https://proceedings.mlr.press/v97/greenfeld19a.html
   P를 PDE 계수에서 생성하고 two-grid error-propagation loss로 학습한다. Black-Box MG
   대비 수렴 개선. 학습된 P로 만든 실제 coarse operators를 추가 학습에 포함한다.
   이 결과를 현재 strong portfolio 대비 wall-clock 배수로 읽으면 안 된다.
2. Luz et al., ICML 2020, *Learning Algebraic Multigrid Using Graph Neural Networks*.
   https://proceedings.mlr.press/v119/luz20a.html
   GNN으로 classical algebraic P의 sparse values를 학습한다. Coarse nodes/sparsity/row
   sum을 classical 기준과 맞추며, GS smoothing과 error-propagation loss를 쓴다. 논문은
   NN setup이 더 비싸다는 점을 명시한다. Fixed support 자체가 학습 실패를 의미하지 않는다.
3. Wang, Gu, Sun, Xu, JCP 493 (2023), 112437,
   *Learning-based local weighted least squares for algebraic multigrid method*.
   https://doi.org/10.1016/j.jcp.2023.112437
   LWLS의 spatial weights, initialization, correction을 학습하여 P를 구성한다.
   Graph Laplacian/diffusion/Helmholtz에서 수렴과 크기·계수 분포 일반화를 보고한다.
   확인한 publisher abstract/section summaries는 현재 코드 대비 wall-clock 배수를 주지 않는다.
4. Liu et al., 2024 preprint, *Learning a generalized multiscale prolongation operator*.
   https://arxiv.org/abs/2410.06832
   U-Net으로 local spectral coarse subspace를 근사하고 subspace-distance loss 사용.
   Local spectral problem 대비 P 생성시간 약20배 절감 사례를 보고하나, PCG+two-grid의
   preconditioner 효율 유지가 목적이다. Bilinear P나 standalone strong_C보다 20배 빠르다는 뜻이 아니다.
5. Fink et al., 2026 preprint, *RAPNet: Accelerating Algebraic Multigrid with Learned Sparse Corrections*.
   https://arxiv.org/abs/2605.26854
   Fine/coarse composite graph에서 P,R,Ac sparse correction을 함께 학습, level-wise shared
   weights와 algebraically smooth training vectors를 활용한다. Setup-only inference 후 sparse
   cycles를 수행한다. Standalone/GMRES, single precision, residual1e-6; 모든 dataset에서
   항상 우월한 것은 아니며 anisotropic diffusion standalone에서는 SpSA가 더 적은 iterations.
   Smoother는 고정 Jacobi다. 이 논문은 H_SP(S+P) 공동학습의 직접 증거가 아니다.
6. Wang & Luo, 2026 preprint / WCCM-ECCOMAS 2026 abstract,
   *A Learnable Multigrid Framework via Graph Convolutions*.
   https://arxiv.org/abs/2608.29082
   https://wccm-eccomas2026.org/event/contribution/d297d1e0-ed22-11f0-b205-000c29ddfc0c
   공개 초록에서 smoothing+inter-grid transfer를 learnable graph convolution으로 구성하고
   수렴 개선·mesh generalization을 보고한다. 이번 조사에서는 원문 PDF를 확보하지 못했으므로
   layer별 상세구성·wall-clock 배율·H_SP 상승폭은 확인되지 않았다. 참고 방향이지 이식 근거 확정 아님.

## 1순위: 네트워크 교체보다 표현가능성과 gradient 진단

TRAIN operator만 골라 현재 P head를 고정하고 P의 자유 weight를 직접 최적화한다.
현재 bounded delta(.35), 넓은 delta(1,2), same-support zero-sum additive parameterization,
제한된 algebraic-support 확장을 순차 비교한다. 여러 초기값을 사용한다.
최적화가 실패했다고 좋은 P의 부재를 증명한 것은 아니다.

측정: ||P-P_C||/||P_C||, 실제 correction 차이, 각 row의 support 수/자유도,
masked/zero gradient 비율, logit 범위, cyclic contraction, operator complexity.
Softmax 전체 logit의 공통 shift는 무효이므로 raw output 크기만 보고 판단하지 않는다.
Current support/injection/SPD Galerkin 보장은 안정한 구조를 주지만 빠른 수렴을 보장하지 않는다.

## 2순위: learned LWLS와 slow-error target

현재 H_S(또는 classical S)를 몇 번 적용해 남은 TRAIN error vectors v를 수집한다.
국소 interpolation은 min_p sum_q omega_iq (v_iq - sum_j p_ij v_jq)^2 + regularizer 형태로
구성하고, NN은 weights/initialization/small correction을 학습한다. 이 식은 이 저장소를
위한 제안이며 Wang et al.의 전체 알고리즘 복제는 아니다.

Output P를 current row-sum/injection/support 제약에 맞춰 사용하고 full multilevel objective로
검증한다. Teacher labels를 만들기 위한 직접 최적화/LWLS는 offline TRAIN 비용이다.
새 A에서도 그 계산을 수행한다면 cold setup에 포함해야 한다.

## 3순위: H_SP는 S와 P의 오차 역할을 맞춘 교대 학습

독립 H_S와 H_P를 더한다고 상승폭이 곱해지는 것은 아니다. E = S_post (I-P(P^TAP)^(-1)P^TA) S_pre
의 전체 역할을 본다. 실제 multilevel에서는 inverse가 recursive coarse solve로 대체된다.

제안 순서:
1) 개선된 S 고정 -> 그 S가 남기는 slow errors로 P 학습.
2) P 고정 -> 새로운 coarse space가 처리하지 못하는 error로 S 미세조정.
3) 작은 learning rate로 짧은 joint fine-tuning.
4) H_S-only보다 더 좋은 warm time/성공률을 별도 validation에서 요구.

S/P 각각의 상대노해 penalty는 C*(A)뿐 아니라 현재 best H_S에 대한 비교를 별도로 보고한다.
Sparse fill, rank, actual learned hierarchy factorization 비용을 숨기지 않는다.
S에 쓰는 coarse-complement target과 P에 쓰는 smoother-slow target은 서로 다르다.

## 4순위: 실제 coarse A 분포 학습과 적용 level 분리

원 fine A만으로 학습하지 않고 생성한 P^TAP도 TRAIN sample로 사용한다(Greenfeld/Luz 방향).
Fine-only와 coarse-only P를 비교해 generation cost와 coarse bottleneck을 분리한다.
Shared-weight level encoder는 새 hierarchy depth에 적용 가능하지만 성능 일반화는 독립 평가 필요.
현재 new config의 smoother_levels/transfer_levels 분리를 후속 실험에서 활용할 수 있다.

## 장기 방향: multiscale basis / non-Galerkin corrections

Subspace-distance teacher와 patch basis는 high-contrast slow modes 표현에 유망하지만 coarse
unknowns가 늘 수 있다. U-Net setup speed를 일반 MG solve speed와 혼동하지 않는다.
RAPNet식 P/R/Ac 동시 correction은 fill 감소에 유망하지만 Ac=P^TAP 계약을 바꿀 수 있다.
따라서 별도 experimental branch, SPD/symmetry/nullspace/true-residual checks, classical
non-Galerkin comparator가 필요하다. 현 strong_C 비교에 조용히 섞지 않는다.

## 권고

H_P의 첫 실험은 큰 GNN/Transformer가 아니라 (a) P 직접최적화 ceiling 진단,
(b) 넓은 범위의 안정한 head, (c) slow-error/LWLS 학습이다.
H_SP는 개선된 S가 남긴 error에 P를 맞춘 뒤 교대 학습하는 것이 1순위다.
이는 선행연구를 바탕으로 한 검증할 가설이며, 아직 성능 향상이나 배율을 측정하지 않았다.
