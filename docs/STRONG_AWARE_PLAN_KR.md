# Strong-aware Neural MG 변경 계획 — 2026-09-16

기존 `strong.py`와 배포 시 사용한 `artifacts/my_v6_7_strong_baseline/calibration/selector_rules.json`을 고정한다. 이 규칙은 앞선 작은 tune에서 한 leaf만 보정된 후보이며 완전한 classical optimum으로 주장하지 않는다. 이번에는 규칙·threshold·strategy bank를 변경하지 않는다.

현재 소스는 Git checkout이 아닌 로컬 프로젝트다. 변경 전 소스·문서 102개의 SHA256을 보존했고, 수정 전 실제 프로젝트의 `.venv/bin/python -m pytest -p no:cacheprovider`는 317개 통과했다. 원본 research, limited-fix, strong-baseline 결과/checkpoint는 읽기 전용이다. 새 결과는 `artifacts/my_v6_7_strong_aware/` 아래에만 만든다.

이미 관찰한 35-case audit의 H_S 약 1.25×는 fixed C 대비다. 별도의 작은 14-case 실험에서는 strong 기반 H_S가 strong C 대비 0.693×였다. 이 수치들을 독립 final 성능으로 사용하지 않는다.

구현 순서:

1. 현재 selector 및 numerical contract 동결, 기존 checkpoint control 재현.
2. 새 operator-disjoint train/validation 데이터와 C*(A) 기록. final/OOD는 학습 명령에서 생성/평가하지 않고 별도 고정 절차로만 공개.
3. 같은 full-V-cycle objective로 direct smoother 및 graph-P 학생을 학습.
4. ordinary/D4/O2 smoother, 기존 P/small GNN/edge-MLP의 독립 ablation.
5. training-only 큰 Teacher를 평가하고, direct 학생보다 실질적 task 개선이 있을 때만 KD 실행. same-size direct와 optimizer update 수 비교.
6. S/P 학생 joint task fine-tuning과 독립 H_S/H_P/H_SP 유지; validation으로 퇴행을 거부.
7. 변경된 expert와 C*(A)로 spatial/temporal label을 새로 만들고 정책을 refit. 기존 정책의 검증을 상속하지 않음.
8. 모든 선택이 고정된 후 별도 명령으로 untouched final/OOD를 1회 평가.

유지하는 수치 실행: FP32 generation → FP64 cache, sparse P 하나, R=Pᵀ, Galerkin Ac, coarse injection/row sums, CLASSICAL LOCK, lazy bank 및 same-A multiple RHS reuse, 기존 C++/OpenMP kernel. Heavy generation은 각 distinct level operator마다 최초 bank 준비 시만 허용한다. A0와 learned coarse A_l은 서로 다른 operator다. 같은 bank를 사용하는 반복 V-cycle/RHS에서는 heavy generation을 다시 호출하지 않는다.

O2의 연속 representation/kernel 제약과 square lattice의 arbitrary-angle resampling 오차를 구분한다. 정사각 격자의 9-point stencil이 임의 회전에 닫혀 있다고 주장하지 않는다. D4의 group averaging 및 O2 생성 비용은 실측하고 숨기지 않는다. sparse graph는 기존 PyTorch 연산을 사용하며 새 package 설치/upgrade는 하지 않는다.

자동 실행 범위는 회귀 테스트, 작은 smoke 학습, n=15/31 짧은 비교다. 큰 train/validation/final/OOD 실행은 config와 명령을 제공하며 자동 실행하지 않는다. Teacher/KD/새 P/O2가 실제 이기지 못하면 채택하지 않고 그 결과를 그대로 기록한다.
