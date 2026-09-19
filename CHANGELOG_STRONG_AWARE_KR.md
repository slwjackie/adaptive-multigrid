# Strong-aware research 변경 기록

이 변경은 기존 Adaptive Neural Multigrid v6.7 위에 독립 연구 경로를 추가합니다.
기존 fixed/strong baseline 결과나 과거 certificate를 새로운 최종 성능 주장으로 바꾸지 않습니다.

## 보존한 기반

- 기존 `strong.py`의 deterministic A-only selector와 classical strategy bank.
- C/H_S/H_P/H_SP의 replacement 의미, tolerance/budget, rollback과 CLASSICAL LOCK.
- CPU FP64 numerical solve, CSR/native C++·OpenMP kernel, eager classical/lazy neural 구분.
- 기존 checkpoint와 historical artifacts. 원래 format 13 모델의 load와 signature 계약.

## 추가한 연구 구성

- `research_data.py`: 새 operator-disjoint train/validation, 기록된 C*(A), 과거 operator
  제외, 사전 고정한 final/OOD specification 및 freeze 뒤 single-use final gate.
- `research_smoothers.py`: 공통 bounded gain/single 9-point basis를 쓰는 ordinary control,
  compact/residual student, 정확한 D4 평균, scalar/order1/order2 O2 candidate,
  training-only CNN/O2 teacher. D4 8-view 비용과 O2 fixed-grid 오차를 명시.
- `research_transfer.py`: sparse A graph의 작은 GNN/edge-MLP와 큰 teacher,
  제한된 support expansion, injection/row sum projection, actual fill cap와 soft work proxy.
- `research_controls.py`: 기존 CNN-P에도 동일 support와 complexity budget을 적용하는 control.
- `research_training.py`: C*(A) 기준 direct full-V-cycle 학습, 별도 teacher 및 조건부 KD,
  KD 후 task-only fine-tuning, joint candidate와 독립 checkpoint/resume 기록.
- `research_policy.py`: 바뀐 expert에 맞춘 spatial/temporal 자료와 작은 control을 재생성하고,
  A-only branch utility·classical abstention을 사용하는 연구용 deployment adapter.
- `research_evaluation.py`: strong C 및 branch 비교, raw JSON/CSV, cold/warm/multiple RHS,
  실제 성공·오차·시간·neural use·fallback·complexity 기록.
- `research_generation.py`: 실제 sparse bank 준비를 포함한 CPU/MPS/CUDA FP32 측정과
  고정 격자 arbitrary-angle resampling 오차 diagnostic. 미측정 hardware를 구분.

## 기존 실행 경로의 제한된 연결 변경

- `models.py`와 `research_runtime.py`: 새 expert의 format 14 architecture spec 복원과
  cache signature. Format 13은 기존 constructor를 그대로 사용.
- `banks.py`: graph-aware P generation, 제한된 candidate support와 FP64 projection,
  complexity 검사, training-only teacher generation guard.
- `unroll.py`: 실제 A/P sparse graph와 같은 numerical projection을 사용하며,
  top-k forward와 straight-through gradient를 구분. Zero/pruned candidate의 학습
  gradient가 FP64 support repair 때문에 사라지는 문제를 방지.
- `solver.py`: model/config에 맞춘 generation cache와 RHS당 teacher guard. Offline에서
  생성한 teacher bank가 context 밖에서 cache reuse로 실행되지 않게 검사.
- H_S/H_SP bank 사이에도 actual level A digest와 smoother/config가 같으면 stencil을
  공유하며, learned P 때문에 실제 coarse A가 변경된 경우만 해당 level을 새로 생성.

## 검증 및 결과 해석

기존 regression에 shape/gain/gradient, D4/continuous tensor identity, teacher 경계,
sparse P 제약·Galerkin·complexity, C*(A) 불변성, 데이터 digest 분리, cache reuse,
checkpoint round-trip 및 비용 집계 검사를 추가했습니다. 테스트 수와 실제 smoke 결과는
run의 verification/result 보고서를 기준으로 확인하세요. 이 문서는 시험 실행 수나 성능
향상을 추정하여 기록하지 않습니다.

연구 CLI는 `scripts/run_v6_7_strong_aware.py`입니다. `develop`은 개발자료만 사용하며,
`freeze`와 `final`은 별도 단계입니다. 긴 전체 학습은 명시적인 full research 명령으로
실행합니다. Detailed usage는 [STRONG_AWARE_README_KR.md](STRONG_AWARE_README_KR.md),
설계와 수학적 한계는 [아키텍처 문서](docs/STRONG_AWARE_ARCHITECTURE_KR.md)를 참고하세요.

## 통합 검토 후 보완

- 실제 P/Ac/aggregate cap을 학습 proposal에도 검사. 위반 시 Classical task와 별도 differentiable repair loss를 기록.
- 실패한 P/S setup도 model/config/actual-A key로 cache하여 같은 RHS/branch에서 반복 heavy inference 방지.
- Final/OOD의 frozen ordinary/D4/O2/GNN 대조군, cold classical-bank 사후 reference 및 expected-RHS blind ablation 추가.
- 기존 smoke에서 이어 하는 검증 CLI와 역할별 CPU/MPS generation CLI 추가.
- 설치 위치 전체 494 tests 통과. 기존 checkpoint로 14-case 비교, 실제 1/4/16/64 RHS, CPU/MPS generation 실측.
- 속도 우위 미확인으로 KD/joint/production 승격 없이 부정적 결과를 보존.
