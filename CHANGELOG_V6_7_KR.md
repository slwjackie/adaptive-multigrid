# v6.6 → v6.7

## 새 production

* `adaptive_mg.v67`을 추가. 기존 exact-K one-way API와 별도의 명확한 새 API.
* Operator-generation smoother/P를 lazy bank로 생성하고 순수 numerical application에서 재사용.
* Classical sweep 및 coarse path를 **replacement**. 별도 auxiliary learned coarse cycle 제거.
* C↔H→LOCK controller, setup 비용을 포함한 break-even, 온라인 scalar observations.
* 최대 16×16 spatial micro-model, 4-cycle refresh, hysteresis, selected-row / full execution dispatch.
* Coupled line smoother는 whole-line replacement만 허용.
* Optional fused restriction/statistics의 **lagged** feature 사용을 명시.
* FP32 inference→CPU FP64 coefficients 및 solver. MPS/CUDA generation dispatch와 synchronization 계측.
* Transactional rejection, trial도 cycle budget에 포함, lock 이후 NN 미호출.

## 학습/실험

* Normalized-stencil/gain 모델 재사용, zero-delta learned P 재도입.
* Sparse value autograd로 Galerkin와 LU를 연결. Dense NxN gradient를 만들지 않음.
* 6-stage trainer와 checkpoint/optimizer/epoch/operator-index atomic resume.
* Stage acceptance는 held-out tune의 실제 time-to-tolerance. Auxiliary loss만 개선되면 채택하지 않음.
* Local balanced BCE/ranking 및 global expected time/instability/efficiency supervision.
* Global controller 재학습 후 full hierarchical replay + 표시된 offline exploration.
* 연구 기본 train 224 operators, 독립 tune/audit 및 normalized operator digest leakage 검사.
* A–J, additive vs replacement, level selection, mixed precision, refresh, cold/warm, RHS reuse CLI.

## 명확히 고친 계약

* Smoother gate-off만으로 learned P가 classical P가 되는 것은 아님. 전체 classical 우회 조건을 구분.
* FP64 true residual 확인과 universal convergence guarantee를 구분.
* Setup 또는 rejected cycle을 시간/iteration에서 숨기지 않음.
* 이미 충분히 생성된 bank를 dtype 전환으로 매 RHS 재생성하는 문제 방지.
* Classical-only/uncertified-fast-bypass에서 불필요한 heavy model signature/compile 제거.
* Nested coarse-operation inclusive timing을 중복 합산하지 않도록 수정.
* Config가 달라진 resume를 거부하고 완료한 audit를 튜닝에 재활용하지 않음.
* 성능이 검증되지 않았다는 이유로 실행 코드/학습 스크립트 전달을 막지 않도록 구현 상태와 성능 인증을 별도 저장.
