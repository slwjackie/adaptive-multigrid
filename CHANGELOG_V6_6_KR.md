# v6.5 → v6.6 변경 내역

| 구분 | v6.6 구현 |
|---|---|
| 공간적 selective NN | Detector/gate/patch 실행 모듈 제거 |
| Adaptive 제어 | One-shot MLP가 초기 K_NN 예측 |
| 상태 전환 | NEURAL → CLASSICAL 최대 한 번; RHS마다 초기화 |
| 입력 | A 요약 + 초기 residual + tolerance + cache 상태 |
| Classical baseline | 명시적 strategy, 외부 solver 없음, 자동 global selector 없음 |
| Early NN | 같은 classical V-cycle 위의 NN smoother increment |
| 공유 hierarchy | P/R/Ac/LU/line factors를 두 phase에서 그대로 사용 |
| Learned transfer | 별도 auxiliary coarse cycle은 새 실행 경로에서 명시적으로 제외 |
| K=0 | 동일 classical 함수, NN bank 생성 없음 |
| CNN 추론 | 사용하는 level당 최초 1회, stencil/gain 재사용 |
| Switch 추론 | NumPy로 동일 tiny MLP 실행; Torch 출력과 parity 테스트 |
| Safeguard | NN 성장/NaN/Inf 결과 rollback + 같은 classical bank로 영구 전환 |
| 정체/마무리 | Scalar residual 기준으로 조기 전환, 별도 detector 없음 |
| Oracle | 각 K를 동일 초기 상태부터 tolerance 또는 budget까지 실제 실행 |
| 학습 목적 | 실측 T/T_C + lambda instability의 정책 기대값 |
| 실패 처리 | Censored time 구분; 더 빨리 실패한 후보에 속도 보상 금지 |
| 데이터 누출 | A 공통 scale 정규화 후 operator fingerprint까지 split 비교 |
| 검증 | Train / tune / audit 분리; audit로 가중치 재조정 금지 |
| 인증 | 가중치·환경·수치 설정·격자·cache scope에 묶인 temporal certificate |
| 미인증 모델 | 기본 temporal은 K=0, 연구용 predicted에서만 실제 예측 사용 |
| 정책과 초기화 | 선택적 smoother residual surrogate와 최종 실측 시간 policy loss를 구분 |
| Cycle 계측 | 수용 update와 실제 V-cycle 시도 수를 별도 저장; rollback 비용 포함 |
| 동일 A 여러 RHS | Hierarchy/NN bank 재사용, 배치 freshness 1회, 상태는 RHS별 재초기화 |
| 시간 재계상 | A 변경 시 rebuild 시간을 setup과 solve에 중복 가산하지 않음 |
| 변경 감지 | A 변경은 hierarchy 재구축; smoother 변경은 NN bank만 무효화 |
| checkpoint | 새 format 12. Legacy 가중치는 명시적 trusted migration만 허용 |
| 큰 격자 | 127/255 phase/cache 계약 테스트와 full tolerance benchmark를 구분 |

## 유지한 수치 primitive

v6.5의 grid nesting, P1 diffusion assembly, line/Chebyshev smoothers, bilinear/operator-dependent transfer, Galerkin construction, native 9-point kernel을 재사용했습니다. Classical cycle 산술은 유지하고 공간 gate와 추가 learned hierarchy가 들어가던 실행 구조를 교체했습니다.

## 중요한 제한

One-way 전환만으로 모든 classical MG가 수렴하거나 NN prefix가 반드시 이득이 되는 것은 아닙니다. Baseline 자체의 수렴과 유한한 switch state가 필요합니다. 작은 smoke policy에서 시간 우위가 나오지 않으면 그 결과를 그대로 기록하고 배포 인증을 거부합니다. `predicted` 진단 결과를 인증된 production 성능으로 표기하지 않습니다.
