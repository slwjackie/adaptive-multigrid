# Strong-aware 구현 완료 및 이어서 수행한 검증

기존 프로젝트에 구현을 반영했고 **494개 테스트가 모두 통과**했습니다. 기존 smoke checkpoint를 재사용한 추가 검증도 완료했습니다. 새 Neural 모델의 strong Classical 대비 속도·정확도 우위는 아직 확인되지 않았으며, production 승격이나 final 인증은 하지 않았습니다.

기존 selector/rules, numerical native kernel, 과거 checkpoint·결과·certificate를 보존했습니다. 102개 원본 파일을 반영 전 검사했고, 중요 historical artifact 47개의 hash가 반영 후에도 같았습니다. 수정된 기존 파일은 README와 models/banks/solver/unroll 연결부 다섯 개입니다.

## 멈춘 지점에서 이어 한 작업

- 첫 `smoke_01`의 모델 10개와 7/7 update, validation/정책/채택 기록을 보존했습니다.
- 설치 후에는 기존 GNN checkpoint에서 **repair update 2회만 이어서** 실행했습니다. 처음부터 학습하지 않았습니다. 한 update는 feasible learned task, 한 update는 Classical fallback task와 P repair loss였습니다.
- 이 2-update checkpoint를 평가 후보로 몰래 교체하지 않았습니다. 비교는 원래 selected student를 그대로 사용합니다.
- 같은 A에서 실패한 P/S generation도 cache하여 다음 RHS/호환 branch에서 heavy NN을 다시 실행하지 않습니다.
- Final/OOD의 frozen CNN/D4/O2/GNN 대조군, offline classical reference, expected-RHS blind ablation을 실행 경로에 추가했습니다. 실제 final/OOD 데이터는 열지 않았습니다.

## 설치 위치에서 측정한 cold 결과

새 validation 14개(n=15/31), 각 arm 3반복, 같은 A/b/x0와 FP64 tolerance·150회 예산입니다. 속도비는 `T_strong_C/T_method`입니다. 실패를 빠른 결과로 집계하지 않았고 paired 속도비는 공통 성공 12개에 한정됩니다.

| Method | 성공 | Strong C 대비 속도비 | 95% CI | Neural 사용 사례 |
| --- | ---: | ---: | --- | ---: |
| fixed_C | 14/14 | 0.366× | [0.232, 0.583] | 0 |
| strong_C | 12/14 | 1.000× | [1.000, 1.000] | 0 |
| H_S | 12/14 | 0.731× | [0.650, 0.812] | 14 |
| H_P | 12/14 | 0.697× | [0.622, 0.773] | 2 |
| H_SP | 12/14 | 0.691× | [0.598, 0.789] | 2 |
| adaptive | 12/14 | 0.707× | [0.615, 0.795] | 0 |
| adaptive_rhs_blind | 12/14 | 0.711× | [0.611, 0.806] | 0 |

Strong C와 neural branch들은 n=31 channel/variable coefficient 두 문제에서 예산 내에 목표 tolerance에 도달하지 못했습니다. Fixed C는 14개 모두 성공했습니다. 따라서 성공한 12개에서 strong C가 빠르다는 사실을 전체 14개의 견고함 우위로 해석하지 않습니다. 요청대로 selector를 수정하지 않았습니다.

Successful H_S의 manufactured L2/energy error가 strong C보다 `5% + 1e-12` 이상 큰 사례는 3개, H_SP는 2개입니다. 같은 residual tolerance만으로 solution error 우위를 주장하지 않습니다.

Adaptive는 이 표에서 neural을 사용하지 않았고 정책 검증/선택 비용이 추가됐습니다. 이를 Neural 가속으로 집계하거나 production에 승격하지 않았습니다. Smoother/P teacher가 direct student를 충분히 개선하지 못해 KD checkpoint를 만들지 않았고, joint 후보도 채택하지 않았습니다. [첫 smoke의 architecture·teacher 해석](STRONG_AWARE_RESULTS_KR.md)을 함께 보세요.

근거: [raw/JSON·CSV 요약](../artifacts/my_v6_7_strong_aware/installed_validation/branches/comparison.json), [이어 한 검증 보고서](../artifacts/my_v6_7_strong_aware/installed_validation/verification_report.json).

## 실제 multiple RHS와 사후 classical reference

실제 서로 다른 RHS를 n=15/31 near-isotropic 두 operator에서 풀었습니다. 모두 2/2 성공했지만 이 두 문제에서 RHS 1/4/16/64 범위의 amortization crossover는 관찰하지 못했습니다. 아래는 해당 batch 전체의 setup 포함 속도비입니다.

| RHS 수 | H_S | H_P | H_SP | adaptive | RHS-blind adaptive |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.613× | 0.532× | 0.498× | 0.531× | 0.560× |
| 4 | 0.675× | 0.609× | 0.627× | 0.675× | 0.629× |
| 16 | 0.802× | 0.797× | 0.771× | 0.846× | 0.860× |
| 64 | 0.865× | 0.865× | 0.737× | 0.970× | 0.951× |

RHS-blind는 동일 batch와 cache를 사용하면서 정책에 제공하는 expected RHS hint만 1로 고정합니다. 여기서는 둘 다 neural 사용이 없으므로 숫자 차이를 branch utility 학습의 효과로 해석하지 않습니다. Cold/warm의 별도 측정도 [전체 요약](../artifacts/my_v6_7_strong_aware/installed_validation/cold_warm_multiple_rhs/comparison.json)에 있습니다.

Offline classical reference는 n=15 near-isotropic 한 문제의 controlled bank에서 실행했습니다. 사후 최선은 chebyshev_bilinear_full, 성공 median 2.210 ms였고 deployable strong selector는 jacobi_bilinear_full, 2.497 ms였습니다. 전체 검색은 warmup 포함 0.658 s를 소모했습니다. 한 문제의 경험적 사후 reference이며 deployable baseline이나 theoretical bound가 아닙니다. [후보별 원시 시간과 검색 비용](../artifacts/my_v6_7_strong_aware/installed_validation/classical_reference/offline_reference.json).

## 실제 CPU/MPS generation

실제 Mac에서 CPU/MPS FP32, 모델·operator당 5반복+warmup 1회로 측정했습니다. Numerical bank는 CPU FP64입니다. 아래는 n=31 rotated operator의 learned bank 생성 시간이며, student/teacher inference뿐 아니라 device 전환·전송, sparse 조립·factorization을 포함합니다. Initial classical constructor는 별도 raw 항목입니다.

| Model | CPU bank setup | MPS bank setup | 용도 |
| --- | ---: | ---: | --- |
| ordinary_direct | 2.838 ms | 17.667 ms | student |
| compact_direct | 2.549 ms | 15.028 ms | student |
| d4_direct | 8.494 ms | 34.460 ms | student |
| o2_direct | 7.948 ms | 72.809 ms | student |
| s_teacher | 10.011 ms | 32.812 ms | offline teacher |
| p_cnn_direct | 14.432 ms | 31.783 ms | student |
| gnn_direct | 24.187 ms | 56.085 ms | student |
| edge_direct | 20.142 ms | 45.533 ms | student |
| p_teacher | 43.866 ms | 71.031 ms | offline teacher |

이번 두 operator에서는 총 bank setup이 CPU에서 더 빨랐습니다. 큰 teacher의 순수 NN 연산만 MPS가 약간 빨랐던 경우도 있지만 전송·bank 구성까지 포함하면 이득이 사라졌습니다. 이 한정된 결과로 모든 크기에서 CPU가 빠르다고 일반화하지 않습니다. CPU를 기본값으로 유지했습니다.

성공한 모든 cached bank lookup에서 추가 heavy generation 호출은 0이었습니다. n=15 near-isotropic GNN/edge/teacher P는 양 장치에서 fill cap으로 거부됐고, 실패를 성공 시간으로 집계하지 않았습니다. CUDA/A100은 사용할 수 없어 `not_measured`로 남겼습니다.

근거: [smoother CPU/MPS](../artifacts/my_v6_7_strong_aware/device_generation/smoother/generation.json), [transfer CPU/MPS](../artifacts/my_v6_7_strong_aware/device_generation/transfer/generation.json).

## 아직 실행하지 않은 연구와 비용

설치 위치의 이어 한 검증은 총 47.69초였습니다. 이 시간은 2 update, 작은 cold/multiple 비교 및 1-case bank reference의 실제 값이며 full-run 예측값이 아닙니다.

전체 설정은 train 280개, validation 56개, 후보당 1,120 updates입니다. Frozen final/OOD는 총 126개입니다. 기본 main final 설정의 측정 대상 RHS 수만 `126 × 8 arms × 5 repeats × (cold 1 + warm 1 + multiple 85) = 438,480`회이며 warmup/prime, architecture/reference 비교가 추가됩니다. Final multi-RHS는 특히 비싼 작업입니다. 수렴·sparse fill과 저장할 raw trace에 따라 시간과 디스크 비용이 크게 늘어나므로 몇 시간 이상 걸릴 수 있는 별도 연구 실행으로 취급해야 합니다. n=255의 실제 비용은 이번에 측정하지 않았습니다.

최종 architecture/OOD 비교와 certification은 [실행 안내](../STRONG_AWARE_README_KR.md)의 명시적인 full development → freeze → final 명령으로만 실행합니다. 이번 작업에서는 해당 full 학습이나 final/OOD를 시작하지 않았습니다. O2의 실제 rotation-OOD 개선, graph P의 병목 해소, KD 이득은 충분한 학습 후 이 검증을 통과해야 주장할 수 있습니다.

테스트 근거: [설치 위치 JUnit: 494 tests, 0 failures/errors/skips](../artifacts/my_v6_7_strong_aware/inspection/installed_tests.xml).
