# 문서 안내

브랜치: **`residual-u-net-teacher`**. 현재 진입 경로는 **BAMG + energy teacher → λ=10·30·50별 Weight CNN 학습 → validation 기반 격자별 선택 → freeze → final/OOD**다. 최신 추가 기능은 `row9`·`symmetric9`의 9채널 행렬 특징이며, 기존 10채널 경로도 비교·재현용으로 유지한다. 브랜치 이름이 현재 U-net teacher 사용을 뜻하지는 않는다.

새 사용자는 [영문 README](../README.md) 또는 [국문 README](../README_KR.md)에서 시작한다. 저장소의 기본 연구 설정은 특징 모드·데이터 규모마다 **scratch/warm × λ 3개 × seed 3개 = 18개 학습**이다. 로컬 scratch 전용 수정본을 기본 설정으로 설명하지 않는다.

## 현재 실험 문서

| 문서 | 내용 |
| --- | --- |
| [bamg_feature9.md](bamg_feature9.md) | 최신 `row9`/`symmetric9` 특징, 실제 입력 28채널, warm 변환, 동일 development cohort, 전용 실행기 |
| [bamg_energy_cnn_grid_routing.md](bamg_energy_cnn_grid_routing.md) | λ=10·30·50, 18개 학습, 560/1,120 TRAIN, 280/56 validation, 격자별·전역 λ 선택, final/OOD |
| [bamg_cnn_low_memory_recovery.md](bamg_cnn_low_memory_recovery.md) | 기존 10채널 routing의 label 메모리 부족 복구, 한 case씩 로딩, 실행 provenance와 재개 |
| [bamg_source_freeze_compatibility.md](bamg_source_freeze_compatibility.md) | 기존 source freeze digest 호환성, 보존해야 할 source/teacher/run 해시 |
| [bamg_energy_cnn_initialization_comparison.md](bamg_energy_cnn_initialization_comparison.md) | 단일 선택 λ의 scratch/warm 실험, epoch shuffle, 층화 데이터, 공통 1,270개 holdout 설계 |
| [bamg_energy_teacher_comparison.md](bamg_energy_teacher_comparison.md) | CNN 없는 수치 teacher 비교, 기본 λ 0·0.01·0.1·1·5·10·30·50, 정규화 목적함수, 과거 pilot 구분 |
| [bamg_compact_weight_study.md](bamg_compact_weight_study.md) | Fixed bilinear support Weight CNN source 경로, warm initialization, 학습·검증 |
| [bamg_two_cnn_study.md](bamg_two_cnn_study.md), [검증 기록](bamg_two_cnn_validation.md) | 기반 BAMG teacher·CNN 경로와 해당 시점의 검증 기록 |
| [Q1_FEM_KR.md](Q1_FEM_KR.md) | Q1 요소, 경계, quadrature, discrete manufactured RHS와 FEM load 구분 |

## 명령의 범위

- `run`은 라벨 생성·학습·development validation·격자별 선택까지만 수행한다.
- `freeze`는 완료된 학습과 선택 근거를 검증하고 모델·규칙·데이터·출처를 고정한다.
- `evaluate-final`이 final과 모든 OOD를 평가한다. 처음에는 `--resume` 없이 시작하고, 중단 후 같은 평가를 재개할 때 붙인다.
- `report`와 `report-final`은 저장된 결과를 읽는다. 학습이나 평가를 새로 수행하지 않는다.

각 output에는 처음 선택한 실행기를 계속 사용한다. Feature9는 자체 label 메모리 절약 경로를 포함한다. 기본 routing 문서의 원래 실행기와 저메모리 adapter의 차이는 저메모리 문서에서 확인한다. 데이터 규모·특징 모드·학습 예산은 final/OOD를 열기 전에 development validation으로 결정한다.

## 이전 실험과 독립 비교 경로

아래 문서의 “새”, “주 경로”, “현재”, “기준 main”은 각 문서를 작성한 시점의 문맥이다. 해당 실험의 실행기·설정·결과에만 적용하며, 위 routing/feature9 경로의 구성으로 읽지 않는다.

| 경로 | 문서 |
| --- | --- |
| EM5 모방 P → P 고정 줄 smoother → 공동 미세조정 | [em_line_study.md](em_line_study.md) |
| 같은 P의 경량 줄 smoother, 선택적 온라인 적응 | [light_smoother_study.md](light_smoother_study.md), [warm validation](light_smoother_warm_validation.md), [figures](light_smoother_figures.md) |
| Line_alt 고정 + P-only Frobenius 학습 | [learned_p_study.md](learned_p_study.md), [learned_p_runbook.md](learned_p_runbook.md), [아키텍처](ARCHITECTURE_KR.md) |
| EM5 기반 U-net teacher → CNN student | [RESIDUAL_UNET_TRANSFER_KR.md](RESIDUAL_UNET_TRANSFER_KR.md) |
| Direct CNN pruning / TTT | [direct_pruning_ttt.md](direct_pruning_ttt.md) |
| EM5 기반 direct CNN + sparse smoother | [direct_sparse_smoother.md](direct_sparse_smoother.md) |
| Bilinear 기반 P·pre/post smoother 공동 학습 | [joint_transfer_study.md](joint_transfer_study.md) |
| EM-affine P | [EM_AFFINE_HP_RUNBOOK_KR.md](EM_AFFINE_HP_RUNBOOK_KR.md) |
| P 개선 여지 진단 / 후속 제안 | [P_HEADROOM_RUNBOOK_KR.md](P_HEADROOM_RUNBOOK_KR.md), [HP_HSP_RESEARCH_PLAN_KR.md](HP_HSP_RESEARCH_PLAN_KR.md) |
| Warm-first H_S | [WARM_STUDY_RUNBOOK_KR.md](WARM_STUDY_RUNBOOK_KR.md) |
| H_S / world model | [HS_WORLD_THESIS_RUNBOOK_KR.md](HS_WORLD_THESIS_RUNBOOK_KR.md), [WORLD_MODEL_SEQUENCE_RUNBOOK_KR.md](WORLD_MODEL_SEQUENCE_RUNBOOK_KR.md) |
| Three-pillars | [THREE_PILLARS_RUNBOOK_KR.md](THREE_PILLARS_RUNBOOK_KR.md), [검증 기록](THREE_PILLARS_VERIFICATION_KR.md) |
| Strong-aware 설계·계획·결과 | [아키텍처](STRONG_AWARE_ARCHITECTURE_KR.md), [계획](STRONG_AWARE_PLAN_KR.md), [결과](STRONG_AWARE_RESULTS_KR.md), [완료 기록](STRONG_AWARE_COMPLETION_KR.md) |
| Strong-aware 데이터 / transfer / smoother / policy | [데이터](research_data_notes.md), [transfer](research_transfer_notes.md), [smoother](research_smoothers_notes.md), [policy](research_policy_notes.md), [generation](research_generation_notes.md) |
| 알고리즘·이전 요구사항·수치 기반·참고자료 | [알고리즘](ALGORITHM_AND_REFERENCES.md), [요구사항](REQUIREMENTS_COVERAGE_KR.md), [provenance](BASELINE_PROVENANCE.md), [references](REFERENCES.md), [원 요구사항](USER_REQUIREMENTS_KR.txt) |
| v6.6 패키지 | [영문 README](legacy/README_v6_6.md), [국문 README](legacy/README_v6_6_KR.md) |

`*_VALIDATION.json`, `BASELINE_PROVENANCE.json`, 과거 package manifest와 `legacy/`의 기록은 각 스냅샷에 대한 것이다. 현재 소스의 성능 증거로 해석하지 않는다. 최신 테스트 실행과 소스 체크섬 명령은 [현재 README](../README.md#verification-and-earlier-work)를 따른다. 이 문서 정리는 실행 코드·설정·학습 결과를 변경하지 않는다.
