> **추가된 독립 실험:** [light_smoother_study.md](light_smoother_study.md) — 완료된 EM-P를
> 고정한 242-parameter MLP / 26-coefficient ridge 줄 smoother, 선택적 온라인 적응,
> 실측 비용·출처·resume와 Mac 전체 명령. 아래 기존 EM-line·P-only 문서는 그대로 유지한다.

# 문서 안내

현재 브랜치: **`residual-u-net-teacher`**. 현재 주 실험은 **EM5 모방 P 사전학습 →
P 고정 줄 smoother 학습 → 전체 V-cycle 공동 미세조정**이다. 브랜치 이름은
이전 U-net 실험에서 이어졌으며 현재 경로가 U-net teacher를 사용한다는 뜻은 아니다.

## 현재 실험에서 읽을 문서

| 문서 | 내용 |
| --- | --- |
| [em_line_study.md](em_line_study.md) | 새 3단계 학습, 학생 coarse 행렬의 EM 정답, 줄 CNN, 재사용·실측 선택, Mac 전체 명령 |
| [PLAIN_CNN_ABLATION_KR.md](PLAIN_CNN_ABLATION_KR.md) | Residual 연결이 없는 같은 깊이의 plain5 CNN, P/smoother별 선택과 실행 설정 |
| [learned_p_study.md](learned_p_study.md) | 이전 P-only 경로의 PDE·P 제약·Frobenius 손실·fallback |
| [learned_p_runbook.md](learned_p_runbook.md) | 이전 P-only의 설치, 확대 학습·평가·재개 |
| [ARCHITECTURE_KR.md](ARCHITECTURE_KR.md) | 이전 P-only 및 공통 코드 위치와 데이터 흐름 |
| [ALGORITHM_AND_REFERENCES.md](ALGORITHM_AND_REFERENCES.md) | P와 smoother의 역할, surrogate 손실과 조건부 수렴 논의 |
| [../README.md](../README.md), [../README_KR.md](../README_KR.md) | 저장소 진입 안내 |

새 `v6_7_em_line_research.json`은 420개 학습 / 140개 validation과
4,200 / 1,400 / 1,400 updates를 명시한다. 이전 P-only research JSON과 별개다.
이 예산은 시작 설정이며 연구 규모에서 속도 우위를 보장하거나 보고하는 수치가 아니다.

## 과거 실험과 독립 비교 경로

아래 문서의 “새”, “주 경로”, “현재”, “기준 main”은 각 실험을 작성한 시점의 문맥이다.
현재 EM-line 설정에는 적용하지 않는다. 경로와 과거 결과는 재현을 위해 보존한다.

| 경로 | 문서 |
| --- | --- |
| Line_alt 고정 + P-only Frobenius 학습 | [learned_p_study.md](learned_p_study.md), [learned_p_runbook.md](learned_p_runbook.md) |
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
| 이전 요구사항·수치 기반·참고자료 | [요구사항](REQUIREMENTS_COVERAGE_KR.md), [provenance](BASELINE_PROVENANCE.md), [references](REFERENCES.md), [원 요구사항](USER_REQUIREMENTS_KR.txt) |
| v6.6 패키지 | [영문 README](legacy/README_v6_6.md), [국문 README](legacy/README_v6_6_KR.md) |

`*_VALIDATION.json`, `BASELINE_PROVENANCE.json`, `legacy/*MANIFEST*`는 과거 특정 버전의
검증·출처 기록이다. 이번 문서 정리에서 갱신하지 않았으며 현재 소스의 성능이나
삭제된 application 기능의 제공을 뜻하지 않는다. 과거 결과에 현재 조건을 소급 적용하지 않는다.
