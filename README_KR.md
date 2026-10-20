# EM 모방 P와 줄 smoother 학습

현재 주 실험은 **EM5 모방 P 사전학습 → P 고정 줄 smoother 학습 → 전체 V-cycle
공동 미세조정**이다. [새 설계와 Mac 전체 실행 명령](docs/em_line_study.md)을 따른다.
`scripts/run_v6_7_em_line_study.py`를 새 run에서 실행하며, 아래 P-only 경로와
기존 실험 기록은 재현을 위해 유지한다. [영문 현재 README](README.md)

## 이전 실험: line_alt 고정 + learned P

작업 브랜치는 **`residual-u-net-teacher`**다. 아래 내용은 이전
`run_v6_7_learned_p_study.py`의 **classical `line_alt`를 고정하고 작은 CNN으로
P만 학습**하는 구조를 설명한다. 이하의 “현재”는 그 P-only 실험의 문맥이며,
그 경로에는 NN smoother·U-net teacher·TTT·GRU 정책이 없다.

- [현재 설계와 학습 목표](docs/learned_p_study.md)
- [Mac 실행: smoke → 420개 A 학습 → 직접 P 평가 → 재개](docs/learned_p_runbook.md)
- [현재 코드 위치](docs/ARCHITECTURE_KR.md)
- [docs 전체 안내와 과거 실험](docs/README.md)

## 현재 구성

| 항목 | 내용 |
| --- | --- |
| 대상 | 합성 2차원 확산 PDE의 SPD 선형계 Ax=b |
| Coarsening | full 고정 |
| Smoother | 모든 nonterminal level에서 `line_alt`, pre/post 각 2회 |
| 학습 대상 | Width-8 residual CNN이 생성하는 희소 P, 생성기 파라미터 3,208개 |
| Restriction / coarse 행렬 | R=Pᵀ, Ac=PᵀAP |
| 최하위 풀이 | 내부 격자 3×3에서 LU |
| 초기 P·fallback | `line_alt + bilinear + full` |
| 독립 비교군 | `line_alt + EM5 + full`; learned setup에서는 EM 실행 안 함 |
| 손실 | 전체 V-cycle 오차 전달행렬의 Frobenius norm 제곱 / 미지수 수 |
| 최종 성능 기준 | 같은 tolerance까지의 setup + solve 총시간 |

새 A의 각 level에서 CNN을 한 번 실행하고, 생성한 P/R/Ac와 수치 분해를 같은 A의
모든 cycle·RHS에서 재사용한다. A가 바뀌면 계층을 갱신한다. 현재 기본 경로는
느리다는 이유만으로 classical로 바꾸지 않으며, 잔차 증가·비정상 값·생성 실패에 대한
fallback과 tolerance 근처의 classical 마무리를 사용한다.

## 현재 실행 순서와 예산

먼저 `scripts/run_v6_7_learned_p_study.py`의 `prepare → train → diagnose`를 실행한다.
그 뒤 [실행 안내](docs/learned_p_runbook.md)의 세 비교군 평가를 수행한다.

| 구성 | Train A | Validation A | Updates |
| --- | ---: | ---: | ---: |
| 저장소 smoke 설정 | 7개, n=7 | 7개, n=7 | 14 |
| 저장소 research 설정: 예비 예산 | 84개, n=15/31/63 | 28개, n=15/31/63/127 | 336 |
| 현재 확대 실행안 | **420개, n=15/31/63** | **140개, n=15/31/63/127** | **4,200** |

확대 실행안은 runbook의 명령으로 별도 설정 파일을 만드는 방식이다. 저장소 기본
JSON이나 이미 준비한 run의 설정을 바꿨다는 뜻은 아니다. 4,200 updates는 학습
시작 예산이며 충분한 학습이나 속도 개선을 보장하는 숫자가 아니다.

일반 `benchmark` CLI는 `distilled_P`도 요구한다. 직접 학습 직후에는 runbook에 있는
기존 평가 API 호출로 **fixed_C / em5_C / learned_P**만 평가한다. 이 결과는 development
validation이며, 네 비교군 `freeze/final`의 필수 증거를 대체하지 않는다.

직접 P 학습의 효과를 확인한 다음 선택적으로 `search → distill`을 실행한다. 이 경로도
P만 최적화하며 NN smoother 공동 학습과 다르다. Final/OOD는 개발 중 학습·선택에
사용하지 않는다.

## 결과를 읽는 기준

`diagnose`는 동일한 A·고정 오차 벡터의 학습 전후 손실과 실제 cycle·시간을 비교한다.
Before 시간은 초기 CNN이 bilinear P를 생성하는 경로의 시간이다. 순수 classical 대비
속도비는 별도 평가의 `fixed_C`를 기준으로 읽는다. 손실 감소, cycle 감소, 총시간 감소는
서로 다른 결과이며 `rollback=0`도 모든 cycle이 learned P였다는 뜻은 아니다.

Smoke와 단위 테스트 통과는 실행 검증이다. 확대 학습의 연구 규모 성능이나 classical
대비 우위는 아직 확인되지 않았다. Sloshing·전단 혼합층 등 CFD 시간 적분기는 이
브랜치의 현재 실험에 포함되지 않는다.

이전 U-net, pruning/TTT, sparse smoother, joint, strong-aware 실험은
[과거 문서 목록](docs/README.md)에 남겨 두었다. 과거 수치와 조건을 현재 결과에
합쳐 해석하지 않는다. 이전 v6.7 패키지 README는
[변경 전 커밋](https://github.com/slwjackie/adaptive-multigrid/blob/6df5ed7c1bb930600b56b9746e59c520c345273c/README_KR.md)에서 확인할 수 있다.
