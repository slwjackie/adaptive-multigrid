# 요구사항 반영 및 검증 구분

| 요구 범위 | 전달 구현 | 구분/주의 |
|---|---|---|
| A–B | standalone C/H/LOCK, eager classical | 기존 classical 함수 재사용 |
| C | 지정 smoothing sweep replacement | 1.05~1.15 cycle cost는 목표/계측값이지 강제 보장 아님 |
| D | 독립 static learned P/R/Ac hierarchy | coarse correction 1회, residual rebuild 없음 |
| E | normalized direction + level gain | D^-1을 coefficient에 접어 raw r에 적용 |
| F | FP32 generation / FP64 numerical solve | MPS FP64 거부, CPU 이동 후 변환 |
| G | C++ stencil, SciPy sparse kernels, no heavy NN in cycles | V-cycle recursion과 일부 line orchestration은 Python에 남음; 전체 C++ solver라는 주장 아님 |
| H–J | tiny temporal, break-even, efficiency, 5% time margin | 대체 action 성능은 offline 추정, 매 cycle shadow probe 없음 |
| K–M | conditional block detector, refresh, semantic-mask-preserving dispatch | line smoother는 whole line 단위; lagged feature는 명시적으로 이전 restriction 사용 |
| N | transactional rollback + permanent lock | 임의 문제의 절대 안정성 향상에 대한 이론 보장은 아님 |
| O | 6단계 trainer, multi-cycle graph, measured stage acceptance | Stage6 offline exploration은 학습 상태 생성일 뿐 positive label이 아님 |
| P | research train 224 / tune 28 / audit 35 구성 | 이 전달에서 전체 장기 연구 학습을 완료했다고 주장하지 않음 |
| Q | fine / first2 / all-level ablation | 실제 time-to-tolerance로 비교하는 CLI |
| R | lazy setup, same A RHS reuse, freshness checks | first hybrid setup 비용 포함 |
| S–T | raw JSONL/JSON/CSV, timing breakdown, A–J 및 추가 ablations | 전체 큰 benchmark/ablation은 실행 스크립트 제공; 실제 실행 내역은 검증 보고서 확인 |
| U | MPS/CUDA generator dispatch, synchronize 비용 계측 | 실측 M2 Pro/A100 없음 |
| V | v6.6 수치 구성요소·회귀 테스트 유지 | 기존 one-way API는 legacy 이름으로 유지 |
| W | ZIP, 문서, config, raw 실행 확인 결과, SHA-256, tests | full performance certified는 실제 audit 통과 시에만 True |

## Training의 정확한 미분 범위

* Neural smoother/P 값과 Galerkin product 및 coarse solve에 대한 1차 gradient.
* Classical support/top-k/drop 결정은 frozen-selection surrogate. 다음 forward에 실제 A로 재계산.
* 고정 action/형상 비용은 상수일 수 있으므로 "compute gradient가 가중치를 빠르게 한다"고 주장하지 않음.
* Online mode/clock은 직접 미분하지 않고 offline measured cost로 감독.
* Gate product STE는 forward production hard gate, backward soft product. 임의의 exact mathematical gradient가 아니라 지정된 surrogate.
