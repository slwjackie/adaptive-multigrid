# BAMG 두 CNN 구현 검증 기록

기준 브랜치: `residual-u-net-teacher`, 기준 commit: `f9552d325ea0aabb2ae68e92bf4737fbc192f8e3`.
검증일: 2026-11-01. 구현은 기존 경로를 수정하지 않고 별도 모듈·CLI·config로 추가했다.

이 기록은 기능 검증용 짧은 실행이다. 각 CNN 지도학습 2 update, 각 residual 미세조정 2 update만 실행했다. 반복 측정 1회이며 성능 순위의 통계적 근거로 사용할 수 없다. 기본 모델은 Mask 3,424개, Weight 4,576개 파라미터이고, 배포 시 frozen reference Weight 복사본까지 총 12,576개 파라미터를 포함한다.

실행 환경은 CPU, CSR/FP64, Python 3.12, PyTorch 2.15.0+cpu, torch/BLAS thread 1이다. 저장소 CI의 PyTorch 2.10 환경과 버전이 다르므로 CI에서도 재검증할 수 있다. Native stencil 빌드는 성공했지만 새 benchmark는 CSR backend를 명시적으로 사용했다.

## 구현된 학습과 상태

- BAMG numerical teacher가 연결 mask와 P 값을 생성한다. Test vector 16개, smoothing sweep 2/4/8 혼합, 초기 구성 포함 setup pass 2회, generalized eigen/adaptive error enrichment를 사용했다.
- Teacher의 ridge는 `mu=1e-6`으로 P 자체의 제곱합을 억제한다. Bilinear와의 거리 항은 없다.
- Stage1은 top-k 이전 logits의 BCE와 teacher P 값의 회귀, 보간 action 손실을 사용한다. 실제 smoke 계수는 mask=1, weight=1, action=0.1이다.
- Stage2는 Mask CNN과 reference Weight CNN, 저장한 문제·level별 boolean mask를 고정하고 actual Weight CNN만 업데이트했다. `K=3→6`, 기본 `omega_k=1/K`, 비교 `omega_K=1`이다.
- Reference 계층은 frozen 모델로 mask를 정하고 actual 계층은 현재 Weight CNN으로 P와 Galerkin A를 다시 만든다. 한 rollout 안에서는 해당 P/계층을 재사용한다.
- Actual coarse 행렬, line solve와 coarsest solve를 포함한 역전파 및 runtime parity를 검사했다. 선택된 가중치가 0이 되어도 구조적 연결은 삭제하지 않는다.

## 수행한 검증

- 신규 테스트 64개: teacher rank/제약/mass/bootstrap, direct decoder, CNN 독립성, mask/cache/checkpoint 불변성, finite-difference gradient, hard-probe 뒤 gradient, warm setup 재실행 방지, loss/rate, 실패 및 fallback 집계, 실제 n=7/15 전체 CLI와 재개.
- 기존/신규 전체 pytest 회귀: **1,195 passed, 0 failed, 0 skipped**, 307.09초. 경고 2개는 pytest plugin deprecation과 기존 PyTorch sparse invariants 경고였다.
- Native 빌드: `python scripts/build_native_stencil.py --no-openmp` 성공.
- 기존 소스 SHA manifest 4개 검증 성공.
- 처음 실행한 기존 회귀 중 6건은 다른 구현 작업이 소스를 쓰는 동안 provenance 검사가 변경을 감지해 실패했다. 이후 소스 고정 상태에서 전체 재실행했으며, 수치 실패를 숨기기 위해 snapshot 검사를 제거하지 않았다.

Raw와 deployed는 별도 측정했고, 아래 smoke cohort에서는 모든 method·RHS가 raw로 수렴해 fallback은 사용되지 않았다. Warm은 hierarchy/factors 준비 후 solve부터 종료 검사까지이며, Cold에는 새로운 reference mask 생성과 실제 hierarchy setup을 포함한다. 수치 teacher 자체의 setup도 제외하지 않았다. 모델 파일 로드는 포함하지 않는다.

## n=7, 7개 PDE 계열

| 비교군 | Setup (ms) | Warm RHS=1 (ms) | Cold RHS=1 (ms) | Setup+순차 16 RHS (ms) | Raw 성공 |
|---|---:|---:|---:|---:|---:|
| fixed_C | 6.118 | 4.390 | 10.507 | 85.106 | 112/112 |
| em5_C | 8.633 | 5.482 | 14.115 | 89.634 | 112/112 |
| ls_once_C | 353.291 | 1.848 | 355.139 | 380.673 | 112/112 |
| bootstrap_C | 705.620 | 2.265 | 707.885 | 733.176 | 112/112 |
| bootstrap_two_cnn_supervised | 17.382 | 6.953 | 24.335 | 108.071 | 112/112 |
| bootstrap_two_cnn_finetuned_uniform | 21.260 | 6.112 | 27.371 | 108.449 | 112/112 |
| bootstrap_two_cnn_finetuned_terminal | 18.166 | 7.197 | 25.364 | 107.215 | 112/112 |

Smoke 선택 결과: `bootstrap_two_cnn_finetuned_uniform`. 선택 대상은 세 learned endpoint이며, 전체 비교군 최속은 `ls_once_C`였다.

## n=15, near-isotropic 1개 계열

| 비교군 | Setup (ms) | Warm RHS=1 (ms) | Cold RHS=1 (ms) | Setup+순차 16 RHS (ms) | Raw 성공 |
|---|---:|---:|---:|---:|---:|
| fixed_C | 25.073 | 7.390 | 32.463 | 94.514 | 16/16 |
| em5_C | 18.033 | 4.295 | 22.328 | 228.553 | 16/16 |
| ls_once_C | 2896.437 | 10.538 | 2906.975 | 2993.217 | 16/16 |
| bootstrap_C | 4384.766 | 6.744 | 4391.510 | 4465.027 | 16/16 |
| bootstrap_two_cnn_supervised | 34.243 | 19.110 | 53.353 | 288.165 | 16/16 |
| bootstrap_two_cnn_finetuned_uniform | 57.336 | 20.741 | 78.076 | 336.253 | 16/16 |
| bootstrap_two_cnn_finetuned_terminal | 37.221 | 24.385 | 61.606 | 295.984 | 16/16 |

Smoke 선택 결과: `bootstrap_two_cnn_supervised`. 선택 대상은 세 learned endpoint이며, 전체 비교군 최속은 `em5_C`였다.

## 해석과 미실행 범위

학습된 CNN 모델의 warm 시간은 이 짧은 smoke에서 classical bilinear/EM5보다 느렸다. 속도 개선을 구현 성공 조건으로 강제하지 않았다. n=15의 한 계열 결과를 7개 계열 또는 큰 격자 결과로 확장해서 해석하면 안 된다.

P1/Q1 research config(280 TRAIN, 56 validation), 장시간 teacher 생성·지도학습·미세조정, untouched final/OOD, k=8·모델 축소·fixed-bilinear support·시간 가중 손실의 성능 비교, GPU 병렬 추론은 실행하지 않았다. `fixed_bilinear`, `late|none`, 고정-mask LS warmup, 시간 profile 기반 objective의 코드/API 지원과 해당 성능 검증은 구분한다. Student 분포 teacher refresh와 실제 병렬 실행/architecture 자동 sweep은 추가 연구 작업이다.

기본 CLI는 validation에서 완료된 supervised/uniform/terminal endpoint를 warm RHS=1로 선택한다. 모든 중간 checkpoint를 평가해 최적 epoch를 탐색한 결과가 아니다. 실패율 기준을 충족하지 못한 모델을 승자로 선택하지 않으며 실패는 전체 cohort와 시간 벌점에 포함한다.

## 재현 명령

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python scripts/run_v6_7_bamg_study.py run \
  --config configs/v6_7_bamg_smoke.json --output artifacts/bamg_smoke

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python scripts/run_v6_7_bamg_study.py run \
  --config configs/v6_7_bamg_smoke_n15.json --output artifacts/bamg_smoke_n15

python scripts/run_v6_7_bamg_study.py run \
  --config configs/v6_7_bamg_smoke.json --output artifacts/bamg_smoke --resume

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -m pytest -ra
```

세부 수식·파일·개별 단계 실행·research 예산은 [실행 안내](bamg_two_cnn_study.md)를 따른다. 같은 출력 경로에서 소스/config/hardware 환경이 바뀌면 캐시와 checkpoint 재개를 거부한다.
