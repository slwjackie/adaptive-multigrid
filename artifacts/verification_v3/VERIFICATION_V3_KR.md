# Adaptive Neural Multigrid v3 — 최종 검증 보고서

## 1. 검증 범위

이번 검증은 다음을 대상으로 했습니다.

1. 수치·구조 회귀 테스트
2. 무작위 PDE/strategy/extreme NN stress test
3. 강한 classical smoother·transfer·semicoarsening benchmark
4. operator-dependent transfer 및 adaptive hierarchy profiling
5. sparse-memory scaling 확인
6. 짧은 cross-grid full-V-cycle 재학습
7. `n=15,31,63` targeted raw/safe 평가
8. 최종 패키지 재설치·재테스트

동봉된 짧은 checkpoint는 **구현 검증용**입니다. 충분한 epoch와 대규모 pool로 학습한 논문 성능 checkpoint가 아닙니다.

---

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| OS | Linux 6.18.35 x86_64 |
| Python | 3.13.5 |
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| PyTorch | 2.10.0+cpu |
| pandas | 2.2.3 |
| CUDA | 사용하지 않음 |

Timing은 이 CPU container에서 측정한 값이며 다른 장비의 절대시간과 직접 비교하면 안 됩니다.

---

## 3. 회귀 테스트

```bash
PYTHONPATH=src pytest -q
```

결과:

```text
........................................ [100%]
40 passed
```

주요 검사 범위:

- PDE matrix symmetry/SPD
- line relaxation과 rectangular semicoarsening
- 16-candidate transfer의 row sum·injection·rank
- tied `R=Pᵀ`와 Galerkin SPD
- zero-init classical equivalence
- full-V-cycle train/inference parity
- geometric validation aggregation
- stage rollback/common validation
- detector feature NumPy/Torch 일치
- reverse post-smoothing feature
- patch-selective NN call skipping
- exact safeguard와 shadow fallback
- CG/FCG/FGMRES edge cases
- sparse operator-dependent Torch/SciPy 일치
- invalid grid/config 사전 검증

---

## 4. Random stress validation

```bash
PYTHONPATH=src python scripts/run_stress_validation.py \
  --cases 12 \
  --output-dir artifacts/verification_v3/stress
```

| 항목 | 결과 |
|---|---:|
| Random PDE cases | 12 |
| 총 checks | 204 |
| Failures | 0 |
| Passed | true |

검사에는 random coefficient/anisotropy 문제, classical strategy 조합, extreme decoder output의 prolongation full-rank 및 Galerkin SPD, pure-classical shadow residual bound가 포함됩니다.

산출물:

- `stress/stress_results.csv`
- `stress/stress_summary.json`

---

## 5. Operator-dependent transfer profiling

검토문에 제시된 기존 측정치는 `n=31: 0.145 s`, `n=63: 0.352 s`, `n=127: 1.420 s`였습니다. 동일 container의 v3 vectorized 구현에서 2회 측정 중 최소시간은 다음과 같습니다.

| n | v3 최소시간 | 검토문 기재치 대비 |
|---:|---:|---:|
| 31 | 0.00711 s | 약 20.4× 빠름 |
| 63 | 0.01723 s | 약 20.4× 빠름 |
| 127 | 0.12866 s | 약 11.0× 빠름 |

개선 요소:

- CSR row loop 제거
- vectorized sparse candidate lookup
- vectorized relative dropping
- candidate top-k
- transfer pattern cache

절대시간은 repeat 수가 작아 benchmark noise가 있으며, 여기서는 병목 제거 여부 확인용입니다.

---

## 6. Sparse-memory 확인

| n | unknowns | sparse matrix bytes | 가상 float32 dense bytes | dense/sparse 비율 |
|---:|---:|---:|---:|---:|
| 31 | 961 | 70,820 | 3,694,084 | 52.2× |
| 63 | 3,969 | 297,124 | 63,011,844 | 212.1× |
| 127 | 16,129 | 1,216,676 | 1,040,578,564 | 855.3× |

Root matrix를 dense화하지 않는 이유가 격자 증가에 따라 급격히 커집니다. Terminal tiny solve와 local 1D line block 외에는 전체 matrix dense conversion을 사용하지 않습니다.

---

## 7. `adaptive_refresh` profiling

`n=63`, local anisotropic patch, 3 cycles, 모든 patch 활성화 조건:

| refresh | elapsed | hierarchy rebuilds | NN calls | relative residual |
|---:|---:|---:|---:|---:|
| 1 | 0.96094 s | 12 | 222 | 1.2507e-2 |
| 3 | 0.72840 s | 4 | 178 | 1.2507e-2 |

같은 수치 결과에서 elapsed time이 약 **24.2% 감소**했고 hierarchy rebuild가 12→4로 줄었습니다.

---

## 8. 실제 selective NN computation

`n=63`, 한 cycle에서 detector bias를 강제로 조정한 profile입니다.

| 조건 | active patches | smoother/transfer NN calls | total NN calls | elapsed |
|---|---:|---:|---:|---:|
| easy, 모두 skip | 0% | 0 | detector 8회 | 0.23050 s |
| hard, 모두 active | 100% | smoother 170 + transfer 85 | 263 | 0.53857 s |

모두 skip할 때 neural smoother/transfer forward가 실제로 0회이며, 이 profile에서 elapsed가 약 **57.2% 감소**했습니다. 이 수치는 강제로 만든 두 극단의 검증이며 실제 학습 detector의 절감률을 뜻하지는 않습니다.

---

## 9. 강한 classical baseline (`n=31`, 12 cycles)

통제된 9개 strategy 결과 중 최저 residual은 다음과 같습니다.

| 문제 | 최저 residual strategy | relative residual | work units |
|---|---|---:|---:|
| Poisson | `line_alt_operator_alternating` | 1.85e-12 | 844.2 |
| Aligned anisotropy | `line_x_bilinear_full` | 9.57e-15 | 133.5 |
| Rotated anisotropy | `jacobi_bilinear_full` | 5.66e-4 | 75.7 |
| Local patch | `line_alt_operator_alternating` | 3.17e-4 | 832.3 |
| Checkerboard | `line_alt_operator_alternating` | 2.70e-6 | 922.8 |
| Channel | `line_alt_operator_alternating` | 3.74e-3 | 809.5 |

다만 semicoarsening 조합은 work가 매우 큽니다. 따라서 `pareto_classical_by_case.csv`에 residual/work non-dominated frontier를 별도로 제공합니다. 예를 들어 channel에서는 저비용 `jacobi_bilinear_full`이 `4.53e-3` at 83.9 WU이고, semicoarsening 최저 residual은 `3.74e-3` at 809.5 WU입니다. 논문에서는 둘 다 제시해야 합니다.

산출물:

- `baseline_n31_final/all_classical_strategies.csv`
- `baseline_n31_final/best_classical_by_case.csv`
- `baseline_n31_final/pareto_classical_by_case.csv`

---

## 10. Verification-profile 재학습

실행 설정:

- train roots: `7,15`
- validation roots: `15,31`
- full V-cycle
- fixed classical strategy: `jacobi_bilinear_full`
- 매우 작은 pool/epoch
- 목적: 코드·gradient·validation·rollback 검증

### Stage 결과

| Stage | entry validation | best validation | accepted |
|---|---:|---:|---|
| global selector CE | 2.7337 | 2.7252 | yes |
| NN smoother ρ | 0.2114 | 0.2062 | yes |
| learned transfer ρ | 0.3327 | 0.3327 | **no, rollback** |
| combined ρ | 0.2675 | 0.2629 | yes |
| detector BCE | 0.1824 | 0.1823 | yes |
| adaptive ρ | 0.2957 | 0.2949 | yes |

Stage마다 mode와 smooth-error preprocessing이 다르므로 위 숫자를 하나의 성능 시계열로 비교하면 안 됩니다. 그래서 동일 protocol의 common validation을 별도로 저장합니다.

### Common validation

| Mode | geometric A-norm contraction |
|---|---:|
| classical | 0.28578 |
| nn_smoother | 0.26838 |
| learned_transfer | 0.28579 |
| combined | 0.26839 |
| adaptive | 0.28428 |

이는 randomized homogeneous-error validation metric이며, PDE benchmark의 residual/time 결과와 동일한 의미가 아닙니다.

산출물:

- `training_final/checkpoint_v3_verification.pt`
- `training_final/checkpoint_v3_verification.training.csv`
- `training_final/checkpoint_v3_verification.metadata.json`
- `training_final/checkpoint_v3_verification.common_validation.json`

---

## 11. Targeted cross-grid 평가

같은 짧은 checkpoint를 `n=15,31,63`, rotated/local patch/checkerboard/channel에 3 cycles 적용했습니다. 고정 classical strategy는 `jacobi_bilinear_full`입니다.

### Learned transfer

| case | n=15 classical / learned P | n=31 | n=63 |
|---|---:|---:|---:|
| Rotated | 0.2549 / 0.2550 | 0.2711 / 0.2714 | 0.2819 / 0.2828 |
| Local patch | 0.2027 / 0.2028 | 0.2252 / 0.2253 | 0.2134 / 0.2137 |
| Checkerboard | 0.5598 / 0.5596 | 0.6566 / 0.6567 | 0.6827 / 0.6829 |
| Channel | 0.2939 / 0.2940 | 0.3149 / 0.3150 | 0.3409 / 0.3409 |

기존 검토에서 관찰된 것처럼 grid level 수가 증가할수록 learned P 열화 비율이 1.75→3.29배로 폭증하는 현상은 이 짧은 v3 검증본에서는 나타나지 않았습니다. 그러나 learned P가 classical보다 의미 있게 우월한 것도 아닙니다. 해당 stage가 rollback된 결과와 일치합니다.

### Raw NN smoother와 adaptive

- 짧은 checkpoint의 raw NN smoother는 모든 targeted case에서 classical보다 나빴습니다.
- production detector gate가 threshold 0.5를 넘지 않아 active patch가 0이었습니다.
- 따라서 raw/safe adaptive 결과는 classical과 동일했습니다.
- safe 결과의 alpha는 모두 0이었으며, 이는 이 checkpoint에서 neural correction이 실질적으로 사용되지 않았다는 직접 증거입니다.

산출물:

- `training_final/cross_grid_targeted.csv`
- `training_final/cross_grid_targeted.log`

---

## 12. 결론과 남은 실험상 한계

### 코드 수준에서 해결한 사항

- two-grid/full-V-cycle 불일치
- single-grid validation
- validation aggregation 버그
- stage metric 혼재
- FCG restart 및 FGMRES residual semantics
- operator baseline Python loop 병목
- full-matrix dense training
- duplicate detector feature 및 reverse mismatch
- false selective computation
- semicoarsening 부재
- global strategy selection 부재
- raw/safe 결과 혼재
- 장기 classical 하한을 보장하지 못하던 fallback
- extreme learned P rank 위험

### 아직 논문 결과로 해결되지 않은 사항

1. 동봉 checkpoint는 짧아 raw NN smoother가 targeted PDE에서 우월하지 않습니다.
2. Learned transfer는 full-V-cycle validation 개선에 실패해 rollback됐습니다.
3. Detector가 targeted benchmark에서 비활성화됐습니다.
4. Global selector는 verification profile에서 1 epoch만 학습했습니다.
5. 최종 주장을 위해서는 기본 설정 `15,31,63` train + `127` OOD로 충분한 장기 학습이 필요합니다.
6. 반드시 strong classical Pareto baseline, raw NN, safeguarded NN, time/work/activation을 함께 보고해야 합니다.

따라서 v3은 **요청된 설계와 수치적 결함을 반영한 최종 실험 코드**이며, “NN이 strong classical MG보다 우월하다”는 결론 자체를 미리 내장하거나 보장하는 코드는 아닙니다.

---

## 13. 최종 ZIP 무결성

최종 archive를 별도 임시 경로에 다시 풀어 검증했습니다.

- ZIP compressed-data test: 통과
- Source manifest: **54/54 files OK**
- Extracted source import: 통과
- Package-root `case_suite` export: 통과
- Extracted source regression tests: **40 passed**

`SOURCE_MANIFEST.sha256`는 manifest 자신을 제외한 모든 packaged file을 포함합니다. ZIP 자체 checksum은 `/mnt/data/adaptive_neural_multigrid_v3_final.zip.sha256`에 별도로 제공합니다.
