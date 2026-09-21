# Three-pillars 로컬 검증 기록

범위: original main `0d20cb26762b7af1add697dfa4a41c7e4c760e0a`에서 수정한 source.
Linux x86_64, Python 3.13, Torch 2.10.0+cpu, SciPy 1.17.0, NumPy 2.3.5,
Torch/BLAS 1-thread, native stencil C++17 (`--no-openmp`). GitHub CI는 별도 Python 3.12 환경에서 검증한다.

## 회귀 테스트

`python -m pytest --junitxml=...`: **546 passed**.
기존 테스트, batched-zebra 패치 테스트, 새 three-pillars 테스트를 모두 포함한다.
축소 final/OOD 계획 테스트는 실제 본 연구의 final dataset을 소비하지 않는다.

## 실제 CLI smoke

`configs/v6_7_three_pillars_smoke.json`, 새 RUN, 이전 checkpoint 없음.

- Calibration train 14 + tune 14 operators, n=7/15, 16 classical 전략, 서로 다른 RHS 2개.
- NN train/validation은 calibration과 normalized-operator-disjoint하게 각각 14개 생성.
- H_S 및 H_P 각각 **8/8 optimizer updates**, skipped update 없음.
- cold/warm/실제 multiple-RHS 1·4, repeats=3, warmups=1.
- fixed_C, strong_C, H_S, H_P 모두 validation **14/14 성공**.
- cold run의 14 cases × 3 repeats 전체에서 H_S/H_P 모두 setup failure 0, rollback 0.
  H_S accepted neural cycles 507; H_P accepted neural cycles 477 및 learned-transfer applies 954.
- 별도 channel n31 demo: H_S residual 6.57e-9, 33 cycles; H_P residual 9.49e-9, 30 cycles.
- 완료된 calibration/train을 `--resume`으로 재호출하는 명령 정상 완료.

## 성능 해석

| regime | H_S / strong_C speedup | H_P / strong_C speedup |
|---|---:|---:|
| cold, RHS 1 | 0.732 | 0.512 |
| warm, RHS 1 | 0.780 | 0.763 |
| multiple, RHS 4 | 0.759 | 0.642 |

비율은 `T_strong_C/T_method`, 동일 성공 cohort의 geometric mean이다.
따라서 **이 짧은 smoke에서는 Neural이 strong_C보다 빠르지 않다.**
P support/complexity 경로가 실제로 동작하며 fallback-only 결과가 아님을 확인한 것이지,
1.05~1.15 cycle budget이나 end-to-end speedup을 달성했다는 인증이 아니다.
수치는 Linux의 작은 개발 set에 대한 진단값이며 macOS/MPS/A100 수치나 최종 논문 수치가 아니다.
전체 1120-update 학습 및 독립 final/OOD 성능 검증은 제공 runbook으로 별도 실행해야 한다.
