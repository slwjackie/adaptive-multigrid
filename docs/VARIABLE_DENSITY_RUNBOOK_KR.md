# 변밀도 비반응 유동을 주 실험으로 실행

주 실험은 **He/N₂ 비반응 혼합 유동**, 추가 적용 사례는 **기존 H₂/air 연소**다.
두 실험은 같은 Classical MG/H_S, fixed-P, 실제 raw L2 residual 기준과
paired CFD 평가 코드를 공유한다. 데이터와 학습 결과는 각각 새 폴더로 분리한다.
H₂ checkpoint를 새 기체의 학습 완료 모델로 취급하지 않는다.

| 구분 | 주 실험 | 추가 적용 사례 |
|---|---|---|
| 물리 사례 | 변밀도 He/N₂ counterflow mixing | H₂/air counterflow combustion |
| 생성기 | `integrations/openfoam13/case/generate_variable_density.py` | `integrations/openfoam13/case/generate.py` |
| 전체 실행기 | `scripts/run_v6_7_variable_density_pipeline.py` | `scripts/run_v6_7_h2_pipeline.py` |
| 설정 | `configs/v6_7_variable_density_fixed_p.json` | `configs/v6_7_h2_fixed_p.json` |
| 비교 필드 | T, He, N₂, ρ, U, p | T, H₂, U, p |
| 반응 확인 | 저장 시점마다 Qdot = 0 | 양의 Qdot 관측 |
| 현재 World Model | 꺼짐 | 꺼짐 |

## 먼저 실행할 명령

저장소 루트에서 OpenFOAM Foundation **13**과 Python 환경을 활성화한다.
기존 H₂ preflight 파일, Cantera, CHEMKIN 변환은 새 주 실험에 필요하지 않다.

```bash
source /opt/openfoam13/etc/bashrc
source .venv/bin/activate
python -m pip install -e '.[dev]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash integrations/openfoam13/Allwmake

python -u scripts/run_v6_7_variable_density_pipeline.py \
  --output artifacts/vd_primary_v1
```

기본 실행은 31×31 격자, Δt = 1e−5 s, 종료 시각 0.002 s다. 각 사례에서
200 timestep × 2 pressure correction을 기록한다. 300 K에서 가벼운 기체
입구의 He 질량분율 0.65/0.95는 train, 0.80은 validation, 0.90은 test다.
가벼운 기체/무거운 기체의 유입 속도는 각각 1/0.5 m/s다. 각 물리 trajectory
전체가 하나의 split에 속하고 동일 행렬의 split 간 중복도 거부한다.
이 작은 조건 집합은 실행 확인 및 초기 비교용이다. 새 형상·유동 조건으로의
일반화를 입증하는 최종 연구 데이터셋은 별도로 설계해야 한다.

실행 순서는 실제 native CFD 수집 → 필드 검사 → 실제 LDU import → 행렬 진단
→ H_S 학습 → validation replay와 coupled CFD → freeze → test 평가다.
validation이 실패하면 test 평가로 넘어가지 않는다. 속도가 느려진 유효 결과는
실패로 숨기지 않고 기록한다. 출력 폴더가 이미 있으면 덮어쓰지 않는다.

먼저 케이스와 행렬만 확인하려면:

```bash
python -u scripts/run_v6_7_variable_density_pipeline.py \
  --output artifacts/vd_collect_v1 --stop-after collect
```

`collect`는 수집·검사·study 준비·행렬 진단까지 한다. MG paired 속도 측정은 하지 않는다.
학습과 validation까지 확인하려면 새 출력 폴더에 `--stop-after validation`을 쓴다.
격자를 바꾸려면 `--n 63` 또는 `--n 127`을 지정한다. 격자 확대는 새로운 실행으로
취급하며, timestep과 물리적 종료 시각의 적절성도 함께 확인한다.

## 결과를 읽는 순서

| 파일 | 확인할 내용 |
|---|---|
| `pipeline_status.json` | 현재 단계 또는 실패 원인, 보존된 출력 위치 |
| `recordings/<case>/physics_profile.json` | ρ > 0, 공간적 밀도차와 시간 변화, 종 질량분율의 범위·합, Qdot = 0 |
| `dataset_profile.json` | train/validation의 연속 압력 행렬 변화와 같은 timestep의 마지막 correction 간 변화 |
| `coupled_validation/benchmark_profile.json` | Classical/H_S 압력 처리 시간 비중 및 전체 CFD 시간 |
| `coupled_validation/report.json` | 모든 반복의 필드 비교, residual/continuity 검증, pressure/전체 속도비 |
| `coupled_test/` | freeze 이후 동일 형식의 독립 test 결과 |
| `summary.json` | 완료한 단계의 결과 및 진단 요약 |

비율의 CFD 분모는 `foamRun` 프로세스 시간이다. 압력 분자는 기존 native pressure
call 타이머의 합이며 행렬 추출, JSON/socket, Python admission, hierarchy setup,
solve, residual 검증을 포함한다. 따라서 **순수 MG solve 시간이나 원래 OpenFOAM
solver의 압력 비중으로 해석하면 안 된다**. `report.json`의 각 pair timing에는
`service_seconds`, `admission_seconds`, `mg_setup_seconds`, `mg_solve_seconds`,
`mg_total_seconds`, recovery와 검증 시간도 저장된다. service·MG total·하위 phase는
서로 포함 관계이므로 합산하지 않는다. 전체 wall에는 Python service 시작 비용까지
포함하고, CFD process 시간은 별도 필드로 남긴다.

설정의 `pressure_fraction_threshold=0.5`는 이 구현에서 미리 정한 진단 기준이다.
공인된 연구 기준이나 성공 보장은 아니다. `pressure_dominant_in_instrumented_run`
은 모든 반복이 유효하고 실제 CFD process 분모가 있을 때만 판정한다.
native OpenFOAM의 병목을 주장하려면 효율적인 native PCG/GAMG의 별도 프로파일도 필요하다.
단순히 압력 비중을 높이려고 classical tolerance나 알고리즘을 불리하게 바꾸지 않는다.

행렬 진단은 Frobenius 상대 변화와 전체 스케일을 제거한 변화를 함께 기록한다.
ρ가 변해도 `rho*rAU`에서 밀도 효과가 일부 상쇄되어 압력 행렬 변화는 작을 수 있다.
행렬 변화가 관측되어도 rebuild/reuse나 World Model의 이득을 입증한 것은 아니다.
test 행렬 진단은 초기 `dataset_profile.json`에 포함하지 않는다. 수집 시 test
자료의 파일·물리·수치 무결성 검사는 수행하며, 성능 평가와 모델 선택은 freeze 규칙을 따른다.

## 물리 모델과 현재 구현 범위

Foundation 13 `multicomponentFluid`의 **압축성 pressure-based solver를 낮은 Mach
조건에서 사용**한다. 엄밀한 비압축성 projection 전용 solver가 아니다. 운동량,
두 기체의 종 수송, 에너지와 압력을 실제 CFD로 계산한다. Neural MG가 대체하는
부분은 그 안의 대칭 SPD 압력 선형계다. 격자·통신 제한은 기존 bridge와 동일하다:
고정 Cartesian 2D, serial, 한 개 z layer, coupled/cyclic patch 없음.

초기 조성 경계는 y방향으로 굽어 있어 실제 2D다. 초기·입구 T는 균일하나 에너지
방정식을 계속 풀기 때문에 온도를 시간 내내 강제로 고정하지 않는다. perfectGas,
상수 종별 Cp·점도, Wilke 혼합 및 unityLewisFourier를 사용한다. 이는 제어된
수치 실험용 근사이며 정밀 다성분 확산이나 Soret 모델이 아니다.
`combustionModel none`과 `chemistry off`로 반응·화학 ODE를 제거한다.
ρ는 열역학에서 계산되고 OpenFOAM이 출력한다. 기존 `fuel/air` patch 이름은
각각 He/N₂ 및 N₂ 입구를 뜻하며 계약 파일에 그 의미를 명시한다.

Classical P는 trajectory 첫 압력 행렬에서 만들고 이후 보존한다. 현재 A에 맞춘
coarse matrix와 수치 분해, H_S는 갱신한다. 신경망 correction 실패 시 같은 P의
classical cycle로 복귀하고 그 비용도 포함한다. World Model은 이번 주 실험의
선행 단계에서 비활성화되어 있다. 이후 train/validation에서 rebuild/reuse별 비용을
측정해 동적 선택에 이득이 있는지 확인한 뒤 별도 temporal 실험으로 연결한다.
기존 [World Model 실행 안내](HS_WORLD_THESIS_RUNBOOK_KR.md)의 GRU는 solver 비용·다음
solver 상태를 예측하는 모델이며 유동장 전체의 물리 예측 모델은 아니다.

## 검증과 기존 결과 보존

```bash
python -m pytest tests/test_h2_fixed_p_*.py tests/test_h2_pipeline.py \
  tests/test_variable_density_*.py tests/test_benchmark_profile.py \
  integrations/openfoam13/case/test_*.py
python integrations/openfoam13/tests/check_wire.py
```

Python 검증에는 실제 신규 기체의 학습을 대신하는 작은 **명시적 수치 fixture**가
포함된다. fixture 결과는 실제 CFD 성능으로 보고하지 않는다. 개발 환경에는
OpenFOAM이 없어 새 case의 native 실행·mesh/time 수렴·물리 정확도·실제 가속은
여기서 검증되지 않았다. 독립 C++ decoder 검사는 가능하지만 Unix socket 생성이
차단된 환경에서는 `check_wire.py --ipc`를 대상 WSL에서 실행해야 한다.

새 소스·설정을 기존 frozen run에 덮어씌워 재개하지 않는다. source/config hash가
달라지므로 새 폴더에서 학습한다. H₂ 추가 사례도 [기존 실행 안내](H2_FIXED_P_RUNBOOK_KR.md)를
따르되 이전 결과의 재현에는 해당 결과를 만든 원래 commit을 사용한다.

사전 확인한 Foundation 13 구현:
[thermo 등록](https://github.com/OpenFOAM/OpenFOAM-13/blob/master/src/thermophysicalModels/multicomponentThermo/psiMulticomponentThermo/psiMulticomponentThermos.C),
[const/hConst 조합](https://github.com/OpenFOAM/OpenFOAM-13/blob/master/src/thermophysicalModels/specie/include/forGases.H),
[비반응 source](https://cpp.openfoam.org/v13/noCombustion_8C_source.html),
[ρ 자동 출력](https://github.com/OpenFOAM/OpenFOAM-13/blob/master/applications/modules/isothermalFluid/isothermalFluid.C).
