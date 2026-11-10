# Hybrid teacher 기반 CNN 멀티그리드

브랜치: **`residual-u-net-teacher`** · [English / 전체 실행 예시](README.md) · [문서 목록](docs/README.md)

현재 경로는 **정렬 Q1 FEM 격자의 합성 2차원 SPD 확산 문제에서 prolongation P의 가중치를 CNN으로 생성**하는 실험이다. BAMG + energy teacher의 **λ=10·30·50**으로 각각 학습하고, scratch와 warm start를 비교한 뒤, development validation으로 **가장 미세한 격자 크기별 CNN 선택 규칙**을 정한다. 최신 추가 실험은 기존 10채널 행렬 특징과 **`row9`·`symmetric9`의 9채널 특징**을 비교한다.

브랜치 이름은 과거 U-net 실험에서 이어졌다. 아래 경로는 작은 Weight CNN과 고정된 classical line smoother를 사용한다. EM-line, U-net, P-only, learned smoother 등 이전 경로의 예산·손실·성능 결과를 현재 실험과 혼용하지 않는다.

## 실행기 선택

| 목적 | `scripts/` 아래 실행기 | 상세 문서 |
| --- | --- | --- |
| 9채널 `row9` / `symmetric9` 실험 | `run_v6_7_bamg_energy_cnn_feature9.py` | [특징·체크포인트 변환·데이터 공유](docs/bamg_feature9.md) |
| 기존 10채널 λ별 CNN + 격자 선택, 메모리 절약 | `run_v6_7_bamg_energy_cnn_routing_low_memory.py` | [메모리·재개](docs/bamg_cnn_low_memory_recovery.md), [routing](docs/bamg_energy_cnn_grid_routing.md) |
| 기존 10채널 routing 원래 실행기 | `run_v6_7_bamg_energy_cnn_routing_study.py` | [routing](docs/bamg_energy_cnn_grid_routing.md) |
| 수치 비교에서 선택한 λ 하나로 scratch/warm 비교 | `run_v6_7_bamg_energy_cnn_study.py` | [단일 λ 실험](docs/bamg_energy_cnn_initialization_comparison.md) |
| CNN 없이 수치 teacher만 비교 | `run_v6_7_bamg_energy_study.py` | [수치 비교](docs/bamg_energy_teacher_comparison.md) |

**저장소의 routing/feature9 연구 설정은 특징 모드·데이터 규모마다 18개 학습을 실행한다.** λ 3개 × scratch/warm 2개 × seed 6101/6102/6103이다. 각 실행은 지도학습과 잔차 미세조정을 포함한다. 이 실행기에는 커밋된 `--scratch-only` 옵션이 없다. 로컬에서 따로 만든 scratch 9개 전용 스크립트는 이 기본 실행 경로와 구분한다. `row9`와 `symmetric9`도 각각 별도의 명령·출력 폴더로 실행한다.

수치 teacher 기본 설정의 λ 목록은 **`[0, 0.01, 0.1, 1, 5, 10, 30, 50]`**이다. Routing은 수치 비교의 최적 λ 하나만 학습하는 것이 아니라 10·30·50 모두를 학습한다. 반면 단일 λ 실행기는 수치 비교의 선택값을 라벨 생성에 사용한다.

## 현재 수치·모델 구성

| 항목 | Routing / feature9 경로 |
| --- | --- |
| 이산화 | 정렬된 사각형 Q1 FEM, 동차 Dirichlet 경계 |
| Coarsening / 연결 | Full coarsening, fixed bilinear support, 행당 최대 4개 coarse 연결 |
| 학습 대상 | P의 부호 있는 가중치; smoother와 coarsening은 학습하지 않음 |
| CNN | 3×3 합성곱 5층, 폭 8, ReLU, 이후 16슬롯 출력 head |
| 가중치 제약 | Injection, 내부·경계 행합, 행별 절댓값 합 L1 ≤ 8 |
| Restriction / coarse 행렬 | R=Pᵀ, 각 후보의 실제 P로 A_c=PᵀAP 재구성 |
| Smoother / 최하위 | Classical alternating line, pre/post 각 2회, 내부 3×3 격자 LU |
| 미세조정 | Uniform residual loss, 6→12 cycle, random/smoothed/hard 오차 벡터 |
| 평가 | 초기 해 0, `||r||₂ ≤ max(1e-12, 1e-8 ||r₀||₂)`, fallback 없음 |
| 선택 시간 | 모델 로딩·CNN 추론·계층 생성·분해를 제외한 warm solve 시간 |

Teacher optimizer와 MG 설정은 검증된 수치 비교에서 가져온다. 실제 적용값은 출력 폴더의 `configuration.json`에 기록된다.

| 행렬 특징 | 구성 | 기하 3 + mask 16 포함 첫 층 입력 |
| --- | --- | ---: |
| 기존 10채널 | 로그 대각 + 중심 포함 행 정규화 9-point | 29채널 |
| `row9` | 로그 대각 + 주변 `A_ij/d_i` 8개 | 28채널 |
| `symmetric9` | 로그 대각 + 주변 `A_ij/sqrt(d_i*d_j)` 8개 | 28채널 |

`d_i=max(A_ii, 1e-14)`, 로그 특징은 `log(d_i/median(d)+1e-12)`이다. Feature9는 양의 대각을 확인하고 부호를 보존한다. 입력 정규화는 원래 선형계 자체를 바꾸지 않는다. 8방향 밖의 coarse fill-in을 모두 표현하지는 않으며, mask 16→4 압축도 이 실험에 포함되지 않는다.

Scratch는 무작위 가중치로 시작한다. Warm은 기존 source CNN의 가중치를 가져와 새 optimizer로 모든 층을 학습한다. Feature9 warm 변환은 첫 합성곱에서 중심 입력 열만 제거하며, **기존 함수와 동등한 변환이나 기존 학습의 resume가 아니다.** `old_cnn` 비교군은 원래 10채널 특징을 유지한다.

## 학습 규모와 격자별 선택

| 항목 | 기본 560개 | 확대 1,120개 |
| --- | ---: | ---: |
| TRAIN 격자 | 15·31·63·127 | 동일 |
| 격자·family당 TRAIN (7 families) | 20 | 40 |
| 전체 validation | 280 | 명시적 연결 시 같은 280 |
| Checkpoint 선택용 validation | 56 | 명시적 연결 시 같은 56 |
| 학습 1회당 지도학습 | 15 epoch / 8,400 update | 15 epoch / 16,800 update |
| 학습 1회당 미세조정 | 8 epoch / 4,480 update | 8 epoch / 8,960 update |
| 특징 모드당 학습 실행 | 18 | 18 |

Epoch shuffle은 완료된 매 epoch에서 모든 문제를 한 번씩 사용한다. Seed는 초기 가중치(scratch), 문제 순서, 오차 벡터에 영향을 주며 동일 실험의 TRAIN/validation이나 같은 λ의 teacher 라벨을 바꾸지 않는다. 배포 체크포인트는 validation이 선택한 앞선 epoch일 수 있다.

**`--development-study`는 비교할 데이터 구성을 연결하는 옵션**이다. 560→1,120 확대에서 TRAIN 포함 관계와 동일 validation을 확인한다. 특징 비교에서도 같은 준비된 development 기준을 지정해야 한다. Seed만 같거나 새 출력 폴더를 만드는 것만으로는 주변 실험의 중복 제외 이력 때문에 데이터가 같다고 보장할 수 없다. Final/OOD를 이미 연 실험을 개발 기준으로 재사용하지 않는다. 자세한 조건은 [feature9 문서](docs/bamg_feature9.md)를 따른다.

`run`이 전체 validation을 마치면 scratch와 warm 각각에 대해 **n=15·31·63·127 → λ**를 선택한다. 세 seed 결과를 종합해 실패율 우선, 이후 벌점 포함 평균 warm 시간, 동률이면 작은 λ 순으로 선택한다. 한 arm의 세 seed는 같은 규칙을 공유하며 가장 잘 나온 seed 하나만 고르지 않는다. 전체 격자에 공통으로 적용하는 λ도 비교 기준으로 선택한다.

선택은 문제의 **가장 미세한 격자에서 한 번** 수행한다. 선택한 CNN을 learned hierarchy 전체에 사용하고 coarse level마다 모델을 바꾸지 않는다. 미관측 격자는 자신 이하의 가장 큰 validation 격자를 기준으로 하며 양 끝을 고정한다. 따라서 **n≥127, 특히 OOD n=255는 n=127 선택값을 사용**한다. 그 규칙의 독립적인 성능 검증은 final/OOD에서 수행한다.

## Lenovo 바탕화면에서 새 실험 시작

아래는 **Ubuntu/WSL 터미널** 명령이다. 연구 실행기는 `fcntl`을 사용하므로 Windows CMD에서 직접 실행하는 안내가 아니다. Python 요구사항은 ≥3.10이고 CI는 3.12 CPU 환경이다. 기존 학습 폴더는 유지하고 새 폴더를 만든다. Windows 사용자명/바탕화면 위치가 다르면 경로를 수정한다.

```bash
cd /mnt/c/Users/user/Desktop
git clone --branch residual-u-net-teacher --single-branch \
  https://github.com/slwjackie/adaptive-multigrid.git adaptive-multigrid-feature9-current
cd adaptive-multigrid-feature9-current
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.10' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1
set -o pipefail
mkdir -p artifacts/logs
```

기존 실행을 재개할 때는 의존성을 다시 설치하지 말고 원래 환경을 복구한다. 아래 두 입력은 **smoke에도 필요**하다. 예시 경로를 실제 **호환되는 frozen BAMG source study**와 **완료된 수치 teacher 비교** 위치로 수정한다. Clone만으로 학습된 모델이나 실험 결과가 생기지는 않는다.

```bash
export BAMG_SOURCE_STUDY="/mnt/c/Users/user/Desktop/adaptive-multigrid-bamg/artifacts/initial_weights/mac_bamg"
export BAMG_TEACHER_COMPARISON="/mnt/c/Users/user/Desktop/adaptive-multigrid-bamg/artifacts/bamg_energy_teacher_lambda_sweep_q1"
test -f "$BAMG_SOURCE_STUDY/data/research_freeze.json" && \
test -f "$BAMG_TEACHER_COMPARISON/teacher_comparison.json"
```

파일 확인이 실패하면 실행을 멈추고 경로를 바로잡는다. 입력 자체가 없다면 [BAMG source/compact 실험](docs/bamg_compact_weight_study.md)과 [수치 teacher 비교](docs/bamg_energy_teacher_comparison.md)를 먼저 완료해야 한다. Source의 모델 구조·이산화와 저장된 해시도 검증된다.

### Smoke → 560개 본 학습

Smoke는 n=7 TRAIN 2개와 validation 2개, seed 1개로 여섯 조건을 짧게 실행한다. 실행 검증용이며 성능 증거가 아니다. 아래 출력 폴더는 처음 쓰는 경로여야 한다.

```bash
python -u scripts/run_v6_7_bamg_energy_cnn_feature9.py run \
  --config configs/v6_7_bamg_energy_cnn_feature9_symmetric9_smoke_q1.json \
  --source-study "$BAMG_SOURCE_STUDY" \
  --teacher-comparison "$BAMG_TEACHER_COMPARISON" \
  --output artifacts/feature9_symmetric9_smoke \
  2>&1 | tee artifacts/logs/feature9_symmetric9_smoke.log
```

Smoke가 성공한 다음 본 학습을 실행한다. 아래는 새로운 development cohort를 만드는 예시다. 기존 routing/feature 실험과 같은 데이터로 비교할 목적이라면 **첫 실행부터** `--development-study`로 해당 기준 폴더를 지정하고 이후 재개에도 유지한다.

```bash
python -u scripts/run_v6_7_bamg_energy_cnn_feature9.py run \
  --config configs/v6_7_bamg_energy_cnn_feature9_symmetric9_q1.json \
  --source-study "$BAMG_SOURCE_STUDY" \
  --teacher-comparison "$BAMG_TEACHER_COMPARISON" \
  --output artifacts/feature9_symmetric9_560_q1 \
  2>&1 | tee artifacts/logs/feature9_symmetric9_560_q1.log
```

`row9`는 대응 config와 별도 output을 사용한다. Feature9 실행기에는 teacher label을 한 case씩 읽는 메모리 절약 방식이 포함되어 있다. 모든 메모리 사용량이 일정해지는 것은 아니며 `/mnt/c`의 추가 읽기는 느릴 수 있다. 실제 자원 사용은 `runtime_metrics.json` 등으로 확인한다.

### 중단 후 재개와 확대 비교

같은 checkout·가상환경·스레드 설정·입력 경로를 복구한 뒤 **원래 전체 `run` 명령에 `--resume`을 추가하고 `tee -a`로 변경**한다. 처음부터 지정했다면 `--development-study`도 유지한다. 저장된 optimizer·난수 상태·epoch 위치를 복구하며 마지막 저장 이후의 update는 반복할 수 있다. 새 빈 output에는 `--resume`을 쓰지 않는다.

```bash
python -u scripts/run_v6_7_bamg_energy_cnn_feature9.py run \
  --config configs/v6_7_bamg_energy_cnn_feature9_symmetric9_q1.json \
  --source-study "$BAMG_SOURCE_STUDY" \
  --teacher-comparison "$BAMG_TEACHER_COMPARISON" \
  --output artifacts/feature9_symmetric9_560_q1 \
  --resume \
  2>&1 | tee -a artifacts/logs/feature9_symmetric9_560_q1.log

python scripts/run_v6_7_bamg_energy_cnn_feature9.py report \
  --output artifacts/feature9_symmetric9_560_q1
```

`report`는 개발 평가가 끝나 저장된 결과를 읽는다. 1,120개 비교는 560개 개발 실험을 완료한 뒤 `configs/v6_7_bamg_energy_cnn_feature9_symmetric9_1120_q1.json`, 새 output, `--development-study artifacts/feature9_symmetric9_560_q1`을 사용한다. 확대 scratch도 다시 초기화하며 560개 모델을 이어 학습하지 않는다. 두 실험의 final/OOD는 열지 않은 상태에서 데이터 규모·특징·학습 예산을 결정한다.

## Freeze와 final/OOD

| 명령 | 범위 |
| --- | --- |
| `run` / `run --resume` | 라벨, 학습, 전체 validation, 격자별 선택까지 |
| `report` | 저장된 개발 평가 읽기 |
| `freeze` | 완료 여부·선택 규칙 재검증, 모델과 데이터·소스·환경·선택 고정 |
| `evaluate-final` | Final과 모든 OOD 평가 |
| `report-final` | 저장된 held-out 결과 읽기, `--group`으로 그룹 지정 가능 |

**`run --resume`이 final/OOD까지 자동 실행하지 않는다.** 최종 평가할 실험을 개발 결과로 정한 다음 같은 실행기를 사용한다. 아래는 560개 실험을 선택한 경우다.

```bash
python scripts/run_v6_7_bamg_energy_cnn_feature9.py freeze \
  --output artifacts/feature9_symmetric9_560_q1 && \
python -u scripts/run_v6_7_bamg_energy_cnn_feature9.py evaluate-final \
  --output artifacts/feature9_symmetric9_560_q1 \
  2>&1 | tee artifacts/logs/feature9_symmetric9_560_q1_final.log
```

Final 첫 실행에는 `--resume`을 붙이지 않는다. Final이 시작된 뒤 중단되었다면 같은 output의 `evaluate-final`에 `--resume`을 추가하고 `tee -a`를 사용한다. `--max-cases 10`으로 새로 완료할 연산자 수를 제한할 수도 있다. Freeze 이후 해당 output의 학습·선택은 닫힌다.

```bash
python scripts/run_v6_7_bamg_energy_cnn_feature9.py report-final \
  --output artifacts/feature9_symmetric9_560_q1 --group final
python scripts/run_v6_7_bamg_energy_cnn_feature9.py report-final \
  --output artifacts/feature9_symmetric9_560_q1 --group ood_grid
```

연구 holdout은 **총 1,270개 독립 연산자**다: within-support final 560, grid OOD 140, rotation 150, anisotropy 180, contrast 120, topology 120. 각 연산자에 manufactured/random/localized RHS와 3회 반복 측정을 사용한다. RHS·seed·반복 수는 독립 문제 수가 아니다.

`fixed_C`, `em5_C`, 수치 `bootstrap_C`, 기존 CNN, 각 λ의 지도학습/미세조정 CNN, routing CNN을 비교한다. 실패는 벌점과 함께 남기고 final/OOD로 λ·모델·seed를 다시 선택하지 않는다. 전체 분포는 [holdout 설계](docs/bamg_energy_cnn_initialization_comparison.md#expanded-finalood-protocol)를 참고한다.

## 결과 파일·재현성·검증

| 출력 폴더 내 경로 | 내용 |
| --- | --- |
| `configuration.json`, `study_manifest.json`, `data/` | 적용 설정·출처·TRAIN/validation |
| `labels/lambda_*/teacher/train/`, `teacher_bank_manifest.json` | λ별 라벨과 공통 vector bank 검증 |
| `runs/lambda_10_scratch_seed6101/` 등 | 재개 상태·학습 이력·선택된 모델 |
| `runtime_metrics.json` | 단계별 시간·프로세스 최대 RSS |
| `comparison.json`, `validation/`, `validation_routed/` | 개발 평가 |
| `routing_selection.json` | 격자별 λ와 전역 고정 λ |
| `holdout_plan.json`, `research_freeze.json` | 사전 holdout 계획과 확정된 실험 |
| `final/report.json`, `final/claim.json` | Final/OOD 결과와 재개 상태 |

Source freeze/수치 비교/학습의 저장 해시를 수동으로 바꾸지 않는다. 입력 digest 오류는 [호환성 안내](docs/bamg_source_freeze_compatibility.md), 기존 10채널 실행의 메모리 부족은 [저메모리 재개 안내](docs/bamg_cnn_low_memory_recovery.md)를 따른다. 다른 특징 모드나 변경한 데이터·예산을 기존 output에 덮어써서 재개하지 않는다. 실행 중인 checkout·환경은 유지한다.

```bash
python -m pytest tests/test_bamg_feature9.py tests/test_bamg_feature9_multilevel.py tests/test_bamg_feature9_study.py
python -m pytest -ra
sha256sum -c THREE_PILLARS_SOURCE.sha256
sha256sum -c P_HEADROOM_SOURCE.sha256
sha256sum -c WORLD_MODEL_SOURCE.sha256
sha256sum -c HS_WORLD_SOURCE.sha256
```

CI는 전체 테스트와 위 네 소스 체크섬을 검사한다. Optional native stencil 빌드도 검사하지만 현재 CPU/CSR 실험의 필수 실행 조건은 아니다. Smoke·테스트 통과는 실행 검증이며 연구 규모의 수렴·속도 우위를 보장하지 않는다. Warm solve 성능은 setup을 포함한 총시간 성능과 다르고, 정렬 합성 확산 문제 결과가 비정렬 격자·비선형 PDE·실제 CFD의 일반화를 증명하지 않는다.

과거 실험·수치·패키지 manifest는 각 시점의 기록이다. [문서 목록](docs/README.md)에서 EM-line, 경량 smoother, P-only, U-net 등으로 이동할 수 있다. 별도 비정렬 neural-AMG 실행기는 이 브랜치 스냅샷의 진입점에 포함되지 않는다.
