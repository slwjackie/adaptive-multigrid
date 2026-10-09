# 고정 P + 수소 연소 H_S 실험

이 경로는 **고정 격자의 실제 2D H2–air CFD에서 Classical MG와 학습한 H_S를 비교**한다.
World Model은 사용하지 않는다. 기존 실행 경로·설정·체크포인트는 변경하지 않았다.
새 진입점은 `scripts/run_v6_7_h2_fixed_p.py`, 설정은
`configs/v6_7_h2_fixed_p.json`이다.

기존 frozen study는 전체 source hash를 검사하므로 새 모듈이 추가된 브랜치에서
재개가 거부될 수 있다. 기존 결과의 재현에는 그 결과를 만든 원래 commit을 사용한다.

## 구현 및 검증 범위

- `P`는 prolongation, 즉 coarse correction을 fine grid로 올리는 보간 연산자다.
  각 물리적 trajectory의 첫 압력 행렬에서 classical P를 한 번 만든다.
  이후 동일 trajectory에서는 모든 레벨의 P를 유지하고,
  현재 행렬에 맞춰 `Ac = P.T @ A @ P`와 수치 분해만 갱신한다.
  독립된 물리적 case를 시작하면 새 hierarchy를 만든다.
- 비교하는 두 arm은 **동일한 classical P + classical smoother**와
  **동일한 classical P + 학습한 H_S**다. 초기 조건이 같은 독립 CFD 재실행이며,
  매 압력 solve에서 H_S를 시도한다. 학습한 선택기나 World Model은 없다.
- 신경망 갱신이 residual을 악화시키면 직전 해로 복귀해 같은 P의 classical
  cycle로 남은 공통 budget을 쓴다. 실패한 neural cycle도 시간과 budget에 포함한다.
  classical P를 다시 만드는 fallback은 없다.
- 학습 입력은 OpenFOAM이 경계조건을 반영한 실제 `A, b, x0`다.
  native `Amul` witness 두 개, 셀 순서, 경계·mesh·chemistry provenance를 확인한다.
  학습은 실제 residual의 여러 cycle 전개 손실과 안정성/no-harm 항으로
  H_S 파라미터만 갱신한다. 학습한 P나 가짜 수소 행렬 생성기는 없다.
- Python 수치/학습/프로토콜/평가 테스트와 독립 C++ wire decoder 테스트를 제공한다.
  개발 환경에는 OpenFOAM이 없어 **Foundation 13 plugin 컴파일,
  `chemkinToFoam`, 실제 `foamRun`은 검증되지 않았다**.
  AF_UNIX 생성도 이 개발 컨테이너에서 차단되어 native socket smoke test는
  대상 Linux 환경에서 실행해야 한다. 실제 화염의 가속률·물리적 타당성은 아직 측정하지 않았다.

## 대상 물리 모델

`integrations/openfoam13/case/generate.py`는 고정 Cartesian mesh의 2D
counterflow H2/air 반응 유동 입력을 만든다. OpenFOAM Foundation **13**의
`multicomponentFluid`가 운동량·종·에너지·압력을 함께 계산한다.
Cantera `h2o2.yaml`의 10종/29반응을 CHEMKIN으로 내보내고 OpenFOAM으로 변환한다.
국소 2D 고온 점화 영역을 사용하며 단순한 1D 해의 복제는 아니다.

첫 구현은 serial, static uniform x-y mesh, 한 셀 두께, 대칭 anchored SPD
압력 행렬만 지원한다. x/y 셀 수는 각각 `2**L - 1`이다.
MPI, AMR, 움직이는 mesh, coupled/cyclic 경계, pure-Neumann nullspace,
비대칭 transonic pressure 행렬은 지원하지 않는다.

수송 모델은 **Sutherland/Wilke + unityLewisFourier, Soret 없음**이다.
수소의 preferential diffusion을 정밀하게 재현하는 모델이 아니므로
화염 속도·소염·고압 반응의 검증된 물리 예측으로 해석하지 않는다.
기본 63×63 격자와 2 ms 종료 시간도 수렴된/정상 화염을 보장하지 않는다.
자세한 가정은 `integrations/openfoam13/case/README.md`를 참고한다.

## 1. 설치 및 native 준비

저장소 root에서 실행한다. OpenCFD의 v2312/v2406 등은 Foundation 13과 API가 다르다.

```bash
python -m pip install -e '.[dev]'
python -m pip install 'cantera>=3.0,<4'
python integrations/openfoam13/tests/check_wire.py
python integrations/openfoam13/tests/check_wire.py --ipc
source /opt/openfoam13/etc/bashrc
integrations/openfoam13/Allwmake
```

먼저 작은 case로 plugin/변환/실행이 실제로 동작하는지 확인하고, 본 실험에서는
공간·시간 해상도와 점화/반응 지속을 별도로 확인한다. `--native-only` preflight는
OpenFOAM 설치 확인용이며 공통 raw-L2 종료 조건의 비교 결과가 아니다.

## 2. 독립적인 train / validation / test case 생성

다음은 서로 다른 유입 온도를 사용하는 **시작용 예시**다.
물리적 case family를 분리하고, 인접 timestep을 서로 다른 split에 섞지 않는다.
단 하나의 train case만으로 넓은 조건에 대한 일반화를 주장하지 않는다.
`--case-group`은 동일하거나 관련된 물리 case를 묶는 이름이다.

```bash
python integrations/openfoam13/case/generate.py \
  --output artifacts/h2_train --case-id h2_train --case-group inlet300 \
  --split train --inlet-temperature 300 --mode native --socket-path /tmp/h2.sock
python integrations/openfoam13/case/generate.py \
  --output artifacts/h2_validation --case-id h2_validation --case-group inlet325 \
  --split validation --inlet-temperature 325 --mode native --socket-path /tmp/h2.sock
python integrations/openfoam13/case/generate.py \
  --output artifacts/h2_test --case-id h2_test --case-group inlet350 \
  --split test --inlet-temperature 350 --mode native --socket-path /tmp/h2.sock

python integrations/openfoam13/case/run_case.py --case artifacts/h2_train --prepare-only
python integrations/openfoam13/case/run_case.py --case artifacts/h2_validation --prepare-only
python integrations/openfoam13/case/run_case.py --case artifacts/h2_test --prepare-only
```

이 단계는 chemistry 변환과 mesh 검사만 한다. `contract.json`에 실제 파일 hash와
`prepared_not_executed` 상태가 남는다. 위 세 원본은 CFD 실행 전 상태로 보존한다.

## 3. 실제 CFD의 압력 시스템 수집

수집은 native PCG/DIC로 실제 CFD를 진행하면서 수행한다. PCG는 데이터 생산용이며,
최종 비교 baseline은 고정 P의 Classical MG다. 원본 case를 복사한다.

```bash
cp -a artifacts/h2_train artifacts/h2_train_recordcase
```

터미널 A에서 service를 시작한다.

```bash
python scripts/run_v6_7_h2_fixed_p.py serve \
  --config configs/v6_7_h2_fixed_p.json \
  --case-contract artifacts/h2_train/contract.json \
  --mode native --socket /tmp/h2.sock \
  --record-dir artifacts/h2_record_train \
  --evidence artifacts/h2_record_train_service.jsonl
```

터미널 B에서 실제 CFD를 실행한다. socket이 생성된 다음 시작한다.

```bash
foamRun -case artifacts/h2_train_recordcase > artifacts/h2_train_recordcase/log.foamRun 2>&1
```

service는 연결이 닫히면 종료한다. 두 프로세스가 성공했는지 확인한다.
`validation`, `test`도 위의 `train` 경로를 해당 이름으로 바꿔 **각각** 수집한다.
socket은 순차 실행 시 재사용할 수 있다. 각 recording에는 적어도 두 solve가 필요하다.
수집된 `case_contract.json`, `ldu_sequence.json`, raw LDU 파일을 함께 보존한다.
모든 압력 solve를 기록하므로 trajectory 길이에 따라 저장량과 후속 replay 비용이 커진다.

## 4. import → H_S 학습 → 동일 시스템 replay

물리 허용 오차는 학습/평가 시작 전에 config에서 정한다. 제공된 값은 시작값이며,
정확도 연구를 대신하지 않는다. `prepare`가 설정과 source/data hash를 고정한다.

```bash
python scripts/run_v6_7_h2_fixed_p.py prepare \
  --run-dir artifacts/h2_study \
  --config configs/v6_7_h2_fixed_p.json \
  --input-ldu artifacts/h2_record_train artifacts/h2_record_validation artifacts/h2_record_test
python scripts/run_v6_7_h2_fixed_p.py train --run-dir artifacts/h2_study
python scripts/run_v6_7_h2_fixed_p.py evaluate \
  --run-dir artifacts/h2_study --split validation --repeats 3
```

이 replay는 두 풀이기에 정확히 같은 `A, b, x0` sequence를 제공한다.
양쪽 arm/repeat에 독립적인 solver를 만들고 처음 만든 P의 hash가 끝까지 같으며
arm 사이에서도 같은지 확인한다. setup·갱신·residual 검사·fallback 비용을 포함한다.
실패를 제외해 평균내지 않으며 실제 neural 호출이 없으면 H_S 가속 주장을 차단한다.
**Replay 시간은 CFD 전체 시간이나 물리량 검증이 아니다.**

## 5. 실제 CFD를 Classical / H_S로 각각 재실행

`--case`에는 2단계의 준비된 미실행 원본을 전달한다.
실행기는 arm/repeat별 새 복사본과 새 service를 만들며 원본은 변경하지 않는다.

```bash
python scripts/run_v6_7_h2_fixed_p.py coupled \
  --run-dir artifacts/h2_study --case artifacts/h2_validation \
  --output artifacts/h2_coupled_validation --repeats 3 --timeout 3600
python scripts/run_v6_7_h2_fixed_p.py freeze --run-dir artifacts/h2_study
python scripts/run_v6_7_h2_fixed_p.py evaluate \
  --run-dir artifacts/h2_study --split test --repeats 3
python scripts/run_v6_7_h2_fixed_p.py coupled \
  --run-dir artifacts/h2_study --case artifacts/h2_test \
  --output artifacts/h2_coupled_test --repeats 3 --timeout 3600
```

test는 freeze 이후에만 평가한다. 재학습/설정 변경/기존 결과 덮어쓰기로 test에
맞추는 대신 새 study와 별도 held-out test를 사용한다. 실패한 CFD도 로그와 함께
남기며 유효하지 않은 pair의 speedup은 숫자로 보고하지 않는다.

확인할 값은 다음과 같다.

| 관측값 | 의미 |
|---|---|
| native pressure seconds | 모든 압력 solve의 plugin 진입부터 최종 native residual 검사까지; JSON/IPC/전처리/갱신/추론/fallback 포함 |
| CFD wall seconds | 실제 CFD 실행과 실행기에서 명시한 service 시작 비용; mesh/chemistry 사전 준비와 offline 학습은 별도 |
| residual / threshold | 양쪽 모두 `||b-Ax||₂ <= max(atol, rtol*||b-Ax0||₂)` |
| fixed P hash | 각 trajectory의 P 불변, paired arm의 첫 P 동일 |
| T, H2, U, p | 같은 물리 시각의 저장된 실제 field 배열을 사전 선언 atol/rtol로 비교 |
| continuity / Qdot | 실제 로그의 연속방정식 오차와 출력 열방출을 확인; 지속 화염/정확도의 독립 검증은 여전히 필요 |
| failures / neural calls | 성공 표본만 골라 보고하거나 classical fallback만으로 H_S 성능을 주장하지 않음 |

계산 과정이 조금 달라지면 뒤 timestep의 압력 행렬도 달라질 수 있다.
동일 입력 replay와 독립 full-CFD 재실행을 모두 제공하는 이유다.
압력 부분의 가속률을 전체 CFD 가속률로 대체하지 않는다.

## 이후 World Model을 붙일 조건

validation replay의 `diagnostic`은 시간·초기 residual별 H_S의 손익을 기록한다.
반복 측정에서 Classical/H_S의 유리한 구간이 바뀌면 별도의 selector 연구를
검토할 수 있다. 현재 `world_model_candidate`는 기술적인 관찰일 뿐,
residual 상태와 효율 사이의 인과성이나 World Model의 이득을 입증하지 않는다.
World Model 학습/활성화와 P 재구성 선택은 이 구현에 없다.

## 테스트

```bash
python scripts/build_native_stencil.py --no-openmp
python -m pytest -ra
python -m pytest integrations/openfoam13/case/test_generation.py -q
python integrations/openfoam13/tests/check_wire.py
sha256sum -c H2_FIXED_P_SOURCE.sha256
```

테스트의 작은 SPD fixture는 순서·residual·학습 gradient·오류 처리 검증용이다.
실제 수소 연소 데이터나 성능 측정으로 보고하지 않는다.
