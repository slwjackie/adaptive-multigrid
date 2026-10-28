# 같은 깊이의 기본 CNN 비교

`plain5`는 기존 `residual5`에서 특징맵의 두 skip 덧셈을 제거한 기본 CNN이다.
기존 설정의 기본값은 계속 `residual5`이며, 새 설정으로 선택해야 `plain5`를 사용한다.

| 항목 | `residual5` | `plain5` |
| --- | --- | --- |
| Body의 공간 convolution | 3×3 Conv 5개 | 동일 |
| 중간 채널 수 | 설정의 `hidden` | 동일 |
| GELU | 4개 | 동일 |
| 마지막 출력 head | 1×1 Conv | 동일 |
| 전체 convolution 수 | 6개 | 6개 |
| 최대 receptive field | 11×11 | 11×11 |
| 특징맵 skip 덧셈 | 2개 | 없음 |
| Body 초기화 | 일부 residual 경로를 identity/zero로 초기화 | Conv2d의 일반 초기화 |

`plain5` body는 `Conv → GELU`를 4번 적용하고 마지막 3×3 Conv를 적용한다.
Body 끝에 GELU를 추가하지 않는다. 같은 채널 수에서 두 body의 파라미터 수는 같다.
기존 zero-initialized 출력 head와 각 모델의 초기 수치 연산자는 유지한다. P 생성기는
bilinear P에서 시작하며, 줄 smoother의 초기값은 실제 A의 삼중대각 줄 성분에서 계산한다.
Residual 경로의 zero 초기화를 기본 CNN body에 그대로 적용하면 특징 전달과 초기
gradient가 차단되므로, `plain5`의 body는 일반 초기화를 사용한다.

제거하는 것은 **CNN 내부의 skip 덧셈**이다. P의 bilinear 기준값에 affine 보정을
더하는 출력 파라미터화, 희소 support·행합·injection 제약, Galerkin 구성, 학습 loss는
그대로다. Conv 수를 줄이는 실험이 아니므로 더 빠른 추론이나 수렴을 보장하지 않는다.

## EM-line 경로에서 선택

`scripts/run_v6_7_em_line_study.py`의 설정 JSON 최상위에서 다음 두 필드를 사용할 수 있다.
생략하면 각각 `residual5`이며, 허용값은 `residual5`와 `plain5`다.

```json
{
  "transfer_body_kind": "plain5",
  "smoother_body_kind": "residual5"
}
```

| 비교 | `transfer_body_kind` | `smoother_body_kind` |
| --- | --- | --- |
| 기존 기준 | `residual5` | `residual5` |
| P CNN만 기본 CNN | `plain5` | `residual5` |
| Smoother CNN만 기본 CNN | `residual5` | `plain5` |
| 두 생성기 모두 기본 CNN | `plain5` | `plain5` |

추가된 `configs/v6_7_em_line_plain5_smoke.json`과
`configs/v6_7_em_line_plain5_research.json`은 각각 기존 smoke/research 설정에
`"transfer_body_kind": "plain5"`만 추가한 **P CNN 비교용** 설정이다.
데이터·seed·채널 수·학습 예산·수치 알고리즘 설정은 대응하는 기존 JSON과 같다.
Q1 FEM 비교는 복사한 설정의 최상위에 `"discretization": "q1"`을 추가한다.
Residual/기본 CNN 비교에서 이산화·실제 operator 분할·seed·학습 예산·스레드 수를 맞춘다.

다음 명령은 저장소 루트에서 실행한다. `prepare` 전에 run 폴더 안에 파일을 만들지 않는다.

```bash
PLAIN_CNN_RUN="artifacts/em_line_plain5_smoke_$(date +%Y%m%d_%H%M%S)"
python scripts/run_v6_7_em_line_study.py prepare \
  --run-dir "$PLAIN_CNN_RUN" --config configs/v6_7_em_line_plain5_smoke.json
python -u scripts/run_v6_7_em_line_study.py train --run-dir "$PLAIN_CNN_RUN"
python -u scripts/run_v6_7_em_line_study.py benchmark --run-dir "$PLAIN_CNN_RUN" \
  --repeats 1 --warmups 0 --rhs-counts 1 --regimes warm multiple
```

본 학습에서는 새 run 경로와 `configs/v6_7_em_line_plain5_research.json`을 사용한다.
반복 측정 예시는 `--repeats 3 --warmups 1 --rhs-counts 1 4 16 64 --regimes cold warm multiple`이다.
`train`은 기존처럼 `em_P → frozen_P_line → joint_line`을 모두 수행하며 `--stage` 옵션은 없다.
P만 바꾼 모델의 성능은 benchmark의 `em_P` arm에서 확인한다. `fixed_C`는 bilinear
classical이고, EM5 classical 비교군은 별도의 `em5_C`다. 전체 실행·로그·재개 방법은
[EM-line runbook](em_line_study.md)을 따른다.

Checkpoint에는 `body_kind`가 저장되어 로드할 때 같은 구조를 복원한다. 기존 residual
checkpoint는 계속 읽을 수 있지만, 그 가중치를 이름만 바꿔 plain 모델에 로드하지 않는다.
기본 CNN은 새로 학습한다. 설정이나 소스를 변경한 실험은 **새 run 디렉터리**를 사용한다.
기존 run의 manifest를 수정하거나 다른 구조로 `--resume`하지 않는다.

## 독립 U-net teacher/student 경로

`scripts/run_v6_7_unet_transfer_study.py`를 사용하는 이전 독립 실험에서는 설정의
`unet_transfer` 안에 `"student_body_kind": "plain5"`를 추가한다. Direct와 distilled
student가 모두 동일한 기본 CNN을 사용하고, Residual U-net teacher는 그대로 유지된다.
이 필드도 생략하면 `residual5`다. 자세한 단계별 명령은
[U-net teacher/student 문서](RESIDUAL_UNET_TRANSFER_KR.md)를 따른다.

`direct_pruning` 경로는 기존 `residual5` 전용이므로 plain5 checkpoint를 입력하지 않는다.
