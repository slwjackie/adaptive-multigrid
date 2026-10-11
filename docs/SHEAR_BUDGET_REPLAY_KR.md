# 기존 smoother 체크포인트로 600회 재평가

초기 64개 압력계씩을 조사한 사용자 WSL 진단에서 Classical MG는 150회
상한으로 22/35/27개 실패했고, 600회 상한에서는 모두 수렴했다. 실제 최대
횟수는 262/275/268회였다. 이는 해당 초기 구간의 반복 예산 부족을 보여주며,
NN 600회 성공이나 전체 CFD 가속을 입증하지는 않는다.

다음 실행은 기존 `study`를 검증한 후 **같은 데이터와 같은 학습된 smoother**로
`mg.max_cycles`만 메모리에서 변경한다. 기존 허용오차, 초기값, classical P,
MG 전략, NN 가중치 및 fallback 설정을 유지한다. 원본 설정·학습 증거·결과를
수정하거나 새 설정으로 학습한 것처럼 기록하지 않는다. 새 실행기의 SHA와
원본/유효 설정은 별도 `manifest.json`에 남는다. 기존 source/config/checkpoint
검증은 그대로 적용된다. 검증 오류가 나면 원본 manifest를 고쳐서 통과시키지 않는다.

## 실행

```bash
cd /home/jackie/adaptive-multigrid-h2
git switch research/modulated-shear-temporal
git pull --ff-only origin research/modulated-shear-temporal
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -u scripts/run_v6_7_shear_budget_replay.py \
  --run-dir artifacts/shear_probe_v1/study \
  --output artifacts/shear_budget600_v1 \
  --max-cycles 600
```

OpenFOAM 재빌드·CFD 재실행·smoother 재학습은 필요 없다. 기록된 600회는
상한이며 수렴 즉시 종료한다. 기존 output은 덮어쓰지 않으므로 재실행 시에는
새 output 이름을 사용한다. `--stop-after comparison`은 비용 비교까지만 수행한다.

## 실행 순서와 비용 범위

1. 원본 source/config/data/checkpoint와 실제 학습 이력을 검증한다.
2. **전체 train/tune 시퀀스**에서 Classical, NN 매번 갱신, NN 재사용을 비교한다.
   첫 두 압력계로 각 정책을 warmup한 뒤, 매 반복마다 새 solver로 시작한다.
   반복 수는 기존 `baseline_repeats`(현재 3), 정책 측정 순서는 고정 seed로
   무작위화한다. 모든 정책이 같은 A/b/x0, 허용오차, 고정 P를 쓰는지 확인한다.
3. 설정·풀이·잔차 검증·복구·정책 결정·기타 처리·solver 생성 비용을 포함한다.
   warmup, 파일 IO, offline 데이터 검증, 결과 기록은 측정에서 제외한다.
   `solve_seconds`와 `recovery_seconds`는 중복되지 않는다. `other_seconds`에는
   특징 계산과 기타 실행 관리 비용이 포함된다. 각 phase의 중앙값 합은 전체
   시간 중앙값과 다를 수 있어 반복별 원본 비용도 함께 보존한다.
4. 어느 정책에서든 실패가 남으면 `comparison_failed`로 종료하고 probe를
   건너뛴다. 실패한 풀이와 소요 시간은 결과에서 삭제하지 않는다.
5. 모두 성공하면 원래 horizon/repeats/seed로 갱신·재사용 probe를 다시 수집한다.
   25개 압력계마다 진행 상황을 표시한다. 각 사례의 원본 측정 파일을 저장한다.
6. 기존 모든 구간 집계와 비교할 수 있게 기존 방식의 승리 수를 남긴다.
   다만 새 후보 판정은 **완전한 horizon의 서로 겹치지 않는 승리 구간**이
   양쪽에 각각 3개 이상 있고, 모든 측정 rollout에 실패가 없을 때만 통과한다.
   겹치지 않는다는 사실이 시간적 상관관계까지 없앤다는 뜻은 아니다.

평가 대상은 train/tune 압력계 재생이다. 원본 무결성 검사는 다른 split의
기록 바이트도 해시하지만 validation/test 압력계를 풀거나 튜닝에 쓰지 않는다.
전체 CFD/IPC 비용과 물리장 일치 검증은 이 실행의 범위 밖이다.
World Model이나 수소 모델은 학습·실행하지 않으며, World Model 이득 여부는
계속 `false`로 남는다. 새 결과는 기존 study의 freeze/evaluate 결과로 사용할 수
없는 별도 개발 진단이며, 이후 정책 학습·검증에는 해당 설정을 명시한 연구가 필요하다.

## 결과

- `summary.json`: 단계, 정책별 실패 수·최대 반복 횟수·비용·probe 요약.
- `comparison.json`: 사례별 반복 측정과 설정/풀이/복구 등 비용 분해.
- `comparison/`: 동일 압력계에 대한 정책별 모든 반복의 수치 결과.
- `probe/collection.json`, `probe/episodes/`: 새 probe 판정과 측정 원본.
- `manifest.json`: 동일 체크포인트, 원본 계약, 변경한 반복 상한과 하드웨어.
- `budget_diagnostics.zip`: 위 JSON을 모은 전달용 압축 파일. 행렬·체크포인트는 제외.

`replay_speedup_vs_classical > 1`이면 기록된 train/tune 압력계에서 NN이 빠른
것이다. 사례별 전체 시퀀스 시간의 중앙값들을 더한 뒤 그 비율을 계산한다.
이는 전체 CFD 가속률이나 미사용 test 사례의 성능, World Model의 이득을
나타내지 않는다. 실패한 정책에는 speedup을 부여하지 않는다.
