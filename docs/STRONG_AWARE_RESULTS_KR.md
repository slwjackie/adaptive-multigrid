# Strong-aware smoke_01: 확인된 결과와 해석 범위

이 문서는 `artifacts/my_v6_7_strong_aware/smoke_01`의 완료된 development 기록을
읽어 작성했습니다. `progress.json`은 `development_complete`이고 final test는
사용하지 않았습니다. 이 기록은 첫 소규모 smoke의 결과이며, 설치 후 추가 hardware
측정이나 후속 연구 결과가 나오더라도 이 원본의 숫자를 소급 변경하지 않습니다.

설치 후 failure cache와 학습 feasibility 검사를 보완한 뒤, 기존 checkpoint에서 이어서
검증했습니다. [최신 결과](STRONG_AWARE_COMPLETION_KR.md)는 별도 기록이며 아래 원본 수치를 대체하지 않습니다.

## 실험이 실제로 수행한 범위

- Train: 새 n=15 operator 일곱 개, family마다 하나.
- Validation: 새 n=15/31 operator 14개. Architecture 선택에 사용한 development 자료.
- Cold 비교: 각 operator/arm당 세 번 측정, warmup 한 번.
- Numerical solve: CPU FP64, relative tolerance `1e-8`, absolute tolerance `1e-12`,
  최대 150 attempts/cycles, classical pre/post 각각 두 번, learned levels 두 개.
- 학습: 각 후보 일곱 optimizer updates, learned full V-cycle 두 번과 classical tail
  한 번을 포함하는 objective. 일곱 train operator를 동일한 순서로 한 번씩 사용.
- 기존 strong selector/rules는 동결했고, final/OOD를 생성·평가하지 않음.

Ordinary, compact, D4, O2, CNN-P, GNN-P, edge-P, smoother teacher, P teacher,
joint candidate의 **열 모델 모두 7/7 updates를 수행했고 skip은 0**입니다. 이는
forward/backward·sparse projection·checkpoint 저장이 실제로 실행됐다는 증거입니다.
충분히 수렴하도록 학습했다는 증거는 아닙니다. 특히 큰 teacher도 작은 student와
똑같이 일곱 번만 갱신했으므로, 여기서 얻은 teacher 결과를 capacity upper bound로
해석할 수 없습니다.

근거: [학습·단계 상태](../artifacts/my_v6_7_strong_aware/smoke_01/progress.json),
[development 보고서](../artifacts/my_v6_7_strong_aware/smoke_01/development_report.json).

## 첫 cold branch 비교

아래 속도비는 `T_strong_C / T_candidate`이며 1보다 커야 빠릅니다. 실패 solve를
속도비에 넣지 않았고, 이번 표의 paired speedup은 공통 성공 operator 12개로 계산했습니다.

| Method | 전체 성공 | Strong C 대비 속도비 | 95% bootstrap CI |
| --- | ---: | ---: | --- |
| Fixed Classical | 14/14 | 0.373× | [0.239, 0.583] |
| Frozen Strong Classical | 12/14 | 1.000× | [1.000, 1.000] |
| Selected H_S | 12/14 | 0.750× | [0.677, 0.830] |
| Selected H_P | 12/14 | 0.719× | [0.644, 0.797] |
| Selected H_SP | 12/14 | 0.675× | [0.559, 0.792] |
| Research adaptive | 12/14 | 0.732× | [0.641, 0.815] |

이번 adaptive는 실제 accepted neural work가 없었습니다. 따라서 이 행은 neural
가속 결과가 아닙니다. 강제 H branch들 역시 cold total time에서 strong C보다
빨라지지 않았습니다. 연구용 후보가 선택됐다는 것과 deployment 개선이 채택됐다는
것을 구분해야 합니다.

고정된 strong selector 자체도 모든 문제를 해결하지 못했습니다. n=31 channel은
150회 뒤 relative residual 약 `3.55e-7`, variable coefficient는 약 `3.34e-6`으로
목표 tolerance에 도달하지 못했습니다. Fixed C는 두 문제를 포함해 14개를 모두
해결했습니다. 그러므로 'strong 대비 새 failure가 없다'는 결과를 '전체 성공'으로
읽으면 안 됩니다. 공통 성공 12개의 시간만으로 전체 14개에서 strong C가 우월하다고
말할 수도 없습니다. 이번 요청에 따라 selector를 바꾸지 않았으며, 이 관찰을 보고
validation 문제에 맞춰 selector를 다시 튜닝하지 않았습니다.

H_S는 성공한 세 문제에서 manufactured L2/energy error 비교의 `5% + 1e-12` 기준을
넘어서 strong C보다 나빴고, H_SP는 두 문제에서 그랬습니다. 같은 residual tolerance를
만족한다는 사실은 같은 solution error를 뜻하지 않습니다. H_P/adaptive의 해당
악화 사례가 없다는 사실도, neural work나 속도 개선이 없을 때 독립적인 neural
정확도 우위의 증거가 되지는 않습니다.

근거: [branch 원시 결과](../artifacts/my_v6_7_strong_aware/smoke_01/validation/branches/raw_results.json),
[branch 요약](../artifacts/my_v6_7_strong_aware/smoke_01/validation/branches/comparison.json).

## Smoother architecture 비교

| Smoother | 실제 smoother parameters | 성공 | 해당 ablation의 Strong C 대비 cold 속도비 |
| --- | ---: | ---: | ---: |
| Ordinary residual5 | 10,906 | 12/14 | 0.748× |
| Compact3 | 6,266 | 12/14 | 0.778× |
| D4 | 10,906 | 12/14 | 0.602× |
| O2, width 18 | 10,093 | 12/14 | 0.575× |

Ordinary/D4는 parameter 수가 같고 O2는 약 7.5% 적습니다. Compact는 더 작은
deployment 후보이며 parameter 수가 같다고 표시하지 않습니다. 모든 모델은 같은
열 feature, 같은 one-basis 9-point output, direction/gain 분리와 numerical kernel을
사용했습니다. 작은 표본에서 compact가 선택됐지만, 이 결정의 상태는
`production_accepted=false`입니다.

D4의 exact 90-degree/reflection 관계는 여덟 view 평균으로 얻습니다. View를
batching하더라도 encoder 연산량은 여덟 배이고 그 비용이 setup에 포함됩니다. O2의
continuous tensor/kernel equivariance 역시 square-grid generation이 싸다는 뜻은
아닙니다. 이 smoke의 첫 validation operator에서 CPU generation median은 ordinary
약 0.644 ms, compact 0.497 ms, D4 3.854 ms, O2 3.227 ms였습니다. 이 숫자는 한
operator의 실제 learned-level generation 측정이며 전체 PDE 집합의 평균이나 solve
시간이 아닙니다. Generation을 포함한 bank setup은 별도로 기록되어 있습니다.

일곱 번의 update, n=15 단일 training 크기와 n≤31 validation으로 O2의 rotation
generalization 잠재력이나 충분히 학습한 큰 teacher 성능을 결론 낼 수 없습니다.
이번 결과는 '현재 smoke checkpoint에는 추가 generation 비용을 상쇄할 cold solve
개선이 나타나지 않았다'는 범위에서 읽어야 합니다.

근거: [smoother ablation](../artifacts/my_v6_7_strong_aware/smoke_01/validation/smoother_architectures/comparison.json),
[선택 기록](../artifacts/my_v6_7_strong_aware/smoke_01/smoother_selection.json),
[generation 원시 기록](../artifacts/my_v6_7_strong_aware/smoke_01/generation/generation.json).

## Graph-aware P가 아직 개선으로 채택되지 않은 이유

| P model | 실제 transfer parameters | 성공 | 해당 ablation의 Strong C 대비 cold 속도비 |
| --- | ---: | ---: | ---: |
| 기존 CNN-P architecture control | 11,008 | 12/14 | 0.719× |
| Small GNN | 4,961 | 12/14 | 0.580× |
| Edge-MLP | 1,409 | 12/14 | 0.640× |

이번에는 CNN-P control이 후속 비교 후보로 선택됐습니다. GNN/edge 모델이 작다는
사실만으로 sparse preprocessing, generation, Galerkin fill 비용이 작아지는 것은
아닙니다. 실제 H_P의 accepted neural work는 rotated anisotropy 두 문제에서만
확인됐습니다.

첫 near-isotropic validation operator의 generation 검사에서는 GNN, edge-MLP,
P teacher가 모두 `max_ac_ratio` cap으로 거부됐습니다. 이는 실패 setup으로 기록됐고
성공 generation median으로 집계하지 않았습니다. Solve 경로는 classical fallback으로
이어질 수 있지만, 그 결과를 정상 learned P 적용 결과라고 해석하면 안 됩니다.
학습은 실행됐어도 실제 배포 가능한 sparse hierarchy로 받아들여졌는지는 별개의 문제입니다.

이 관찰은 graph architecture의 일반적인 무효를 보여주지 않습니다. 짧게 학습한
weight, support/projection, 실제 coarse fill 제한이 함께 영향을 준 결과입니다.
현재 cap을 validation 결과에 맞춰 느슨하게 하지 않았습니다. 더 긴 연구에서는
동일한 사전 고정 cap 아래 실제 neural 적용·fill·setup·total time이 함께 개선되는지
판단해야 합니다.

근거: [P ablation](../artifacts/my_v6_7_strong_aware/smoke_01/validation/p_architectures/comparison.json),
[generation 실패 비용](../artifacts/my_v6_7_strong_aware/smoke_01/generation/generation.json).

## Teacher, KD, joint 결정

Smoother teacher의 direct compact 대비 contraction geometric improvement는
약 `1.000026×`, P teacher의 direct GNN 대비 값은 약 `1.0000004×`였습니다.
사전 기준 `1.052632×`에 도달하지 못했습니다. P teacher는 비교 가능한 모든 문제에서
accepted neural work를 보인 조건도 만족하지 못했습니다.

따라서 **이 smoke에서는 두 teacher 모두 거부했고, distilled checkpoints를 만들지
않았습니다.** 'KD를 실행했지만 효과가 없었다'가 아니라 'teacher 우위가 없어 KD
실행 조건을 통과하지 못했다'가 정확한 설명입니다. KD 코드의 gradient/teacher detach와
checkpoint 독립성은 별도 테스트 대상이며, 성능 실험의 positive KD 결과와는 다릅니다.

Joint candidate는 independent pair 대비 H_S 보존 조건을 통과했으나, H_SP total-time
개선은 약 `0.9972×`로 기준 `1.03093×`를 충족하지 못해 거부했습니다. 더 복잡한
teacher/KD/joint를 무조건 넣지 않는 선택 절차가 실제로 작동한 결과입니다.

근거: [teacher 결정](../artifacts/my_v6_7_strong_aware/smoke_01/teacher_decisions.json),
[KD 결정](../artifacts/my_v6_7_strong_aware/smoke_01/distillation_decisions.json),
[joint 결정](../artifacts/my_v6_7_strong_aware/smoke_01/joint_decision.json).

## 회전 진단과 hardware 기록을 읽을 때의 한계

Arbitrary-angle resampling diagnostic은 fixed-grid feature/stencil의 moment-channel
action을 사용하며, 실제 회전 PDE를 새로 assemble한 OOD solve가 아닙니다. 첫 문제에서
ordinary/D4의 crop에는 다섯 pixel, O2의 crop에는 33~37 pixel이 남았습니다. 모델의
receptive radius가 달라 crop도 달랐으므로 raw error 숫자로 공정한 architecture
순위를 만들지 않습니다. 모델들은 일곱 update밖에 하지 않았고 초기 중앙 stencil에
가깝습니다. 이런 상태에서 작은 rotation error가 일반화 능력의 증거는 아닙니다.

정확한 D4 관계와 임의각도 tensor algebra는 비자명한 head를 사용한 수치 테스트로
별도 확인합니다. 실제 continuous rotation OOD 성능은 아직 untouched인 별도 PDE
평가에서 판단해야 합니다.

Smoke_01 generation 기록에서는 해당 실행 process의 MPS/CUDA availability가 false여서
CPU만 측정했습니다. 이는 **Mac hardware에 MPS가 없다는 주장과 다릅니다.** Sandbox 밖
실제 hardware 측정이 추가되는 경우 별도 결과 경로와 실제 device명을 명시해야 하며,
smoke_01의 `not_measured` 기록을 측정값으로 덮어쓰지 않습니다. A100 성능은 이
smoke에서 측정하지 않았습니다.

근거: [회전 진단](../artifacts/my_v6_7_strong_aware/smoke_01/validation_rotation_resampling.json),
[device availability 기록](../artifacts/my_v6_7_strong_aware/smoke_01/generation/generation_manifest.json).

이 smoke로 확인한 것은 연구 경로와 보수적인 거부 절차의 작동, 그리고 현재 짧게
학습한 후보들이 strong baseline보다 cold time에서 빠르지 않다는 사실입니다.
수백 operator의 충분한 학습과 고정 후 final/OOD 평가가 남아 있으며, 이번 validation
결과를 독립 final certificate로 사용하지 않습니다.
