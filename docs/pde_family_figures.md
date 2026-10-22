# 7개 PDE family의 수치해·확산 flux figure

`scripts/run_v6_7_pde_figures.py`는 EM-line·경량 smoother 실험에서 사용한
7개 정상 확산 PDE의 계수장과 수치해를 그린다. 보고서에서 **“7개 확산 PDE
family의 유한요소 수치해 및 확산 flux”**로 설명할 수 있다. CFD에서 사용하는
contour·벡터 그림 형식을 이용하지만, 여기서 계산한 미지수는 일반적인 스칼라
`u`이며 flux는 유체 속도가 아니다.

기존 generator·학습·benchmark 코드는 수정하지 않는다. 그림 생성은 학습이나
model selection이 아니며, final/OOD 문제를 생성하거나 읽지 않는다.

## PDE와 이산화

\[
-\nabla\cdot(\widetilde K\nabla u)=f\quad\text{in }(0,1)^2,
\qquad u|_{\partial\Omega}=0,
\qquad \widetilde K=sK,\quad s>0.
\]

내부 격자점 수는 방향마다 `n`개, 전체 미지수는 `n*n`개이며
`h=1/(n+1)`이다. 경계 포함 격자는 `(n+2)*(n+2)`개 점을 가진다.
각 사각형을 동일한 대각선으로 두 삼각형으로 나누고 P1 유한요소를 적용한다.
텐서는 각 요소의 무게중심에서 평가하며 경계 미지수는 제거한다.

\[
A^T_{ij}=|T|\,\nabla\phi_i^T\widetilde K_T\nabla\phi_j.
\]

원래 코드는 `src/adaptive_mg/pde.py`, `src/adaptive_mg/v67/data.py`,
`src/adaptive_mg/v67/research_data.py`에 있다. 이 경로에는 반응항·질량항이나
인위적인 대각 보정이 없다. 내부 배열은 `x` 인덱스가 먼저인 순서이고,
그림은 실제 좌표와 삼각형 연결 정보를 사용한다.

다음을 정의하면 7개 family의 계수는 아래와 같다.

\[
K_\theta=R_\theta\operatorname{diag}(1,\epsilon)R_\theta^T,
\quad
R_\theta=\begin{pmatrix}\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta\end{pmatrix}.
\]

| Family | 기본 텐서 `K(x,y)` |
|---|---|
| near_isotropic | `K_theta`, `epsilon`은 0.8~1 |
| aligned_anisotropy | `K_theta`, `theta`는 0도 또는 90도 |
| rotated_anisotropy | `K_theta`, 개발 분할의 각도는 15·30·45·60·75도 |
| local_patch | 지정 직사각형 안에서 `contrast*K_theta`, 밖에서 단위행렬 |
| checkerboard | `floor(m*x)+floor(m*y)`가 짝수인 블록에서 `contrast*K_theta`, 나머지 단위행렬; `m`은 3·4·5 |
| channel | `abs(y-x) <= channel_width`에서 `contrast*K_theta`, 밖에서 단위행렬 |
| variable_coefficient | `exp(amplitude*sin(2*pi*frequency*x+phase)*cos(2*pi*y-phase))*K_theta` |

최종 텐서는 모두 `s`를 곱한 값이다. near-isotropic 이외의 기본
`epsilon` 범위는 `10^-3~10^-1`, 이질 계수의 `contrast` 범위는 10~1000,
행렬 스케일 `s`는 `10^-0.5~10^0.5`이다. 그림마다 **선택한 원본 record의
실제 파라미터**를 적용한다. `channel_width`는 코드의 `abs(y-x)` 기준이며
대각선까지의 수직 거리로 재해석하지 않는다.

## 원래 실험 RHS와 공통 source의 구분

`--rhs study`가 기본값이다. 기존 실험의 첫 번째 RHS는

\[
u_* = \sin(\pi x)\sin(\pi y)
 +\sum_{j=1}^{4}a_j\sin(k_j\pi x)\sin(l_j\pi y),
\qquad b=Au_*.
\]

여기서 mode·진폭은 원본 record의 seed로 복원한다. `u_*`는 **이산 선형계의
제조해**이다. 이를 연속 PDE의 정확해로 간주하거나 `b`를 연속 source
`f(x,y)`의 점별 값이라고 표시하지 않는다. 그림의 RHS 패널은 조립된 load
`b_i`로 표기한다. family별 해의 차이는 제조해 seed에도 의존하므로,
그 차이를 동일한 외력에 대한 재료 응답 차이라고 설명하면 안 된다.

`--rhs unit_source`는 같은 행렬에 공통 source `f=1`을 적용한 **별도 보조
문제**이다. 이 삼각형 격자의 내부 절점에서는 정확한 P1 load가

\[
b_i=\int_\Omega\phi_i\,dx\,dy=h^2
\]

이므로 이 값을 사용한다. 동일 forcing에 대한 계수장별 응답을 보여주기에
적합하지만, 기존 benchmark의 RHS라고 보고하지 않는다. `--rhs both`를
쓰면 두 조건을 별도 폴더와 그림으로 만든다.

## 해와 flux의 의미

수치해 그림은 **전체 fine-grid 행렬을 `scipy.sparse.linalg.splu`로 푼
reference 해**이다. 최하위 격자만 LU로 푸는 MG 실행과 구분한다.
원본 설정의 `max(atol, rtol*||b||)` 기준으로 직접 풀이 잔차를 확인하며,
기준을 만족하지 못하면 성공한 reference 그림으로 처리하지 않는다.

요소별 flux는 실제 요소 텐서와 P1 해의 기울기를 사용한다.

\[
\mathbf q_T=-\widetilde K_T\nabla u_h|_T.
\]

flux는 요소별로 계산하며 계수 불연속 경계에서 평활화하지 않는다.
화살표의 길이는 정규화한 방향 표시이고 색상이 flux 크기를 나타낸다.
흰 선분으로 표시한 주 확산축은 부호가 없는 방향축이다.
이는 유체 속도·유선이나 시간에 따른 유동 변화가 아니다.

`--with-mg`는 동일 문제에 기존 `fixed_C`인 `line_alt + bilinear + full`
MG를 추가로 실행하여 실제 잔차 이력·cycle 수·성공/실패를 저장한다.
계수장·해·flux 패널은 계속 LU reference이며, MG가 실패하면 잔차 패널에
실패를 표시한다. NN checkpoint는 필요하지 않고 NN과의 성능 비교도 하지 않는다.
기록된 단회 시간은 그림 생성 진단용이며, warm/cold 성능 benchmark 근거가 아니다.

## Mac에서 실제 완료된 run의 문제로 생성

기존 가상환경을 활성화한 뒤 새 스크립트를 받아 온다. 새 plotting 의존성은
별도로 설치한다.

```bash
cd "$HOME/Desktop/adaptive-multigrid-light"
source .venv/bin/activate
git pull --ff-only origin residual-u-net-teacher
python -m pip install matplotlib
```

현재 subset run의 validation 중 각 family에서 **n=63, sample index=0**을
결과를 보지 않고 일괄 선택한다. 실패 여부에 따라 다른 sample로 바꾸지 않는다.

```bash
LIGHT_RUN="$(cat artifacts/light_subset_current.txt)"
FIG_RUN="artifacts/pde_figures_$(date +%Y%m%d_%H%M%S)"
python -u scripts/run_v6_7_pde_figures.py --run-dir "$LIGHT_RUN" \
  --split validation --n 63 --sample-index 0 --rhs both --with-mg --output "$FIG_RUN"
```

`--run-dir`는 `configuration.json`, `data/development_manifest.json`,
`data/dataset_plan.json`을 가진 기존 EM-line·light·subset run을 가리킨다.
해당 분할에 존재하는 격자와 sample index를 사용해야 한다. 원본 run의
선택된 행렬·RHS·제조해 hash와 plan digest를 검사하고 source 파일 hash를
기록한다. 이는 **그림에 사용한 문제의 복원 검증**이며, 과거 run의 모든
학습·평가·checkpoint를 다시 인증한다는 뜻은 아니다.

그림 출력은 원본 run 바깥의 **새 폴더**여야 한다. 원본 manifest,
freeze hash, checkpoint, 평가 결과는 수정하지 않는다. 재실행할 때도 새
출력 폴더를 지정한다. 이 스크립트는 benchmark resume 명령이 아니다.

## 기존 run이 없는 환경에서 예시 생성

```bash
FIG_RUN="artifacts/pde_illustrations_$(date +%Y%m%d_%H%M%S)"
python -u scripts/run_v6_7_pde_figures.py \
  --config configs/v6_7_light_smoother_research.json --n 63 \
  --rhs both --with-mg --output "$FIG_RUN"
```

이 경우 동일 generator와 seed 규칙으로 만든 **설정 기반 예시**라고
그림과 manifest에 명시한다. 사용자의 기존 run에서 historical 중복 회피로
선택됐을 수 있는 retry를 복원하지 않으므로, 실제 validation case를 그대로
읽은 결과라고 주장하지 않는다. 실제 run의 그림이 필요하면 위 `--run-dir`
경로를 사용한다.

## 생성 파일

| 파일 | 내용 |
|---|---|
| `coefficients.png`, `.svg` | 7개 family의 최대 고유값과 주 확산축 |
| `anisotropy.png`, `.svg` | 최대/최소 고유값 비와 주 확산축 |
| `study_solutions.png`, `.svg` | 원래 첫 RHS에 대한 7개 수치해 |
| `study_flux.png`, `.svg` | 원래 첫 RHS에 대한 확산 flux |
| `unit_source_solutions.png`, `.svg` | 선택적 공통 `f=1` 수치해 |
| `unit_source_flux.png`, `.svg` | 선택적 공통 `f=1` 확산 flux |
| `study/`, `unit_source/`의 `<family>.png`, `.svg` | family별 계수·이방성·해·load·flux·잔차 또는 설명의 6패널 그림 |
| 각 폴더의 `<family>_A.npz` | 실제 사용한 SciPy 희소 행렬 |
| 각 폴더의 `<family>_fields.npz` | 좌표·삼각형·텐서·gradient·flux·해·RHS·선택적 제조해 |
| 각 폴더의 `<family>_mg.json`, `_mg.npz` | `--with-mg` 사용 시 MG 기록과 수치해; 실패 정보 보존 |
| `metrics.json`, `metrics.csv` | 실제 잔차, 직접 풀이 성공 여부, 제조해 대비 선형계 풀이 오차, 선택적 MG 결과 |
| `manifest.json` | 원본 record·파일 hash·소스 hash·환경·실행 조건·그림의 해석 범위 |
| `progress.json` | 진행/완료 상태 |
| `CAPTIONS.md` | 그림별 설명과 보고 시 구분할 사항 |

전체 family 비교 그림은 각 물리량에 대해 공통 색상 범위를 사용한다.
SVG에도 많은 삼각형을 표시하는 field 부분은 rasterized 되어 있으므로,
텍스트·축을 편집할 수 있다고 해서 field 자체가 완전한 벡터 도형인 것은 아니다.
기본 PNG 해상도는 180 dpi이며 `--dpi 300` 등으로 변경할 수 있다.

## 보고서 해석 예시

> 단위 정사각형 영역에서 7개 SPD 확산 텐서 family에 대한 정상 확산
> 방정식을 P1 유한요소로 이산화하였다. 그림은 확산 텐서의 최대 고유값과
> 주 방향, 수치해 및 요소별 확산 flux를 나타낸다. 원래 실험의 제조 RHS와
> 공통 source `f=1`을 적용한 보조 문제를 구분하여 제시하였다.

이 그림은 PDE·계수 구조와 수치해를 설명한다. 단독으로 NN의 수렴 개선,
총시간 개선, 연속 PDE에 대한 격자 수렴성 또는 유체해석 정확성을 증명하지 않는다.
