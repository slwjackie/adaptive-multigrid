# Q1 사각형 FEM: 구현과 7개 PDE family의 수학적 이산화

이 문서는 `residual-u-net-teacher`의 선택 가능한 Q1 이산화를 설명한다.
기존 삼각형 P1 경로는 기본값으로 유지한다. Q1은 PDE나 계수장의 정의를
바꾸지 않고, 가장 미세한 격자의 형상함수와 요소 적분을 바꾼다.
Q1 추가 자체가 multigrid의 수렴 속도 또는 NN의 우위를 보장하지는 않는다.

## 1. 선택 방법과 구현 위치

U-net, EM-line, joint/learned-P, light-smoother, direct-pruning 및 sparse-smoother
연구 경로의 설정 JSON 최상위에 다음 항목을 사용한다. 이들 경로와 공통
research plan을 쓰는 three-pillars/warm 연구에도 선택값이 전파된다.
과거 `dataset()` 기반 별도 실행기의 설정까지 일괄 변경하는 옵션은 아니다.

```json
{
  "discretization": "q1"
}
```

생략하면 기존 `p1`이다. Q1을 위해 별도 설정 파일을 제공한다.

- `configs/v6_7_unet_transfer_q1_smoke.json`
- `configs/v6_7_unet_transfer_q1_research.json`
- `configs/v6_7_em_line_q1_smoke.json`
- `configs/v6_7_em_line_q1_research.json`

예를 들어 저장소 루트에서 U-net 경로의 작은 배선 검증을 시작하려면:

```bash
python scripts/run_v6_7_unet_transfer_study.py prepare \
  --config configs/v6_7_unet_transfer_q1_smoke.json \
  --run-dir artifacts/unet_q1_smoke
python scripts/run_v6_7_unet_transfer_study.py train \
  --run-dir artifacts/unet_q1_smoke --stage direct
```

EM-line 경로는 다음과 같다.

```bash
python scripts/run_v6_7_em_line_study.py prepare \
  --config configs/v6_7_em_line_q1_smoke.json \
  --run-dir artifacts/em_line_q1_smoke
python scripts/run_v6_7_em_line_study.py train \
  --run-dir artifacts/em_line_q1_smoke
```

`research` 설정은 실제 연구 예산을 사용하므로 smoke 검증보다 훨씬 오래 걸린다.
기존 P1 실행 디렉터리를 재사용하지 말고 Q1 전용 디렉터리를 사용한다.
기존 manifest와 checkpoint의 재현성 검사는 source/config/data 변경을 검출한다.
같은 구조의 P1 checkpoint를 Q1에 넣는 것은 이산화 변경에 대한 일반화 실험이며,
Q1으로 새로 학습한 모델의 성능과 구별해서 보고해야 한다.

Python에서 기본 문제를 직접 만드는 예:

```python
from adaptive_mg.pde import DiffusionCase, assemble_problem

case = DiffusionCase(n=31, epsilon=1e-3, angle_deg=45,
                     pattern="uniform", discretization="q1")
A, b, u_star = assemble_problem(case)
assert A.shape == (31**2, 31**2)
```

7개 family의 연구 계획은
`make_research_plan(discretization="q1")`로 만든다. 선택 정보는 계획·split·case에
전파되어 나중에 같은 행렬을 복원할 때도 사용된다.

| 역할 | 코드 |
|---|---|
| Q1 형상함수·Gauss 적분·벡터화 조립 | `src/adaptive_mg/pde.py` |
| P1/Q1 공통 기본 계수장 | `diffusion_tensor`, `DiffusionCase` |
| 매끄러운 가변계수의 P1/Q1 조립 | `src/adaptive_mg/v67/data.py` |
| 7개 family·OOD·seed·행렬/RHS 재구성 | `src/adaptive_mg/v67/research_data.py` |
| Bilinear prolongation 가중치 | `src/adaptive_mg/transfer.py` |

## 2. 공통 연속 문제와 약형

7개 family는 서로 다른 종류의 지배방정식 7개가 아니다. 다음 스칼라
타원형 확산 방정식에서 확산 텐서장 \(K(x,y)\)가 달라지는 문제군이다.

\[
\Omega=(0,1)^2,\qquad
-\nabla\cdot(K\nabla u)=f\quad\text{in }\Omega,
\qquad u=0\quad\text{on }\partial\Omega.
\]

여기서 \(u\)는 스칼라이지만 \(K\)는 대칭 양의 정부호인 \(2\times2\) 텐서다.
유체의 Navier–Stokes 방정식이나 시간 적분을 직접 푸는 것은 아니다.
예를 들어 `channel`도 높은 확산계수를 가진 띠 모양 영역을 뜻한다.

\(v\in H_0^1(\Omega)\)를 곱하고 부분적분하면, \(v=0\)이므로 경계항이 사라져

\[
\boxed{
\int_\Omega (\nabla v)^T K\nabla u\,d\Omega
=\int_\Omega vf\,d\Omega
\qquad\forall v\in H_0^1(\Omega)
}
\]

가 된다. 대칭 이차형식은

\[
a(u,v)=\int_\Omega (\nabla v)^T K\nabla u\,d\Omega
\]

이다. Q1 FEM은 유한차원 공간 \(V_h\subset H_0^1(\Omega)\)에서 이 조건을
만족하도록 미지의 절점값을 구한다.

## 3. 격자와 자유도

`n`은 한 방향의 **내부 절점 수**다. 간격과 전체 격자점은

\[
h=\frac1{n+1},\qquad
(x_i,y_j)=(ih,jh),\quad i,j=0,\ldots,n+1.
\]

따라서 전체 절점은 \((n+2)^2\)개, 사각 요소는 \((n+1)^2\)개,
Dirichlet 경계값을 제거한 미지수는 \(n^2\)개다.
P1은 사각형 하나를 삼각형 2개로 나누지만 Q1은 사각형 1개로 사용한다.
절점 위치와 미지수 수는 같다.

한 요소는

\[
E_{ij}=[ih,(i+1)h]\times[jh,(j+1)h]
\]

이고, 국소 꼭짓점 순서는 반시계 방향이다.

| 국소 번호 | 전체 격자 좌표 | 참조좌표 |
|---|---|---|
| 1 | \((i,j)\) | \((0,0)\) |
| 2 | \((i+1,j)\) | \((1,0)\) |
| 3 | \((i+1,j+1)\) | \((1,1)\) |
| 4 | \((i,j+1)\) | \((0,1)\) |

코드의 내부 절점 인덱스는 0부터 시작하는

\[
I(i,j)=(i-1)n+(j-1),\qquad 1\le i,j\le n
\]

이다. 즉 `meshgrid(..., indexing="ij")`와 `ravel()`의 순서에 맞춰
두 번째 좌표 \(j\)가 먼저 변한다. 시각화에서 어느 축을 수평으로 놓는지와
행렬의 평탄화 순서는 구별해야 한다.

## 4. Q1 형상함수: 평균 한 개로 요소를 대체하지 않는다

참조 사각형 \(\widehat E=[0,1]^2\)의 좌표를 \((\xi,\eta)\)라고 하자.
Q1의 네 형상함수는

\[
\begin{aligned}
N_1&=(1-\xi)(1-\eta),&N_2&=\xi(1-\eta),\\
N_3&=\xi\eta,&N_4&=(1-\xi)\eta.
\end{aligned}
\]

네 꼭짓점의 미지수를 \(U_1,\ldots,U_4\)라 하면 요소 내부의 근사해는

\[
\boxed{u_h|_E(\xi,\eta)=\sum_{a=1}^4 U_aN_a(\xi,\eta)}.
\]

이는 \(\alpha+\beta\xi+\gamma\eta+\delta\xi\eta\) 꼴이다.
Q1의 1은 각 좌표 방향에서 1차라는 뜻이며, 총차수가 1이라는 뜻은 아니다.
\(\xi\eta\) 항을 포함하지만 Q2는 아니다.

오직 중심에서만 \(N_a(1/2,1/2)=1/4\)이므로

\[
u_h(1/2,1/2)=\frac{U_1+U_2+U_3+U_4}{4}.
\]

요소 전체를 이 평균값으로 놓고 푸는 것이 아니다. 형상함수는 위치마다
달라지며, 행렬을 조립할 때도 형상함수의 **기울기**가 들어간다.

예를 들어 \((U_1,U_2,U_3,U_4)=(0,0,1,0)\)이면
\(u_h=\xi\eta\), 중심값은 \(1/4\), 기울기는
\((\partial_\xi u_h,\partial_\eta u_h)=(\eta,\xi)\)이다.
값과 기울기는 요소 내부에서 계속 변한다.

## 5. 좌표 변환, Jacobian, 기울기

현재 코드는 축에 정렬된 정사각형을 사용한다. 일반 직사각형까지 나타내면

\[
x=x_0+h_x\xi,\qquad y=y_0+h_y\eta,
\qquad
J=\begin{bmatrix}h_x&0\\0&h_y\end{bmatrix},
\qquad |\det J|=h_xh_y.
\]

실제 구현에서는 \(h_x=h_y=h\)이다. 참조 기울기는

\[
\begin{array}{c|cc}
a&\partial_\xi N_a&\partial_\eta N_a\\\hline
1&-(1-\eta)&-(1-\xi)\\
2&1-\eta&-\xi\\
3&\eta&\xi\\
4&-\eta&1-\xi
\end{array}
\]

이며, 물리좌표의 기울기는 연쇄법칙으로

\[
\nabla_xN_a=J^{-T}\widehat\nabla N_a
=\begin{bmatrix}
h_x^{-1}\partial_\xi N_a\\h_y^{-1}\partial_\eta N_a
\end{bmatrix}.
\]

요소 강성행렬의 원래 정의는

\[
\boxed{
A^E_{ab}=\int_0^1\int_0^1
(J^{-T}\widehat\nabla N_a)^T
K(x_0+h_x\xi,y_0+h_y\eta)
(J^{-T}\widehat\nabla N_b)
h_xh_y\,d\xi\,d\eta
}.
\]

정사각형에서는 두 기울기의 \(1/h\)와 면적의 \(h^2\)가 상쇄된다.
그래서 2차원 stiffness의 일정 계수 요소행렬에 \(h^{-2}\)가 남지 않는다.
유한차분의 강형식 행렬과 FEM stiffness를 비교할 때 주의할 부분이다.
FEM의 우변은 본래 면적 적분을 포함하며, 이 저장소의 제조 우변은 뒤에서
설명하는 \(b=AU_*\) 방식이다.

## 6. 실제 구현: 사각형당 2×2 Gauss 적분

단위 구간의 두 Gauss점과 가중치는

\[
\xi_- =\frac12\left(1-\frac1{\sqrt3}\right),\qquad
\xi_+ =\frac12\left(1+\frac1{\sqrt3}\right),
\qquad w_-=w_+=\frac12.
\]

텐서곱으로 사각형 내부의 네 점

\[
(\xi_-,\xi_-),\ (\xi_+,\xi_-),\
(\xi_+,\xi_+),\ (\xi_-,\xi_+)
\]

을 사용하며 각 가중치는 \(w_q=1/4\)다. 요소마다 각 점의 물리좌표에서
\(K_q=K(x_q,y_q)\)를 평가한다. 구현하는 이산 행렬은

\[
\boxed{
\widetilde A^E_{ab}
=\sum_{q=1}^4\frac{h^2}{4}
(\nabla_xN_a(x_q,y_q))^TK_q\nabla_xN_b(x_q,y_q)
=\frac14\sum_{q=1}^4
(\widehat\nabla N_a(\xi_q,\eta_q))^T
K_q\widehat\nabla N_b(\xi_q,\eta_q)
}.
\]

벡터화 구현은 같은 Gauss점의 계수 평가를 여러 요소에 대해 한꺼번에
수행한다. 수학적으로는 위의 합과 동일하며, scalar reference 조립 경로도 있다.

일정 텐서 \(K\)에서는 integrand의 각 좌표별 차수가 2 이하이므로
2점 Gauss 적분의 3차 다항식 정확성으로 요소행렬을 정확히 적분한다.
매끄러운 가변계수나 요소 내부를 가르는 불연속 계수에서는 일반적으로
근사 적분이다.

중심 한 점만 사용하는 Q1 reduced integration은 채택하지 않는다.
예를 들어 \(u_h=(\xi-1/2)(\eta-1/2)\)는 0이 아닌 함수인데 중심에서
기울기가 0이다. 중심 한 점 에너지 평가만으로는 이런 모드를 놓칠 수 있다.
네 Gauss점은 이를 구별한다.

## 7. 일정 텐서의 요소행렬을 전개하면

\[
K=\begin{bmatrix}k_{xx}&k_{xy}\\k_{xy}&k_{yy}\end{bmatrix}
\]

가 요소 내 일정하면, 정사각형에서

\[
A^E=k_{xx}M_x+k_{yy}M_y+k_{xy}M_{xy},
\]

\[
M_x=\frac16\begin{bmatrix}
2&-2&-1&1\\-2&2&1&-1\\-1&1&2&-2\\1&-1&-2&2
\end{bmatrix},\qquad
M_y=\frac16\begin{bmatrix}
2&1&-1&-2\\1&2&-2&-1\\-1&-2&2&1\\-2&-1&1&2
\end{bmatrix},
\]

\[
M_{xy}=\frac12\begin{bmatrix}
1&0&-1&0\\0&-1&0&1\\-1&0&1&0\\0&1&0&-1
\end{bmatrix}.
\]

이는 각각
\(\int\partial_\xi N_a\partial_\xi N_b\),
\(\int\partial_\eta N_a\partial_\eta N_b\),
\(\int(\partial_\xi N_a\partial_\eta N_b+
\partial_\eta N_a\partial_\xi N_b)\)의 값이다.
직사각형에서는 \(M_x\)에 \(h_y/h_x\), \(M_y\)에 \(h_x/h_y\)가 곱해지고
\(M_{xy}\)는 그대로다. 현재 square-grid 구현은 이 비율이 모두 1이다.

특히 \(K=I\)이면

\[
\boxed{A^E=\frac16\begin{bmatrix}
4&-1&-2&-1\\-1&4&-1&-2\\-2&-1&4&-1\\-1&-2&-1&4
\end{bmatrix}}.
\]

이 행렬은 단일 요소의 상수 모드 때문에 양의 준정부호다.
전체 mesh에 조립하고 0 Dirichlet 조건을 적용한 내부 행렬이 양의 정부호가 된다.

## 8. 7개 family의 확산 텐서와 실제 Gauss점 평가

공통 기본 텐서를

\[
R_\theta=\begin{bmatrix}\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta\end{bmatrix},\qquad
D_{\epsilon,\theta}=R_\theta
\begin{bmatrix}1&0\\0&\epsilon\end{bmatrix}R_\theta^T
\]

라고 놓자. 성분별로는

\[
D_{\epsilon,\theta}=
\begin{bmatrix}
\cos^2\theta+\epsilon\sin^2\theta&(1-\epsilon)\sin\theta\cos\theta\\
(1-\epsilon)\sin\theta\cos\theta&\sin^2\theta+\epsilon\cos^2\theta
\end{bmatrix}.
\]

고유값은 1과 \(\epsilon>0\), 강한 확산 방향은
\((\cos\theta,\sin\theta)^T\)다. 이하 `contrast`를 \(c>0\)라 쓴다.
생성된 행렬에는 나중에 양의 전체 scale \(s\)도 곱해진다.

### 8.1 `near_isotropic`: 거의 등방성인 일정 계수

\[
K(x,y)=D_{\epsilon,\theta},\qquad
\epsilon\sim U(0.8,1),\quad c=1.
\]

모든 요소·모든 Gauss점에서 같은 텐서를 사용한다. 따라서 7절의 닫힌 식과
Gauss 조립이 일치한다. \(\epsilon=1\)인 극한은 \(K=I\)이며 각도에
무관하다. 실제 무작위 표본은 보통 정확한 Poisson이 아니라 약한 비등방성이다.

### 8.2 `aligned_anisotropy`: 격자축에 정렬된 일정 비등방성

\[
K=D_{\epsilon,\theta},\qquad\theta\in\{0^\circ,90^\circ\},\quad c=1.
\]

\(\theta=0\)이면 \(K=\operatorname{diag}(1,\epsilon)\),
\(\theta=90^\circ\)이면 \(K=\operatorname{diag}(\epsilon,1)\)이다.
따라서 \(\theta=0\)의 요소행렬은

\[
A^E=M_x+\epsilon M_y.
\]

연속 강형식은 \(-u_{xx}-\epsilon u_{yy}=f\)이고,
90도에서는 두 좌표의 역할이 바뀐다.

### 8.3 `rotated_anisotropy`: 회전된 일정 비등방성

\[
K=D_{\epsilon,\theta},\qquad c=1.
\]

일정 계수이므로 강형식을 전개할 수 있다.

\[
-k_{xx}u_{xx}-2k_{xy}u_{xy}-k_{yy}u_{yy}=f.
\]

Q1은 별도의 \(u_{xy}\) 유한차분 공식을 넣지 않는다.
약형의 \(k_{xy}(N_{a,x}N_{b,y}+N_{a,y}N_{b,x})\) 항을 적분하여
교차방향 결합을 자동으로 행렬에 넣는다.

예를 들어 \(\epsilon=10^{-3},\theta=45^\circ\)이면

\[
K=\begin{bmatrix}0.5005&0.4995\\0.4995&0.5005\end{bmatrix},
\quad
A^E=0.5005(M_x+M_y)+0.4995M_{xy}.
\]

Q1에서는 사각 요소를 임의의 대각선으로 나누지 않는다.
따라서 P1 삼각분할의 대각선 선택에 따른 효과는 제거되지만,
사각 격자의 축 방향에 대한 수치적 의존성까지 제거되는 것은 아니다.

### 8.4 `local_patch`: 국소 사각형 고대비 영역

패치 영역을

\[
\mathcal P=[x_{\min},x_{\max}]\times[y_{\min},y_{\max}]
\]

라고 하면

\[
\boxed{K(x,y)=\begin{cases}
cD_{\epsilon,\theta},&(x,y)\in\mathcal P,\\
I,&(x,y)\notin\mathcal P.
\end{cases}}
\]

내부는 단순히 스칼라 계수만 큰 등방 영역이 아니라 회전 비등방 텐서이고,
외부는 \(I\)다. 내부 고유값은 \(c,c\epsilon\)이므로
`contrast=c`가 모든 방향에서 외부보다 정확히 \(c\)배 강하다는 뜻은 아니다.
예를 들어 \(c=100,\epsilon=10^{-3}\)이면 내부 고유값은 100과 0.1이다.

각 Gauss점의 membership을 판정하고 해당 텐서를 넣는다.
\(\chi_q=1_{\mathcal P}(x_q,y_q)\)라 하면

\[
\widetilde A^E_{ab}=\frac14\sum_q
\widehat\nabla N_a(q)^T
\left[I+\chi_q(cD_{\epsilon,\theta}-I)\right]
\widehat\nabla N_b(q).
\]

패치 안에 완전히 들어간 요소는 \(cA^E(D)\), 완전히 바깥 요소는
\(A^E(I)\)이다. 경계가 요소를 가르면 네 점에서 서로 다른 텐서가 선택될 수
있다. 이 경우 정확한 면적분과의 일치는 일반적으로 보장하지 않는다.

### 8.5 `checkerboard`: 번갈아 배치되는 고대비 블록

방향별 블록 수를 \(m=\texttt{checker_cells}\)라 하고

\[
p(x)=\min(\lfloor mx\rfloor,m-1),\qquad
q(y)=\min(\lfloor my\rfloor,m-1)
\]

로 정의한다. 합의 짝홀에 따라

\[
K(x,y)=\begin{cases}
cD_{\epsilon,\theta},&p(x)+q(y)\text{가 짝수},\\
I,&p(x)+q(y)\text{가 홀수}
\end{cases}
\]

이다. Q1 요소의 네 Gauss점에서 위 조건을 각각 평가한다.
블록 경계가 mesh 선에 정렬되면 각 요소의 텐서가 일정하여 정확 적분이고,
정렬되지 않으면 잘린 요소에서 4점 근사 적분을 사용한다.

따라서 `checker_cells=4`이고 셀 수 \(n+1\)가 4의 배수이면 정렬되지만,
`checker_cells=3` 또는 5인 경우에는 일반적으로 그렇지 않다.

### 8.6 `channel`: 대각선 띠 영역

\[
\mathcal C=\{(x,y):|y-x|\le w\},\qquad
w=\texttt{channel_width},
\]

\[
K(x,y)=\begin{cases}
cD_{\epsilon,\theta},&(x,y)\in\mathcal C,\\
I,&\text{그 외}.
\end{cases}
\]

여기서 \(w\)는 코드가 사용하는 \(|y-x|\)의 threshold이다.
대각선 \(y=x\)에 수직인 물리적 반폭은 \(w/\sqrt2\)다.
띠의 방향은 대각선으로 고정되어 있고, 확산 텐서의 각도 \(\theta\)는
별도로 선택된다. 두 방향이 항상 일치하는 것은 아니다.

각 Gauss점에서 \(|y_q-x_q|\le w\)를 평가하고 8.4절과 같은 합을 만든다.
경계 \(y-x=\pm w\)가 셀 내부를 지나는 곳에는 적분 오차가 남을 수 있다.

### 8.7 `variable_coefficient`: 매끄러운 양의 가변계수

\[
\begin{aligned}
K(x,y)&=a(x,y)D_{\epsilon,\theta},\\
a(x,y)&=\exp\!\left[
\alpha\sin(2\pi\nu x+\phi)\cos(2\pi y-\phi)
\right],\\
\alpha&=\tfrac12\log c.
\end{aligned}
\]

\(\nu\)는 `frequency`, \(\phi\)는 `phase`다.
지수함수를 사용하므로 \(a(x,y)>0\)이고,

\[
c^{-1/2}\le a(x,y)\le c^{1/2},\qquad
\frac{\max a}{\min a}\le c
\]

이다. 유한한 영역과 선택한 \(\nu,\phi\) 때문에 양 끝값이 반드시 실제로
달성되는 것은 아니다. 따라서 \(c\)는 여기서 가능한 계수 대비의 상한이다.

요소행렬은

\[
\widetilde A^E_{ab}=
\frac14\sum_{q=1}^4 a(x_q,y_q)
\widehat\nabla N_a(q)^TD_{\epsilon,\theta}\widehat\nabla N_b(q).
\]

네 점에 서로 다른 가중치가 붙는다. 평균 계수 \(\bar a\)를 먼저 구해서
\(\bar a A^E(D)\)로 만드는 것과 일반적으로 같지 않다.
형상함수의 기울기 곱도 Gauss점마다 달라지기 때문이다.

또한 가변계수의 강형식을 전개하면

\[
-a\,\nabla\cdot(D\nabla u)-(\nabla a)^TD\nabla u=f
\]

지만, 약형 조립에서는 \(\nabla a\)를 직접 계산할 필요가 없다.
네 Gauss점의 \(aD\)를 사용하면 된다.

### 연구 생성기의 기본 parameter 범위

아래는 `research_data._specification`의 범위이며, 구형 `v67.data.dataset`
함수의 별도 난수 범위나 개별 수동 case와 혼동하면 안 된다.

| 항목 | 기본 train/validation/final |
|---|---|
| 일반 비등방성 | \(\log_{10}\epsilon\sim U(-3,-1)\) |
| near-isotropic | \(\epsilon\sim U(0.8,1)\) |
| 일반 회전각 | \(15,30,45,60,75^\circ\) 중 하나 |
| aligned 회전각 | \(0,90^\circ\) 중 하나 |
| heterogeneous contrast | \(\log_{10}c\sim U(1,3)\) |
| patch 중심 | 두 좌표 각각 \(U(0.35,0.65)\) |
| patch의 x/y 반폭 | 각각 \(U(0.13,0.24)\) |
| checker block 수 | 3, 4, 5 중 하나 |
| channel threshold \(w\) | \(U(0.06,0.13)\) |
| variable frequency \(\nu\) | \(U(0.7,2.5)\) |
| phase \(\phi\), geometry phase \(\psi\) | 각각 \(U(0,2\pi)\) |
| 전체 행렬 scale | \(\log_{10}s\sim U(-0.5,0.5)\) |

세 uniform family는 \(c=1\)로 덮어쓴다. 모든 case에 생성되는 geometry
parameter 중 일부는 해당 family에서 실제로 사용되지 않는다.
훈련 격자 수와 family당 표본 수는 study 설정이 결정하며 Q1의 수학적
정의와는 별개다.

## 9. OOD에서도 Q1을 유지한다

기존 OOD protocol은 보존된다. Grid OOD는 더 큰 격자,
rotation OOD는 \(7.5,22.5,37.5,52.5,67.5,82.5^\circ\),
anisotropy OOD는 \(\log_{10}\epsilon\sim U(-5,-3.1)\),
contrast OOD는 \(\log_{10}c\sim U(3.2,5)\)를 사용한다.
Aligned family의 각도는 해당 family 정의에 따라 계속 0/90도다.

Topology OOD는 heterogeneous 4개 family에서 계수장의 기하만 바꾼다.
선택한 Q1 형상함수, 4점 적분, 경계 제거 규칙은 그대로다.
\(\psi=\texttt{geometry_phase}\), \(d=0.04\sin\psi\)라 하자.

**Local patch:** 두 타원의 합집합을 hard 영역으로 사용한다.

\[
\left(\frac{x-0.30-d}{0.20}\right)^2+
\left(\frac{y-0.36}{0.13}\right)^2<1
\quad\text{또는}\quad
\left(\frac{x-0.72}{0.12}\right)^2+
\left(\frac{y-0.65-d}{0.23}\right)^2<1.
\]

**Checkerboard:** 행마다 어긋나는 블록을 사용한다.

\[
r=\lfloor(m+1)y\rfloor,\qquad
t=\left\lfloor mx+\tfrac12(r\bmod2)+\frac\psi{2\pi}\right\rfloor,
\qquad\text{hard}\iff(r+t)\bmod2=0.
\]

**Channel:** \(g(x)=0.20\sin(2\pi x+\psi)\)에 대해 두 곡선 띠의 합집합이다.

\[
|y-0.32-g(x)|<w
\quad\text{또는}\quad
|y-0.74+0.5g(x)|<0.65w.
\]

이 세 경우는 hard 영역 안의 \(cD\), 밖의 \(I\)를 사용한다.

**Variable coefficient:** 서로 다른 공간 주파수를 섞는다.

\[
a_{\rm OOD}(x,y)=\exp\!\left[\alpha\left\{
0.65\sin(2\pi\nu x+\phi)\cos(4\pi y)
+0.35\cos(6\pi x-2\pi y+\psi)
\right\}\right],
\qquad K=a_{\rm OOD}D.
\]

Q1은 위 조건/식을 각 물리 Gauss점에서 평가하므로,
OOD topology 경로가 P1으로 되돌아가지 않는다.

## 10. 전체 조립, Dirichlet 경계, SPD

요소의 내부 자유도를 전역 벡터에서 읽어 오는 행렬을 \(S_E\)라 하자.
경계 자유도는 값이 0이므로 해당 행이 0인 형태로 생각할 수 있다.
전역 행렬은

\[
\boxed{A_h=\sum_E S_E^T\widetilde A^ES_E\in\mathbb R^{n^2\times n^2}}.
\]

코드는 요소 기여를 COO 형식으로 모으고, 같은 전역 위치의 중복을 합산한
후 CSR로 변환한다. 대칭화 \((A+A^T)/2\)는 부동소수점 누적의 작은
비대칭을 제거하는 용도이며, SPD를 임의로 강제하는 shift가 아니다.

경계를 포함한 시스템을 블록으로 쓰면

\[
\begin{bmatrix}A_{II}&A_{IB}\\A_{BI}&A_{BB}\end{bmatrix}
\begin{bmatrix}U_I\\U_B\end{bmatrix}
=\begin{bmatrix}b_I\\b_B\end{bmatrix},\qquad U_B=0
\]

이므로 내부 방정식은 \(A_{II}U_I=b_I\)다.
구현은 처음부터 내부 행/열만 모으며, 비영 Dirichlet 경계조건에 필요한
\(-A_{IB}U_B\) 우변 보정은 현재 문제 범위에 없다.

SPD의 이유도 Gauss 적분식에서 직접 확인된다. 임의의 내부 벡터 \(V\)와
그 Q1 함수 \(v_h\)에 대해

\[
V^TA_hV=\sum_E\sum_{q=1}^4\frac{h^2}{4}
\nabla v_h(x_q)^TK(x_q)\nabla v_h(x_q)\ge0.
\]

모든 \(K(x_q)\)는 SPD이고 가중치는 양수다.
에너지가 0이면 네 점에서 기울기가 모두 0이다.
Q1 함수의 기울기는 \(\beta+\delta\eta,\gamma+\delta\xi\) 꼴이므로
서로 다른 두 \(\xi\), 두 \(\eta\)에서 모두 0이면
\(\beta=\gamma=\delta=0\)이다. 각 요소에서 상수이고, 연속성으로
연결된 전체 mesh에서 같은 상수이며, 0 경계조건 때문에 그 상수는 0이다.
따라서 \(V\ne0\)이면 \(V^TA_hV>0\)이다.

이 성질은 계수 불연속을 네 점으로 완벽히 적분했다는 주장과 다르다.
**근사 적분으로 만든 행렬의 SPD**와 **연속 PDE에 대한 정확성**은 별도다.

## 11. 왜 9-point stencil인가

Q1 절점 하나는 자기 주변 네 사각 요소에 들어간다. 그 요소에서 공유하는
절점의 합집합이 자신과 상하좌우·네 대각선이므로 최대 9개 연결이 생긴다.
계수에 따라 일부 항은 0일 수 있다.

일정 \(K=I\)의 내부 stencil은 FEM stiffness 기준으로

\[
\boxed{\frac13\begin{bmatrix}
-1&-1&-1\\-1&8&-1\\-1&-1&-1
\end{bmatrix}}.
\]

일정 텐서를 \(K=\bigl[\begin{smallmatrix}a&b\\b&c\end{smallmatrix}\bigr]\)라 쓰고,
그림의 가로축을 \(x\), 세로축을 \(y\)로 잡으면 일반식은

\[
\begin{bmatrix}
-\frac{a+c}{6}+\frac b2&\frac{a-2c}{3}&-\frac{a+c}{6}-\frac b2\\
\frac{-2a+c}{3}&\frac{4(a+c)}3&\frac{-2a+c}{3}\\
-\frac{a+c}{6}-\frac b2&\frac{a-2c}{3}&-\frac{a+c}{6}+\frac b2
\end{bmatrix}.
\]

강한 비등방성에서는 일부 비대각 원소가 양수가 될 수 있다.
예를 들어 \(a=1,c=10^{-3},b=0\)이면 수직 이웃 계수
\((a-2c)/3\)는 양수다. SPD라고 해서 M-matrix 조건이나 discrete
maximum principle, 빠른 point smoother 수렴까지 자동으로 얻는 것은 아니다.

기존 CNN의 가장 미세한 행렬 입력은 대각 성분 관련 1채널과 국소 3×3
stencil 9채널이므로 Q1 연결을 담을 수 있다. Galerkin으로 만들어진 더
거친 행렬의 support와 feature 추출 규칙은 기존 solver 설계를 따른다.

## 12. 우변은 현재도 이산 제조해 방식이다

연속 FEM의 정석적인 load vector는

\[
b_I^{\rm load}=\int_\Omega N_If\,d\Omega
\]

이며, Q1의 4점 적분으로 근사한다면

\[
\widetilde b_a^E=\frac{h^2}{4}\sum_{q=1}^4N_a(q)f(x_q,y_q)
\]

이다. 그러나 **기존 연구 데이터 생성기는 이 연속 부하 적분을 사용하지
않는다.** Q1 추가 후에도 solver 비교 조건을 유지하기 위해

\[
\begin{aligned}
U_*(x_i,y_j)
&=\sin(\pi x_i)\sin(\pi y_j)
+\sum_{r=1}^{4}\beta_r
\sin(k_r\pi x_i)\sin(\ell_r\pi y_j),\\
\beta_r&\sim U(-0.3,0.3),\qquad
k_r,\ell_r\in\{1,\ldots,\min(n,12)\},\\
A&=sA_h,\qquad \boxed{b=AU_*}
\end{aligned}
\]

를 사용한다. `assemble_problem`의 단순 예제는 합성 모드 4개 없이
첫 사인 함수만 사용한다.

따라서 \(U_*\)는 생성된 **이산 선형계**의 정확해다.
원래 연속 PDE의 특정 \(f\)에 대한 해석적 정확해라는 뜻은 아니다.
특히 불연속 \(K\)에서 임의의 매끄러운 사인 함수를 넣으면 계면 flux
연속성을 자동으로 만족하지 않는다. 이를 보통의 매끄러운 연속 source를
가진 interface 문제의 정확해로 해석하면 안 된다.

같은 계수 parameter·격자·seed를 P1과 Q1에 사용하면 같은 절점 벡터
\(U_*\)를 유지하면서 \(A\)가 달라져 각각의 \(b\)도 달라진다.
이는 각각의 이산계를 같은 목표 오차까지 푸는 solver 실험이다.
P1/Q1의 연속 PDE 이산화 오차를 비교하려면 **같은 연속 \(f\)**를 정하고
각 요소의 load를 조립해 해석해 또는 충분히 정밀한 기준해와 비교해야 한다.

전체 scale \(s\in[10^{-0.5},10^{0.5}]\)는 실제 \(A,b\)에 모두 곱해진다.
Solver 입력을 자동으로 최대 원소 1로 정규화하는 것은 아니다.
다만 duplicate/leakage 검사에서는

\[
\operatorname{round}\left(A_{ij}/\max_{kl}|A_{kl}|,11\right)
\]

의 희소 구조·값으로 scale-invariant hash를 만든다.
따라서 동일 행렬을 상수 배율만 달리해 별개의 학습/평가 operator라고
세는 일을 방지한다. 이 hash 정규화와 CNN feature 정규화는 다른 목적이다.

## 13. Bilinear prolongation과의 정확한 관계

균일하게 세분화한 Q1 공간에서는 \(V_H^{Q1}\subset V_h^{Q1}\)이다.
Coarse Q1 함수는 fine 사각형에 제한해도 각 좌표별 1차 다항식이기 때문이다.
Coarse 형상함수를 \(\Phi_J^H\), fine 절점을 \(x_i^h\)라 하면

\[
\boxed{P_{iJ}=\Phi_J^H(x_i^h)}
\]

로 설정하면 coarse 함수를 fine Q1 함수로 정확히 표현할 수 있다.
그 값은 바로 bilinear 가중치다.

- 기존 coarse 절점: injection, 가중치 1.
- coarse 변의 중점: 양 끝 가중치 각각 1/2.
- coarse 셀 중심: 네 꼭짓점 가중치 각각 1/4.

전체 내부 자유도 행렬에서는 경계 열을 제거하므로 경계에 인접한 행의
가중치 합이 1보다 작을 수 있다. 0 경계값을 곱할 항을 생략한 것이므로
이를 무조건 재정규화해 1로 만들면 다른 연산이 된다.

Galerkin 계층은 기존과 같이

\[
R_\ell=P_\ell^T,\qquad A_{\ell+1}=P_\ell^TA_\ell P_\ell
\]

를 사용한다. 바뀌는 것은 최초 \(A_0\)의 조립이고, coarse level마다
다시 Q1 PDE를 조립하는 구조가 아니다.
설정에 따라 bilinear, EM, learned \(P_\ell\)를 사용하는 기존 로직은
이 행렬 위에서 작동한다.

정확히 같은 bilinear form을 적분한 중첩 FEM이라면 자연 보간의 Galerkin
행렬과 coarse FEM 행렬은 일치한다. 그러나 variable/discontinuous \(K\)에
대해 fine 셀들의 2×2 quadrature와 coarse 셀의 2×2 quadrature를 각각 쓰면
샘플링 점이 다르므로 **재이산화한 coarse Q1 행렬과의 일치가 일반적으로
보장되지 않는다.** 현재 구현은 Galerkin 정의를 사용한다.

EM/NN이 \(P\)의 가중치를 조정하면 coarse 보정 공간도 바뀐다.
그 결과를 자연스러운 coarse Q1 공간과 무조건 동일시해서는 안 된다.
다만 \(A\)가 SPD이고 \(P\)가 full column rank이면
\(z^TP^TAPz=(Pz)^TA(Pz)>0\)이므로 Galerkin coarse 행렬도 SPD다.

## 14. 정확성과 해석에서 남는 제한

Q1 구현은 네 Gauss점으로 tensor field를 평가하며, 계수 불연속선을 추적해
요소를 나누는 interface integration이나 cut-cell 적분을 하지 않는다.
따라서 다음을 구별해야 한다.

1. **표현 오차:** Q1 공간이 정확한 연속 해를 얼마나 잘 표현하는가.
2. **적분 오차:** 네 Gauss점 합이 정확한 weak-form 적분과 얼마나 다른가.
3. **대수적 오차:** MG가 만들어진 \(AU=b\)를 얼마나 정확히 풀었는가.

현재 \(b=AU_*\)와 residual tolerance는 세 번째를 직접 측정한다.
잔차가 작고 \(U\approx U_*\)라는 사실만으로 첫째·둘째 오차가 작다고
검증되는 것은 아니다. 불연속 계면에서의 기울기 변화는 적분점 증가만으로
Q1 기저에 새로 생기지 않는다.

Q1의 직접적인 연구상 이점은 기존 bilinear 보간과의 함수 공간 일관성,
삼각분할 대각선 선택의 제거, 그리고 같은 structured-grid CNN 입력 구조에서
이산화 방식에 따른 일반화를 비교할 수 있다는 점이다. 실제 총시간의 변화는
행렬 비영 원소 수, smoother 비용, 계층 생성 비용, 필요한 cycle 수를 모두
측정해야 판단할 수 있다.
