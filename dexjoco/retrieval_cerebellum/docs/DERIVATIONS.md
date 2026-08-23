# DexHand-IEKF-PBVS 推导补全

> 状态：纯理论推导，对应 [METHODOLOGY.md](METHODOLOGY.md)。不含 MuJoCo / 视觉网络实验。
> 数值符号校验见 `retrieval_cerebellum/theory_validation.py` 与 `tests/test_theory_validation.py`。

## 0. 记号

- \(\mathrm{Exp}:\mathbb R^6\to SE(3)\)，\(\mathrm{Log}:SE(3)\to\mathbb R^6\)，右乘扰动。
- 齐次变换 \(T=\begin{bmatrix}R&p\\0&1\end{bmatrix}\)，twist \(\xi=[\rho^\top,\phi^\top]^\top\)。
- \([\cdot]_\times\) 为叉乘矩阵；\(\mathrm{Ad}_T\) 为 6×6 伴随。
- 投影 \(\pi:\mathbb R^3\to\mathbb R^2\) 为针孔模型；\(J_\pi(y)\) 为 2×3 图像 Jacobian。

---

## 1. 双侧在手相对位姿与协方差传播

### 1.1 相对位姿定义

\[
T_W^R=T_W^R(q_t),\quad T_W^L=T_W^L(q_t),
\qquad
T_W^P=T_W^R T_R^P,\quad T_W^H=T_W^L T_L^H.
\]

控制用相对姿态：

\[
T_H^P=(T_W^H)^{-1}T_W^P
=\big(T_W^L\hat T_L^H\,\mathrm{Exp}(\delta\xi_L)\big)^{-1}
\big(T_W^R\hat T_R^P\,\mathrm{Exp}(\delta\xi_R)\big).
\]

定义名义相对姿态 \(\hat T_H^P=(T_W^L\hat T_L^H)^{-1}T_W^R\hat T_R^P\)。一阶展开：

\[
T_H^P
=\mathrm{Exp}(-\delta\xi_L)\,\hat T_H^P\,\mathrm{Exp}(\delta\xi_R).
\]

右乘误差 \(\delta\xi_{rel}\) 满足：

\[
\mathrm{Exp}(\delta\xi_{rel})
=\hat T_H^{P,-1}\,\mathrm{Exp}(-\delta\xi_L)\,\hat T_H^P\,\mathrm{Exp}(\delta\xi_R).
\]

BCH 一阶：

\[
\delta\xi_{rel}
\approx
\delta\xi_R
-\mathrm{Ad}_{\hat T_H^{P,-1}}\delta\xi_L.
\]

**结论**：相对误差 Jacobian

\[
J_{rel,err}=
\begin{bmatrix}
I_6 & -\mathrm{Ad}_{\hat T_H^{P,-1}}
\end{bmatrix},
\qquad
\delta\xi_{rel}=J_{rel,err}
\begin{bmatrix}\delta\xi_R\\\delta\xi_L\end{bmatrix}.
\]

### 1.2 协方差传播

令 \(\delta x=[\delta\xi_R^\top,\delta\xi_L^\top]^\top\)，\(P=\mathbb E[\delta x\delta x^\top]\)。则

\[
P_{rel}=J_{rel,err}\,P\,J_{rel,err}^\top.
\]

**要点**：不能写 \(P_{rel}=P_R+P_L\)；左手滑移通过 \(\mathrm{Ad}_{\hat T_H^{P,-1}}\) 耦合到 peg 相对孔轴。

---

## 2. 误差状态 IEKF 预测

状态 \(X_t=(T_R^P,T_L^H,v^P,v^H,b^R,b^L)\)，误差

\[
\delta x_t=
[\delta\xi_R^\top,\delta\xi_L^\top,\delta v_P^\top,\delta v_H^\top,\delta b_R^\top,\delta b_L^\top]^\top.
\]

### 2.1 名义预测

\[
\hat T_{R,t+1}^{P}=\hat T_{R,t}^{P}\,\mathrm{Exp}(\Delta t\,\hat v_t^P),
\qquad
\hat T_{L,t+1}^{H}=\hat T_{L,t}^{H}\,\mathrm{Exp}(\Delta t\,\hat v_t^H).
\]

\[
\hat v_{t+1}^P=\hat v_t^P,\quad \hat v_{t+1}^H=\hat v_t^H,\quad \hat b_{t+1}=\hat b_t.
\]

### 2.2 误差线性化

右乘扰动下，姿态一步传播：

\[
\delta\xi_{R,t+1}\approx \delta\xi_{R,t}+\Delta t\,\delta v_P + w_{\xi R},
\qquad
\delta\xi_{L,t+1}\approx \delta\xi_{L,t}+\Delta t\,\delta v_H + w_{\xi L}.
\]

滑移随机游走：\(\delta v_{t+1}=\delta v_t+w_v\)。偏置随机游走或常值：\(\delta b_{t+1}=\delta b_t+w_b\)。

块对角 \(F_t\)（略去耦合高阶项）：

\[
F_t=
\mathrm{blkdiag}
\big(
I_6,\,I_6,\,I_6,\,I_6,\,I_6,\,I_6
\big)
+
\Delta t
\begin{bmatrix}
0&0&I&0&0&0\\
0&0&0&I&0&0\\
0&0&0&0&0&0\\
0&0&0&0&0&0\\
0&0&0&0&0&0\\
0&0&0&0&0&0
\end{bmatrix}.
\]

过程噪声：滑移证据强时增大 \(Q_{\xi},Q_v\)，对应 METHODOLOGY §6 的 \(Q_{slip}\succ Q_{stick}\)。

---

## 3. Ego 视觉关键点观测 Jacobian

### 3.1 观测方程

peg 关键点 \(p_j^P\) 在相机系：

\[
y_j^C
=T_C^W(q_t)\,T_W^R(q_t)\,\hat T_R^P\,\mathrm{Exp}(\delta\xi_R)\,p_j^P.
\]

一阶：

\[
\delta y_j^C
=R_{C}^{RP}
\begin{bmatrix}
I_3 & -[p_j^P]_\times
\end{bmatrix}
\delta\xi_R,
\]

其中 \(R_C^{RP}\) 为 \(T_C^W T_W^R \hat T_R^P\) 的旋转部分。

像素残差：

\[
r_{j}^{vis}=u_{j}^{meas}-\pi(y_j^C),
\qquad
\delta r_j^{vis}=-J_\pi(y_j^C)\,\delta y_j^C.
\]

\[
H_{j,R}^{vis}
=-\frac{\partial h}{\partial \delta\xi}
=J_\pi(y_j^C)\,R_C^{WR}R_{RP}
\begin{bmatrix}
I_3 & -[p_j^P]_\times
\end{bmatrix}.
\]

（`theory_validation.visual_keypoint_jacobian` 返回 \(\partial h/\partial\delta\xi\)，IEKF 堆叠时取负号。）

孔座关键点对 \(\delta\xi_L\) 同理。\(H^{vis}\) 只作用于姿态误差块，对 \(v,b\) 块为零。

### 3.2 置信度加权

\[
R_{j,t}^{vis}=\frac{R_0}{\max(\gamma_{j,t},\epsilon)}+R_{occ}.
\]

低置信度等价于放大 \(R\)，在 IEKF 中自动降权，不修改 \(H\)。

---

## 4. 多指接触观测

### 4.1 表面距离（SDF）

指尖世界位置 \(p_{F_i}^W(q_t)\)。peg 侧残差：

\[
r_i^{surf}=\phi_P\!\big((T_W^P)^{-1}p_{F_i}^W\big).
\]

令 \(q_i^P=(T_W^P)^{-1}p_{F_i}^W\)。对右乘误差：

\[
\delta q_i^P
\approx
-\delta\rho_R
-[q_i^P]_\times\,\delta\phi_R.
\]

链式法则：

\[
H_{i,R}^{surf}
=\nabla\phi_P(q_i^P)^\top
\big(-I_3,\,-[q_i^P]_\times\big).
\]

### 4.2 粘着速度约束

\[
r_i^{stick}
=J_{F_i}(q_t)\dot q_t - J_{P,i}(\hat X_t)\,V_P.
\]

\(V_P\in\mathbb R^6\) 为 peg 空间 twist。对 \(\delta\xi_R\) 和 \(\delta v_P\) 线性化得 \(H^{stick}\)。滑移时增大 \(R^{stick}\) 并提高 \(Q_{slip}\)，而非硬约束。

### 4.3 多指信息矩阵

\[
\mathcal I_{hand}=\sum_i H_i^\top R_i^{-1} H_i.
\]

任务相关子空间上 \(\mathrm{rank}(\mathcal I_{hand})\) 随接触指尖数、法向分散度增加。单指通常只约束 1–2 个自由度。

---

## 5. IEKF 更新（单帧迭代）

堆叠 \(r_t\)、\(H_t\)、\(R_t\)。第 \(k\) 次迭代在当前 \(\hat X^{(k)}\) 处重算 \(H,r\)：

\[
S_t=H_t P_t^- H_t^\top + R_t,
\quad
K_t=P_t^- H_t^\top S_t^{-1},
\quad
\delta x_t=K_t r_t.
\]

姿态注入：

\[
\hat T_R^{P,+}=\hat T_R^{P,-}\mathrm{Exp}(\delta\xi_R),\quad
\hat T_L^{H,+}=\hat T_L^{H,-}\mathrm{Exp}(\delta\xi_L).
\]

Joseph 协方差：

\[
P^+=(I-KH)P^-(I-KH)^\top + K R K^\top.
\]

NIS 门控：\(\mathrm{NIS}=r^\top S^{-1}r>\tau_{\chi^2}\) 则拒收或放大 \(R\)。

**迭代 IEKF**：重复至 \(\|\delta x\|<\epsilon\) 或达 \(N_{iter}\)。与 Invariant EKF 不同，此处不假设群不变噪声结构。

---

## 6. PBVS 误差与双臂多指速度 Jacobian

### 6.1 位姿误差

\[
\hat T_H^P
=(T_W^L\hat T_L^H)^{-1}(T_W^R\hat T_R^P),
\qquad
e=\mathrm{Log}\!\big((T_{H,m}^{P,*})^{-1}\hat T_H^P\big).
\]

圆柱孔去掉绕轴分量：\(\bar e=S_m e\)。

### 6.2 误差动力学（一阶）

\[
\dot{\bar e}
=J_{rel}(\hat X,q)\,u + d^{slip},
\qquad
u=
\begin{bmatrix}
\dot q_{arm,R}\\\dot q_{arm,L}\\\dot q_{hand,R}\\\dot q_{hand,L}
\end{bmatrix}.
\]

\(J_{rel}\) 由链式法则：

\[
T_H^P=f(T_W^R(q),T_R^P,T_W^L(q),T_L^H),
\]

\[
J_{rel}=S_m L_e
\begin{bmatrix}
J_P^{arm,R} & J_H^{arm,L} & J_P^{hand,R} & J_H^{hand,L}
\end{bmatrix},
\]

其中 \(J_P^{arm,R}=\partial \mathrm{Log}(T_{des}^{-1}T_H^P)/\partial \dot q_{arm,R}\) 等，在 \(\hat X\) 处求值。\(L_e\) 为 \(\mathrm{Log}\) 在 \(\hat e\) 处的左 Jacobian 逆。

**与普通 PBVS 区别**：显式保留 \(J^{hand,\cdot}\)，允许手指调整在手姿态而不只动腕部。

### 6.3 控制律与平衡点

理想反馈 \(u=-J_{rel}^\dagger \Lambda \hat e\)，\(\Lambda\succ0\)。估计误差 \(\tilde e=\hat e-e\)：

\[
\dot e = -\Lambda e - \Lambda \tilde e + d^{slip}.
\]

**命题 B 平衡点**：若 \(\dot e=0\) 且常值扰动，

\[
0=-\Lambda e_\infty - \Lambda \tilde e + d^{slip}
\quad\Rightarrow\quad
e_\infty = -\tilde e + \Lambda^{-1} d^{slip}.
\]

对角 \(\Lambda\) 时逐分量成立；METHODOLOGY §15.2 的邻域半径由此给出。

### 6.4 Lyapunov 不等式（命题 B 完整步）

\(V=\frac12 e^\top Q e\)，\(Q\succ0\)：

\[
\dot V = e^\top Q\dot e
\le e^\top Q(-\Lambda e - \Lambda\tilde e + d^{slip}).
\]

若 \(Q\Lambda\succeq \alpha I\)，Young 不等式：

\[
\dot V
\le -\alpha\|e\|^2 + \|e\|(\beta\|\tilde e\|+\gamma\|d^{slip}\|).
\]

当 \(\|e\|>r=(\beta\|\tilde e\|+\gamma\|d^{slip}\|)/\alpha\) 时 \(\dot V<0\)。

---

## 7. 双臂多指 QP 与抓持零空间

### 7.1 优化问题

\[
\min_{u,\lambda}
\frac12\|J_{rel}u+\Lambda_m\bar e\|_{Q_m}^2
+\frac12\|u\|_{R_u}^2
+\frac12\|J_h\dot q_h - G^\top V_o\|_{W_c}^2
+\frac12\|\lambda-\lambda^*\|_{W_\lambda}^2.
\]

s.t. 关节限幅、摩擦锥、法向力下界、孔壁非穿透。

### 7.2 内部力零空间

接触力分解 \(f_c=f_{motion}+N_G\lambda\)，\(GN_G=0\)。物体 wrench：

\[
w_o=G f_c = G f_{motion}.
\]

**命题**：对任意 \(\lambda\)，\(G(N_G\lambda)=0\)，故第四项只调节抓持裕度，不改变 \(w_o\)，在一阶上不干扰 PBVS 项——前提是 \(f_{motion}\) 由前三项决定、\(N_G\) 张成纯内部力。

KKT 必要条件（无不等式活跃时）：

\[
J_{rel}^\top Q_m(J_{rel}u+\Lambda_m\bar e)+R_u u + \cdots = 0,
\]

\[
G^\top W_c(J_h\dot q_h - G^\top V_o)=0,\quad
N_G^\top W_\lambda(\lambda-\lambda^*)=0.
\]

### 7.3 与 IEKF 的耦合（命题 C 的假设）

IEKF 不将 PBVS 目标写入观测。级联系统：

\[
\dot e = -\Lambda e - \Lambda\tilde e + d^{slip},
\qquad
\dot{\tilde e} = f_{iekf}(\tilde e, n_t).
\]

**未证部分**：需 \(f_{iekf}\) 输入到状态稳定（ISS）且 \(\|\tilde e\|\) 有界；遮挡、模型失配、QP 饱和可破坏该假设。METHODOLOGY §15.4 已列出。

---

## 8. 可观测性与 SEARCH 的充分条件（必要非充分）

任务子空间 \(\mathcal S=\mathrm{span}(S_m)\)。若

\[
\mathrm{rank}\!
\begin{bmatrix}
H_t^{vis}\\H_t^{hand}
\end{bmatrix}
\big|_{\mathcal S}
< \dim(\mathcal S),
\]

则局部不可观测。SEARCH 通过改变 \(T_C^W,T_W^R,T_W^L\) 使 \(H^{vis}\) 行空间变化，增大 \(\mathcal I_{hand}+\mathcal I_{vis}\) 的秩——这是**启发式**，不是全局收敛证明。

---

## 9. 推导可信度分层

| 层级 | 内容 | 状态 |
|------|------|------|
| L1 | SE(3) 相对误差、协方差、视觉/接触 Jacobian | 标准李群 + PBVS，可解析验证 |
| L2 | IEKF 线性化 + NIS 门控 | 标准误差状态 EKF |
| L3 | PBVS 平衡点、Lyapunov 邻域（命题 A/B） | 局部线性，已闭式 |
| L4 | 多指 QP 零空间分离 | 经典 grasp，KKT 下成立 |
| L5 | IEKF–PBVS 级联 ISS（命题 C） | **假设链**，需 O2/O3 |
| L6 | SEARCH/模式切换全局稳定 | **未证**，仅滞回启发 |

---

## 10. 与实现的对应

| 推导对象 | 代码函数 |
|----------|----------|
| §1 相对误差 Jacobian | `relative_right_error_jacobian` |
| §1.2 协方差传播 | `propagate_relative_covariance` |
| §3 视觉 Jacobian | `visual_keypoint_jacobian` |
| §4.1 表面距离 Jacobian | `contact_sdf_jacobian` |
| §6.3 线性 PBVS 平衡点 | `pbvs_linear_rollout` |
| §6.3 SE(3) PBVS 滚动 | `pbvs_se3_rollout` |
| §7.2 抓持零空间 | `grasp_internal_force_nullspace` |
