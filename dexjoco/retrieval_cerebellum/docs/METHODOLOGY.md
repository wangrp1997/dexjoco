# DexJoCo 主方法论

## Intent-Preserving Sensor-Compliant Bimanual Insertion

最后更新：2026-08-23

本文件是当前唯一权威方法论。2026-08-23 的硬路线门已经停止 IEKF-PBVS 主线：held-out ego 视觉估计横向 P90 为 `21.06 mm`、倾斜 P90 为 `0.2077 rad`、深度 P90 为 `30.55 mm`，分别未达到 `1.2 mm`、`0.03 rad` 和 `3 mm` 门槛。三项任一失败即停止，不允许通过修改阈值、只报告训练集或使用 Oracle 输入恢复该路线。

当前唯一主线是：**保留 π0.5 的时变双腕操作意图，由只读取可部署传感器的独立小脑进行重定时、双手约束释放、柔顺执行、力安全、退让和成功/退出判断。** IEKF-PBVS 的公式推导保留在本文第二部分，作为已经停止的条件性候选档案，不再指导实现顺序或项目结论。

## 0. 路线停止决定

自动判定入口为：

```text
ego 视觉模型 held-out P1 summary
→ 三项精度硬门
→ 任一失败：STOP IEKF-PBVS
→ 切换 intent_preserving_sensor_compliance
```

判定器不提供 override 参数。当前决定文件必须由 `methodology_gate.py` 根据冻结测试 summary 生成。只有建立新的项目、提出新的传感器条件并重新预注册门槛，才允许重新研究完整姿态估计路线；当前项目内禁止复活。

## 1. 当前核心研究问题

任务是固定装配几何下的双臂灵巧手精细插孔。算法边界从“π0.5 已完成双手抓取、搬运并到达粗对准区域”开始，不负责从 `env.reset()` 解决前置长程操作。已有实验证据表明，π0.5 handoff 后若把时变双腕意图压缩成固定孔轴下探，成功代理会从 `8/9` 降到 `2/9`；真实在线 π0.5 chunk 在冻结自然 handoff 上也达到 `8/9`。因此当前问题改为：

> 如何在不估计毫米级完整物体姿态、不读取真实几何的前提下，保留 π0.5 的时变双腕接近意图，并利用腕力、多指力和本体信号安全地重定时、释放闭链内力、退让和完成插入。

主链路为：

```text
π0.5 完成抓取、粗搬运和粗对准
→ 上层状态机进入精细插入阶段并提供一个新的时变双腕 action chunk
→ 小脑把 chunk 转换成相对双腕速度意图
→ 根据腕力、多指抓持稳定性和执行偏差调节相位速度
→ 动态分配左右手运动以释放闭链内力
→ 超过安全阈值时停止、退让或请求重新规划
→ 仅由可部署信号判断继续、成功或退出
→ 成功或明确安全退出
```

设 π0.5 chunk 给出名义双腕控制 \(u_{\pi}(s)\)，\(s\) 是 chunk 相位。当前控制律为：

\[
u_t=
\Pi_{\mathcal U_t}
\left[
\dot s_tu_{\pi}(s_t)
+B_w\Delta w_t
+B_g\Delta g_t
+u_t^{retreat}
\right],
\]

其中：

- \(u_{\pi}(s_t)\) 保留 π0.5 的时变双腕方向和相对节奏；
- \(\dot s_t\in[0,1]\) 根据软力门、抓持稳定性和执行滞后连续减速；
- \(B_w\Delta w_t\) 根据双腕力残差释放闭链内力；
- \(B_g\Delta g_t\) 根据两侧多指稳定性重新分配相对运动；
- \(u_t^{retreat}\) 只在硬力门、卡紧或滑移门触发时激活；
- \(\Pi_{\mathcal U_t}\) 投影到单步位移、转角、关节、抓持和力安全集合。

相位速度写为：

\[
\dot s_t=
\alpha_{force}(\Delta w_t)
\alpha_{grasp}(g_t)
\alpha_{tracking}(q_t,u_{\pi}),
\qquad
0\le\alpha_{\cdot}\le1.
\]

任一硬安全门触发时直接令 \(\dot s_t=0\)，不得通过提高控制增益强行完成插入。

## 2. 职责边界

### 2.1 π0.5

π0.5 及其上层任务状态机负责：

- 从 `env.reset()` 开始理解任务；
- 左手抓取孔座、右手抓取 peg；
- 抬离桌面并搬运到固定装配工作空间；
- 确认系统已经进入预定义的粗对准接管区域；
- 在接管时提供一个新的时变双腕 action chunk，而不是记录示范后续动作。

π0.5 不负责力安全、闭链内力释放、chunk 重定时、退让和成功判断。小脑不能裸回放 chunk，而必须逐步投影和门控。

### 2.2 独立小脑

小脑负责：

- 将 π0.5 chunk 转换为连续双腕相对运动意图；
- 根据当前本体状态重定时并限制单步平移和旋转；
- 根据双腕力残差和多指稳定性动态分配左右手运动；
- 软力门减速，硬力门停止并执行有界退让；
- 检测 chunk 耗尽、抓持失稳、卡紧、超时和不可恢复状态；
- 仅根据可部署传感器输出成功、继续、退出或重新请求 π0.5。

当前主线明确不要求：完整 \(T_R^P,T_L^H\) 估计、IEKF、PBVS、主动协方差优化或在线 CAD 姿态恢复。视觉只能作为非必需的粗可见性或安全信号；在新的 P1 项目通过前不能进入精密控制目标。

## 3. 在线数据防火墙

在线小脑允许读取：

\[
y_t=
\left(
I_t^{ego},
q_t,\dot q_t,
f_t^{tip,R},f_t^{tip,L},
w_t^R,w_t^L,
u_{t-1}
\right).
\]

第一版允许只使用 ego RGB；腕部相机只能作为明确消融项加入，不能偷偷成为主结果输入。

在线控制严格禁止读取：

- MuJoCo peg、孔座、孔口和接触点真实位姿；
- `hybrid_insert` 中的 privileged geometry；
- 教师 handoff 帧、教师相对误差和记录后续动作；
- demo 初始状态恢复或 zarr 动作回放；
- 外部 evaluator 的成功、接触模式和几何标签；
- 根据测试 episode 真值调整的参数。

仿真真值只允许由独立 evaluator 写入离线评测文件，控制进程不能导入或调用 evaluator。

## 当前路线验收门

局部精细插入方法与完整系统集成分开验收。局部方法首先在冻结、held-out、已抓持且已粗对准的状态上验证；自然 `env.reset()` 和 π0.5 显式 handoff 只属于后续系统集成，不得反向否定局部视触觉方法。

### L0：局部接管契约

- 初态必须来自冻结 held-out 粗对准状态，且双侧工件已经抓持；
- 控制器只接收当前可部署传感、新 π0.5 action chunk 和接管时刻的上游任务契约；
- 记录轨迹只允许恢复和定义测试初态，恢复后不得向控制器提供后续动作或真实几何；
- privileged evaluator 只能事后评分，不能生成动作或改变控制参数。

### L1：意图保持执行

- 使用真实 π0.5 chunk，不使用记录后续动作；
- 与 π0.5 裸执行、固定轴下探和固定左右手分配比较；
- 在 held-out 自然状态上成功率不得低于真实 chunk 基线；
- 若动态重定时不能减少超力退出或提高有限 horizon 完成率，停止复杂重定时扩展，只保留最小安全投影器。

### L2：视触觉补偿与安全

- 粗视觉只限定接触搜索区域或提供可靠度，不要求毫米级完整姿态；
- 接触后的横向和旋转补偿必须来自腕力、指尖力、动作响应和有界微探测；
- 卡紧、滑移、抓持失稳和超时只由可部署传感与动作历史判断；
- 独立 evaluator 只在进程外评分，不能终止控制循环；
- 硬力门必须在所有测试中零漏检；
- 若完整补偿器不能优于最小安全投影器，停止复杂补偿扩展，只保留安全层。

### L3：局部自主终止

- 局部控制器必须仅用可部署信号输出完成、重规划或安全退出；
- evaluator 终止的 rollout 只计作几何可达性上界，不计作自主完成率；
- 成功检测误触发或漏检未达到预注册门槛时，不得报告自主完成率。

### I0：完整系统集成

- 从自然 `env.reset()` 到成功或安全退出连续运行；
- 不恢复 demo 状态，不冻结教师 handoff，不读取记录 action chunk；
- 输出完整 ego 视频、在线输入审计、峰值力、退出原因和 π0.5 请求次数；
- 该门只验证上游抓取、粗对准和接管接口能否与已通过 L0–L3 的局部控制器集成，不作为局部算法成立的前置条件。

# 第二部分：已停止的 IEKF-PBVS 条件性候选档案

以下内容只说明该候选在理想局部假设下的数学结构，**不再属于当前实现路线，不得用于声称项目可行，也不得继续消耗主线实验预算。** 保留这些公式仅用于解释停止前做过什么、哪些假设没有被真实感知系统满足。

## 4. 坐标系与未知量

定义坐标系：

- \(W\)：机器人世界坐标系；
- \(C\)：ego 相机坐标系；
- \(R\)：右手掌坐标系；
- \(L\)：左手掌坐标系；
- \(P\)：peg CAD 坐标系；
- \(H\)：孔座 CAD 坐标系。

机器人正运动学给出可测手掌姿态：

\[
T_W^R(q_t),\qquad T_W^L(q_t).
\]

小脑真正需要估计的是两个在手变换：

\[
T_R^P,\qquad T_L^H.
\]

由此得到世界系物体姿态：

\[
T_W^P=T_W^R T_R^P,
\qquad
T_W^H=T_W^L T_L^H.
\]

peg 相对孔座的核心控制状态为：

\[
T_H^P=(T_W^H)^{-1}T_W^P
=(T_W^L T_L^H)^{-1}(T_W^R T_R^P).
\]

这一等式是方法论的关键：手臂本体状态只提供手掌运动，真正的对孔误差还取决于 peg 和孔座是否在各自灵巧手内发生滑移。

若两侧在手姿态采用右乘小误差：

\[
T_R^P=\hat T_R^P\operatorname{Exp}(\delta\xi_R),
\qquad
T_L^H=\hat T_L^H\operatorname{Exp}(\delta\xi_L),
\]

则相对姿态满足：

\[
T_H^P
=\operatorname{Exp}(-\delta\xi_L)
\hat T_H^P
\operatorname{Exp}(\delta\xi_R).
\]

把左侧扰动转换成 \(\hat T_H^P\) 的右乘扰动，一阶近似得到：

\[
\delta\xi_{rel}
\approx
\delta\xi_R
-\operatorname{Ad}_{(\hat T_H^P)^{-1}}\delta\xi_L.
\]

因此相对姿态协方差不是两侧协方差的简单相加，而是：

\[
P_{rel}=A_{rel}PA_{rel}^T,
\qquad
A_{rel}=
\begin{bmatrix}
I_6 &
-\operatorname{Ad}_{(\hat T_H^P)^{-1}} &
0
\end{bmatrix}.
\]

## 5. IEKF 状态与误差定义

滤波状态定义为：

\[
X_t=
\left(
T_{R,t}^{P},
T_{L,t}^{H},
v_t^P,
v_t^H,
b_t^R,
b_t^L
\right),
\]

其中：

- \(T_R^P,T_L^H\) 是两侧在手姿态；
- \(v^P,v^H\in\mathbb R^6\) 是在手滑移 twist；
- \(b^R,b^L\) 是腕力或接触伪观测的慢变偏置。

SE(3) 状态不能直接对旋转矩阵做普通加法。采用右乘误差状态：

\[
T_R^P=\hat T_R^P\operatorname{Exp}(\delta\xi_R),
\qquad
T_L^H=\hat T_L^H\operatorname{Exp}(\delta\xi_L),
\]

其中 \(\delta\xi_R,\delta\xi_L\in\mathbb R^6\)。完整误差向量为：

\[
\delta x_t=
\begin{bmatrix}
\delta\xi_R^T &
\delta\xi_L^T &
\delta v_P^T &
\delta v_H^T &
\delta b_R^T &
\delta b_L^T
\end{bmatrix}^T.
\]

滤波器维护 \((\hat X_t,P_t)\)，其中 \(P_t\) 是误差状态协方差，不把它称为额外的“信念模块”。

## 6. 在手运动模型

在短时间内，物体相对手掌的运动由滑移 twist 驱动：

\[
T_{R,t+1}^{P}
=T_{R,t}^{P}\operatorname{Exp}(\Delta t\,v_t^P+w_t^P),
\]

\[
T_{L,t+1}^{H}
=T_{L,t}^{H}\operatorname{Exp}(\Delta t\,v_t^H+w_t^H).
\]

滑移速度采用随机游走：

\[
v_{t+1}^P=v_t^P+w_t^{vP},
\qquad
v_{t+1}^H=v_t^H+w_t^{vH}.
\]

误差状态预测为：

\[
\delta x_{t+1}=F_t\delta x_t+G_tw_t,
\]

\[
P_{t+1}^{-}=F_tP_t^{+}F_t^T+G_tQ_tG_t^T.
\]

过程噪声 \(Q_t\) 由抓持状态调节：

\[
Q_t^P=
\begin{cases}
Q_{stick}, & \text{多指稳定接触且无滑移证据},\\
Q_{slip}, & \text{切向力、接触变化或图像残差提示滑移},
\end{cases}
\qquad Q_{slip}\succ Q_{stick}.
\]

因此滤波器不会永久假设“物体与手掌刚性固定”，这是灵巧手场景与普通腕部 PBVS 的主要区别。

## 7. Ego 视觉观测模型

对 peg 和孔座分别定义固定 CAD 关键点：

\[
\{p_j^P\}_{j=1}^{N_P},
\qquad
\{p_k^H\}_{k=1}^{N_H}.
\]

视觉网络从 ego RGB 输出二维关键点、轮廓和置信度：

\[
z_t^{vis}=
\left(
\hat u_{j,t}^P,
\hat u_{k,t}^H,
\gamma_{j,t}^P,
\gamma_{k,t}^H
\right).
\]

预测像素为：

\[
h_{j}^{P}(X_t,q_t)
=\pi\!\left(
T_C^W T_W^R(q_t)T_R^P p_j^P
\right),
\]

\[
h_{k}^{H}(X_t,q_t)
=\pi\!\left(
T_C^W T_W^L(q_t)T_L^H p_k^H
\right).
\]

视觉残差：

\[
r_t^{vis}=z_t^{vis}-h^{vis}(\hat X_t^-,q_t).
\]

以 peg 关键点为例，令：

\[
y_j^C
=T_C^WT_W^R\hat T_R^Pp_j^P.
\]

对右乘误差 \(\delta\xi_R=[\delta\rho_R^T,\delta\phi_R^T]^T\) 一阶展开：

\[
\delta y_j^C
=R_C^{RP}
\begin{bmatrix}
I_3 & -[p_j^P]_\times
\end{bmatrix}
\delta\xi_R,
\]

其中 \(R_C^{RP}\) 是从 peg 局部扰动到相机坐标系的旋转部分。于是视觉观测 Jacobian 为：

\[
H_{j,R}^{vis}
=J_\pi(y_j^C)
R_C^{RP}
\begin{bmatrix}
I_3 & -[p_j^P]_\times
\end{bmatrix}.
\]

孔座关键点对 \(\delta\xi_L\) 的 Jacobian 同理。这个 Jacobian 必须由解析式或自动微分验证，不能用 MuJoCo 真实物体位姿在线构造。

视觉噪声按置信度和遮挡动态调节：

\[
R_{j,t}^{vis}
=\frac{R_0^{vis}}{\max(\gamma_{j,t},\epsilon)}
+R_{occ}(o_{j,t}).
\]

低置信度观测只降低权重；只有几何一致性检验失败时才拒绝更新，不能把错误检测强行写入滤波状态。

## 8. 灵巧手运动学与接触观测

### 8.1 指尖几何约束

由手部正运动学得到第 \(i\) 个指尖世界位置：

\[
p_{F_i}^W=p_{F_i}^W(q_t).
\]

若该指尖与 peg 稳定接触，则转换到 peg 坐标系后应位于 CAD 表面：

\[
r_i^{surf}
=\phi_P\!\left(
(T_W^P)^{-1}p_{F_i}^W
\right)=0,
\]

其中 \(\phi_P\) 是 peg CAD 的有符号距离函数。孔座侧同理：

\[
r_i^{surf,H}
=\phi_H\!\left(
(T_W^H)^{-1}p_{F_i}^W
\right)=0.
\]

接触法向可靠时增加法向一致性残差：

\[
r_i^{normal}
=1-n_{F_i}^{W\,T}R_W^Pn_i^P.
\]

### 8.2 无滑移速度约束

稳定粘着接触时，指尖接触点和物体表面点速度近似一致：

\[
r_i^{stick}
=J_{F_i}(q_t)\dot q_t
-J_{P,i}(\hat X_t) V_P
\approx0.
\]

若切向力比值、指尖外力变化或视觉残差显示滑移，则不再施加强粘着约束，而是：

- 增大对应观测噪声；
- 增大在手滑移过程噪声；
- 允许 \(T_R^P\) 或 \(T_L^H\) 在线变化。

### 8.3 多指互补与姿态可观测性

对当前接触物体的 \(m\) 个有效指尖，将各指尖观测 Jacobian 堆叠：

\[
H_{hand}=
\begin{bmatrix}
H_1^T&\cdots&H_m^T
\end{bmatrix}^T,
\qquad
\mathcal I_{hand}
=\sum_{i=1}^{m}H_i^TR_i^{-1}H_i.
\]

\(\mathcal I_{hand}\) 是多指接触对在手姿态提供的信息矩阵。单指通常只能约束局部表面距离；分布在不同位置、法向不同的多个指尖可以提高任务相关自由度的秩，并在视觉遮挡时继续约束物体位置、方向和滑移。

多指接触运动学写为：

\[
v_c=J_h(q)\dot q_h-G^TV_o,
\]

其中 \(J_h\) 将手指关节速度映射到所有接触点速度，\(G\) 是抓持映射，\(G^TV_o\) 是物体运动在各接触点产生的速度。粘着接触要求 \(v_c\approx0\)；滑移时该残差用于提高 IEKF 过程噪声，而不是继续假设刚性抓持。

### 8.4 插孔接触约束

接近孔口后，腕力和多指力用于生成接触伪观测。对估计接触法向 \(n_c\)，无穿透条件为：

\[
n_c^T v_{rel,c}\ge -\epsilon_v.
\]

单边孔壁接触产生横向几何约束：

\[
r_c^{wall}=n_c^T(p_c^P-p_c^H)\approx0.
\]

这些约束只根据在线力和当前 CAD 估计生成，不能读取 MuJoCo 真实接触点。

## 9. IEKF 更新

将当前有效观测堆叠为：

\[
r_t=
\begin{bmatrix}
r_t^{vis}\\
r_t^{surf}\\
r_t^{normal}\\
r_t^{stick}\\
r_t^{wall}
\end{bmatrix},
\qquad
R_t=\operatorname{blkdiag}
\left(
R_t^{vis},R_t^{hand},R_t^{contact}
\right).
\]

在当前预测状态处对误差状态线性化：

\[
r_t\approx r_t^0-H_t\delta x_t+n_t.
\]

一次 EKF 更新为：

\[
S_t=H_tP_t^-H_t^T+R_t,
\]

\[
K_t=P_t^-H_t^TS_t^{-1},
\]

\[
\delta x_t=K_tr_t.
\]

SE(3) 状态通过指数映射注入：

\[
\hat T_R^{P,+}
=\hat T_R^{P,-}\operatorname{Exp}(\delta\xi_R),
\]

\[
\hat T_L^{H,+}
=\hat T_L^{H,-}\operatorname{Exp}(\delta\xi_L).
\]

协方差使用 Joseph 形式：

\[
P_t^+
=(I-K_tH_t)P_t^-(I-K_tH_t)^T
+K_tR_tK_t^T.
\]

由于视觉投影和接触几何是强非线性的，同一帧最多执行 \(N_{iter}\) 次重线性化：

```text
预测状态
→ 计算残差和 Jacobian
→ IEKF 更新
→ 在新状态重新计算残差
→ 收敛或达到迭代上限
```

如果归一化创新平方超过卡方门限：

\[
\operatorname{NIS}=r_t^TS_t^{-1}r_t>\tau_{\chi^2},
\]

则拒绝该观测或提高其噪声，不能通过控制动作去“解释掉”错误视觉估计。

## 10. 从滤波状态得到 PBVS 误差

滤波后相对姿态为：

\[
\hat T_H^P
=(T_W^L\hat T_L^H)^{-1}
(T_W^R\hat T_R^P).
\]

定义当前模式下的目标相对姿态 \(T_{H,m}^{P,*}\)。PBVS 位姿误差为：

\[
e_t
=\operatorname{Log}
\left(
(T_{H,m}^{P,*})^{-1}\hat T_H^P
\right)
=
\begin{bmatrix}
e_p\\e_R
\end{bmatrix}
\in\mathbb R^6.
\]

圆柱插孔绕轴旋转不可观测且不影响任务时，使用选择矩阵 \(S_m\) 去掉该自由度：

\[
\bar e_t=S_me_t.
\]

SEARCH、ALIGN 和 INSERT 使用同一个相对姿态，只改变目标和权重，不建立三套互相矛盾的状态表示。

## 11. 双臂多指 PBVS 模型

机器人运动控制变量为：

\[
u_t=
\begin{bmatrix}
\dot q_{arm,R}\\
\dot q_{arm,L}\\
\dot q_{hand,R}\\
\dot q_{hand,L}
\end{bmatrix}.
\]

忽略二阶项时，相对位姿误差动力学为：

\[
\dot{\bar e}_t
=J_{rel}(\hat X_t,q_t)u_t+d_t^{slip},
\]

其中：

\[
J_{rel}
=S_mL_e
\begin{bmatrix}
J_P^R & -J_H^L & J_P^{hand,R} & -J_H^{hand,L}
\end{bmatrix}.
\]

普通机械臂 PBVS 只包含前两个腕部 Jacobian；本项目显式保留 \(J_P^{hand,R}\) 和 \(J_H^{hand,L}\)，使多指抓持调整可以参与相对位姿控制。

多指接触力满足：

\[
w_o=Gf_c,
\qquad
f_c=f_{motion}+N_G\lambda,
\qquad
GN_G=0.
\]

其中 \(N_G\lambda\) 是不改变物体合力的内部抓持力。优化变量扩展为：

\[
z_t=
\begin{bmatrix}
u_t\\\lambda_t
\end{bmatrix}.
\]

这使灵巧手具有两个独立作用：手指运动改变在手物体姿态，内部力维持抓持和摩擦裕度；两者不能被一个固定闭合手型替代。

每个控制周期求解约束二次规划：

\[
z_t^*=\arg\min_{u,\lambda}
\frac12
\left\|
J_{rel}u+\Lambda_m\bar e_t
\right\|_{Q_m}^2
+\frac12\|u\|_{R_u}^2
+\frac12\|J_h\dot q_h-G^TV_o\|_{W_c}^2
+\frac12\|\lambda-\lambda^*\|_{W_\lambda}^2.
\]

其中：

- 第一项实现 PBVS 误差下降；
- 第二项限制动作幅值与关节抖动；
- 第三项保持多指接触运动一致并限制在手滑移；
- 第四项利用抓持映射零空间调节内部力，保持摩擦裕度但不干扰 PBVS 物体运动。

约束包括：

\[
\dot q_{min}\le u\le\dot q_{max},
\]

\[
q_{min}\le q_t+\Delta t\,u\le q_{max},
\]

\[
f_n\ge f_{min},
\qquad
\|f_t\|\le\mu f_n,
\]

\[
\|w_{internal}\|\le w_{max},
\]

以及孔壁非穿透和各模式特定的轴向速度约束。

## 12. IEKF 与 PBVS 如何联合

本方法不把 PBVS 目标加入 IEKF 观测损失，因为这样会把“希望 peg 在孔中”错误地当成“测量证明 peg 已经在孔中”。正确联合方式是共享状态、闭环交替：

```text
1. IEKF 根据上一步动作预测在手姿态
2. ego、手指和力观测更新在手姿态
3. 由更新后的相对位姿构造 PBVS 误差
4. 约束优化器输出一个短控制动作
5. 新动作改变视角、接触和相对构型
6. 下一帧重新进入 IEKF
```

联合发生在控制器对“下一帧估计质量”的预测中，而不是修改 IEKF 的状态均值。令

\[
A_t=S_mA_{rel,t},
\qquad
\Sigma_t=A_tP_t^+A_t^T
\]

为受控对孔误差协方差。给定候选手臂/手指速度 \(u\) 和内部力参数 \(\lambda\)，下一帧协方差采用局部信息形式预测：

\[
P_{t+1}^-(u,\lambda)
=F_t(u)P_t^+F_t(u)^T+G_tQ_t(u,\lambda)G_t^T,
\]

\[
P_{t+1}^+(u,\lambda)
=\left[
\left(P_{t+1}^-(u,\lambda)\right)^{-1}
+\mathcal I_{vis}(u)
+\sum_{i=1}^{m}a_i(u)H_i(u)^TR_i^{-1}H_i(u)
\right]^{-1},
\]

\[
\Sigma_{t+1}(u,\lambda)
=A_{t+1}P_{t+1}^+(u,\lambda)A_{t+1}^T.
\]

其中所有 Jacobian 均在预测构型 \(q_{t+1}=q_t+\Delta t\,u\) 处计算，\(a_i\in[0,1]\) 是第 \(i\) 指接触可信度。手指运动通过 \(H_i(u)\) 和 \(a_i(u)\) 改变多指几何可观测性；内部力通过摩擦裕度和滑移概率改变 \(Q_t(u,\lambda)\)。\(Q_t(u,\lambda)\) 必须来自离线标定的滑移模型，不能由优化器任意缩小。因此灵巧手同时进入观测信息项和过程噪声项，但不能通过虚报低噪声获得虚假置信度。

其中 \(\mathcal I_{vis}(u)=H_{vis}(u)^TR_{vis}^{-1}H_{vis}(u)\)。忽略约束激活时，下一周期 PBVS 使用 \(u_{t+1}=-J_{rel}^{\dagger}\Lambda_m\hat e_{t+1}\)。在条件零均值估计误差下，其对下一步真实误差造成的二次代价为：

\[
\mathbb E\!\left[
\|\Delta t\,\Lambda_m\tilde e_{t+1}\|_{Q_m}^2
\right]
=\operatorname{tr}
\left(W_P\Sigma_{t+1}(u,\lambda)\right),
\qquad
W_P=\Delta t^2\Lambda_m^TQ_m\Lambda_m.
\]

由此得到统一的信念空间控制目标：

\[
\min_{u,\lambda}
\frac12\|\bar e_t+\Delta t J_{rel}u\|_{Q_m}^2
+\frac{\beta}{2}\operatorname{tr}
\left(W_P\Sigma_{t+1}(u,\lambda)\right)
+J_{grasp}(u,\lambda),
\]

其中 \(J_{grasp}\) 包含第 11 节的动作正则、接触速度一致性和内部力代价。第一项减小当前对孔误差，第二项减小会在下一次闭环控制中放大的任务相关估计误差，第三项抑制滑移并保持摩擦裕度。

为避免“均值下降但真实姿态可能变差”，控制器同时施加置信椭球鲁棒下降约束。令 \(e_t^{true}\) 为受控自由度上的真实误差，\(\tilde e_t=\bar e_t-e_t^{true}\)。当 IEKF 一致且 \(\Sigma_t\succ0\) 时，以概率至少 \(1-\alpha\) 有

\[
\tilde e_t^T\Sigma_t^{-1}\tilde e_t\le\chi_{r,1-\alpha}^2,
\qquad
\|d_t^{slip}\|\le\bar d_s,
\]

其中 \(r=\operatorname{rank}(S_m)\)。对 \(V=\frac12e^TQ_me\)，有最坏情况上界：

\[
\dot V
\le
\bar e_t^TQ_mJ_{rel}u
+\sqrt{\chi_{r,1-\alpha}^2
(Q_mJ_{rel}u)^T\Sigma_t(Q_mJ_{rel}u)}
+\bar d_s\left(\|Q_m\bar e_t\|+\rho_{\Sigma}\right),
\]

\[
\rho_{\Sigma}
=\sqrt{\chi_{r,1-\alpha}^2
\lambda_{max}\!\left(\Sigma_t^{1/2}Q_m^2\Sigma_t^{1/2}\right)}.
\]

统一优化还需满足第 11 节的运动学、摩擦和非穿透约束，以及：

\[
\bar e_t^TQ_mJ_{rel}u
+\sqrt{\chi_{r,1-\alpha}^2
(Q_mJ_{rel}u)^T\Sigma_t(Q_mJ_{rel}u)}
+\bar d_s\left(\|Q_m\bar e_t\|+\rho_{\Sigma}\right)
\le-\kappa\|\bar e_t\|_{Q_m}^2.
\]

在固定线性化点上，该鲁棒下降条件是二阶锥约束；若约束不可行或 \(\Sigma_t\) 超过模式门限，则禁止继续 INSERT，转入 SEARCH、ALIGN 或 CONTACT_RECOVERY。协方差预测关于 \(u,\lambda\) 一般非凸，在线采用候选动作评估或逐次凸化，而不把它错误地称为单次标准 QP。

## 13. SEARCH、ALIGN、INSERT 的统一模式

### 13.1 SEARCH

进入条件：孔关键点不足、NIS 持续异常、相对姿态协方差过大或严重遮挡。

SEARCH 目标：

\[
\min_{u,\lambda}
\lambda_P\operatorname{tr}(W_P\Sigma_{t+1}(u,\lambda))
+\lambda_o C_{occ}(u)
+\lambda_u\|u\|^2.
\]

候选动作是厘米级安全宏动作，例如抬高 peg、横向改变视角、轻微改变孔座朝向；SEARCH 禁止持续沿孔轴下压。

### 13.2 ALIGN

目标姿态是孔口上方的预插入位姿：

\[
T_{H,align}^{P,*}
=T_{H,entry}^{P}\operatorname{Trans}(0,0,d_{safe}).
\]

ALIGN 对横向位置和轴线方向赋高权重，对插入深度赋低权重。连续 \(N_{align}\) 帧满足误差、NIS 和协方差门槛后进入 INSERT。

### 13.3 INSERT

INSERT 目标沿孔轴递增：

\[
d_{t+1}^*=d_t^*+\Delta d,
\]

同时保持：

\[
\|e_{xy}\|\le\epsilon_{xy},
\qquad
\|e_{axis}\|\le\epsilon_{axis}.
\]

轴向速度受接触门控：

\[
0\le v_{ins}\le
v_{max}\alpha_{vis}\alpha_{contact}\alpha_{grasp}.
\]

任何一项可靠度下降时，插入速度自动降为零并返回 ALIGN 或 CONTACT_RECOVERY。

### 13.4 CONTACT_RECOVERY

接触残差和力方向区分：

- 单边接触：保持轻微轴向预载，执行切向 PBVS 修正；
- 双边卡紧：停止下压，沿孔轴退回，再更新 IEKF；
- 抓持滑移：提高过程噪声，优先重新估计在手姿态；
- 目标丢失：退出接触并返回 SEARCH；
- 力超过硬阈值：立即安全退出。

## 14. 局部稳定性推导

在目标附近，真实 PBVS 误差满足：

\[
\dot e=J_{rel}u+d^{slip}.
\]

设滤波误差为：

\[
\tilde e=\hat e-e.
\]

忽略约束激活时，PBVS 控制律为：

\[
u=-J_{rel}^{\dagger}\Lambda\hat e.
\]

代入得到：

\[
\dot e
=-\Lambda e-\Lambda\tilde e+d^{slip}.
\]

取 Lyapunov 函数：

\[
V(e)=\frac12e^TQe,
\qquad Q\succ0.
\]

则：

\[
\dot V
\le
-\alpha\|e\|^2
+\beta\|e\|\|\tilde e\|
+\gamma\|e\|\|d^{slip}\|.
\]

因此，只要：

1. 相对 Jacobian 在受控自由度上满秩；
2. IEKF 误差有界并能在持续观测下收敛；
3. 滑移扰动有界；
4. PBVS 增益不超过执行器和接触稳定范围；

则闭环误差收敛到一个由估计误差和滑移上界决定的邻域：

\[
\limsup_{t\to\infty}\|e_t\|
\le
c_1\sup_t\|\tilde e_t\|
+c_2\sup_t\|d_t^{slip}\|.
\]

这说明本项目必须同时提高在手姿态估计和接触稳定性；仅提高 PBVS 增益不能消除估计偏差，反而可能导致卡紧。

## 15. 闭环命题与证明边界

### 15.1 命题 A：完美姿态下的局部 PBVS 收敛

在目标邻域内，假设：

1. 受控自由度上的相对 Jacobian \(J_{rel}\) 满行秩；
2. QP 约束未激活；
3. 使用真实相对误差 \(e\)；
4. 不存在滑移和未建模扰动。

采用控制律：

\[
u=-J_{rel}^{\dagger}\Lambda e,
\qquad \Lambda\succ0.
\]

由于满行秩时 \(J_{rel}J_{rel}^{\dagger}=I\)，得到：

\[
\dot e=-\Lambda e.
\]

因此：

\[
e(t)=\operatorname{Exp}(-\Lambda t)e(0),
\]

目标姿态在该局部模型下指数稳定。这个命题只证明控制器在完美姿态输入下理论可行，不证明视觉估计可用。

### 15.2 命题 B：有界估计误差下的实际稳定

若：

\[
\|\tilde e_t\|\le\epsilon_e,
\qquad
\|d_t^{slip}\|\le\epsilon_s,
\]

由上一节的 Lyapunov 不等式：

\[
\dot V
\le
-\alpha\|e\|^2
+\beta\epsilon_e\|e\|
+\gamma\epsilon_s\|e\|.
\]

当：

\[
\|e\|>
r
=\frac{\beta\epsilon_e+\gamma\epsilon_s}{\alpha},
\]

有 \(\dot V<0\)。因此闭环误差最终进入并保持在半径 \(r\) 的邻域内：

\[
\limsup_{t\to\infty}\|e_t\|\le r.
\]

这给出可直接实验检验的预测：向姿态输入注入不同幅值的有界偏差时，最终 PBVS 对孔误差应近似随偏差上界线性增长。

### 15.3 命题 C：IEKF-PBVS 级联系统

若 IEKF 在局部可观测条件下满足：

\[
\|\tilde e(t)\|
\le
M\exp(-\lambda_f t)\|\tilde e(0)\|
+c_n\bar n,
\]

其中 \(\bar n\) 是视觉、触觉和过程噪声上界；同时 PBVS 对估计误差和滑移扰动是输入到状态稳定的，则：

- 无噪声、无持续滑移时，\(\tilde e\to0\) 且 \(e\to0\)；
- 有界噪声和滑移时，\(e\) 收敛到由 \(\bar n\) 与 \(\epsilon_s\) 决定的邻域；
- 持续系统偏差未被创新门控发现时，PBVS 会收敛到错误位置。

因此完整方法必须同时验证滤波一致性和控制 ISS，不能只展示一次成功插入。

### 15.4 当前没有证明的内容

当前理论不声称：

- 从任意错误初始化全局收敛；
- 在任意遮挡下 IEKF 一定收敛；
- QP 长期饱和时仍保持指数稳定；
- SEARCH、ALIGN、INSERT 和 CONTACT_RECOVERY 任意切换时存在统一全局 Lyapunov 函数；
- 系统性视觉偏差不会误导 PBVS。

模式切换后的严格证明需要公共 Lyapunov 函数或驻留时间条件；第一阶段只要求每个模式局部稳定、切换门具有滞回，并通过端到端实验排除频繁振荡。

### 15.5 可执行理论验证

解析推导必须同时通过数值单元测试，避免 SE(3) 左右扰动、Adjoint 和 PBVS 符号在实现中写反。当前验证入口为：

- `retrieval_cerebellum/theory_validation.py`；
- `retrieval_cerebellum/tests/test_theory_validation.py`。

测试覆盖：

1. SE(3) 指数映射与对数映射互逆；
2. 双侧在手右乘误差到相对姿态误差的解析 Jacobian 与有限差分一致；
3. 完美姿态输入下 PBVS 误差单调收敛；
4. 常值估计偏差和滑移扰动下，数值平衡点满足：

\[
e_{\infty}
=-\tilde e
+\Lambda^{-1}d^{slip}.
\]

这些测试只验证局部线性理论和实现符号，不替代 O0 物理仿真、O1 非线性误差注入或 O3 正式视觉闭环。

## 16. Oracle 分阶段验证协议

当前可以用真实姿态替代视觉估计，但只能用于隔离验证控制层，必须明确标记为 Oracle，不能作为方法主结果。

### O0：完美状态 PBVS

由独立 truth adapter 输出：

\[
\hat T_H^P=T_{H,true}^P,
\qquad
P_{rel}=0.
\]

控制器只能读取统一的 `AssemblyStateEstimate` 接口，不能直接导入 MuJoCo model/data。O0 用于检验：

- 相对坐标系和 Jacobian 符号是否正确；
- 双臂 PBVS-QP 是否降低真实对孔误差；
- 完美姿态下是否能完成 DexJoCo 插入；
- 关节、接触和安全约束是否可执行。

现有 `hybrid_insert` 可以作为近似 O0 上界，因为它使用真实几何做 PBVS；但它内部直接读取 privileged geometry、包含专用 handoff 和 release 逻辑，不等价于新 IEKF-PBVS 控制器，只能证明任务在完美状态反馈下存在可行闭环。

### O1：受控误差注入

在真实姿态上注入已知误差：

\[
\hat T_H^P
=T_{H,true}^P
\operatorname{Exp}(\delta\xi),
\]

并分别测试：

- 常值位置和角度偏差；
- 零均值噪声；
- 低频漂移；
- 短时观测丢失；
- 协方差低估和高估。

O1 必须验证命题 B：最终对孔误差是否随注入误差上界增长，协方差门控是否能在误差过大时阻止继续下插。

### O2：离线 IEKF

使用记录的 ego、本体和力信号驱动 IEKF，但动作只作为已执行输入，不从记录后续动作生成控制。真值只在进程外评估：

- 在手姿态误差；
- 相对孔轴误差；
- NIS 和协方差一致性；
- 遮挡与滑移恢复时间。

### O3：在线 IEKF-PBVS

将 O0/O1 的 truth adapter 替换为在线 IEKF，控制器接口、PBVS-QP 和安全门保持不变。从 `env.reset()` 开始完成正式无 Oracle 评测。

通过顺序必须是：

```text
O0 完美状态控制可行
→ O1 有界误差鲁棒性符合理论
→ O2 IEKF 估计一致性
→ O3 在线完整闭环
```

O0 成功不能证明 IEKF 可行；O2 估计准确也不能证明闭环插入成功；只有 O3 可以作为最终方法结果。

## 17. 可观测性要求

系统不能在所有几何条件下恢复完整六自由度姿态。正式实现必须检查局部可观测性矩阵：

\[
\mathcal O_t=
\begin{bmatrix}
H_t\\
H_{t+1}F_t\\
\cdots\\
H_{t+k}F_{t+k-1}\cdots F_t
\end{bmatrix}.
\]

当 \(\mathcal O_t\) 在任务相关自由度上秩不足时：

- 对称圆柱的绕轴旋转不进入控制误差；
- SEARCH 主动改变视角或接触构型；
- 不允许以低置信姿态直接进入 INSERT；
- 若长期不可观测，则明确退出而不是使用真值补全。

## 18. 自然端到端实验协议

每次正式实验必须：

1. 从 `env.reset()` 开始；
2. 在线启动 π0.5 服务；
3. π0.5 完成自然抓取和粗搬运；
4. 小脑根据在线 ego、手部和力信号自行接管；
5. 完整运行 SEARCH、ALIGN、INSERT 和必要的 CONTACT_RECOVERY；
6. 只由 DexJoCo 原生 evaluator 在进程外判定成功；
7. 输出单一完整 `ego.mp4` 和在线输入审计日志；
8. episode 结束后立即关闭 π0.5 服务。

禁止：

- demo 回放到某一帧；
- 恢复教师初态；
- 用真实姿态触发 handoff；
- 使用 `hybrid_insert` 生成主结果；
- 截掉失败前半段或只展示孔口附近片段。

## 19. 验收门槛

### P0：防火墙

- 控制器依赖扫描中不含 privileged geometry、evaluator 或 demo loader；
- 运行日志声明所有在线输入来源；
- 独立测试证明真值进程不能向控制器返回数据。

### P1：IEKF 在手姿态跟踪

- 从在线 ego、手部和力信号估计 \(T_R^P,T_L^H\)；
- 与独立真值比较位置、轴向和相对位姿误差；
- 报告 NIS、一致性、遮挡和人工滑移下的恢复能力；
- 必须优于“物体刚性固定在手掌”基线。

### P2：IEKF-PBVS 对孔

- 从 π0.5 自然粗搬运状态自行接管；
- 不使用 demo handoff 或真实姿态；
- 展示 SEARCH 后相对误差下降并进入 ALIGN；
- 必须优于“单帧视觉估计 + PBVS”基线。

### P3：完整插入

- 从 `env.reset()` 连续运行到 DexJoCo `succeed`；
- 视频完整包含抓取、搜孔、对孔和插入；
- 报告成功率、安全退出率、峰值力和接触恢复次数；
- 主结果不得来自 privileged `hybrid_insert`。

## 20. 必需消融

必须至少比较：

1. π0.5 单独执行；
2. 单帧 ego 姿态估计 + PBVS；
3. 假设物体刚性固定于手掌的 EKF + PBVS；
4. ego IEKF + PBVS；
5. ego + 多指接触 IEKF + 腕部 PBVS；
6. 完整双臂多指 IEKF-PBVS；
7. privileged geometry 上界，只作为 Oracle，不进入主表方法列。

关键消融问题：

- 多指接触是否改善遮挡和滑移下的姿态跟踪；
- IEKF 是否比单帧估计降低对孔抖动和失败率；
- 主动 SEARCH 是否改善可观测性；
- 手指自由度是否比纯双腕控制更能保持在手姿态；
- 接触更新是否减少卡边和重复下压。

## 21. 实现顺序

按以下顺序实现，不允许跳过理论和估计直接调插入：

1. 定义坐标系、SE(3) 状态、误差状态和数值 Jacobian 测试；
2. 使用 truth adapter 完成 O0 完美状态 PBVS-QP；
3. 完成 O1 偏差、噪声、漂移和丢帧注入；
4. 实现只含 ego CAD 关键点的 IEKF；
5. 加入手掌和指尖运动学观测；
6. 加入滑移检测与自适应过程噪声；
7. 完成 O2 离线 IEKF 一致性评估；
8. 将 O0 truth adapter 无缝替换为 IEKF；
9. 加入多指接触保持项；
10. 加入 SEARCH 和可观测性门控；
11. 加入插孔接触更新与 CONTACT_RECOVERY；
12. 从 `env.reset()` 运行无 demo、无 Oracle 的完整 ego 视频。

在 P1 估计一致性和 P2 自然闭环对孔通过前，不允许再次用 privileged controller 生成“方法交付视频”。
