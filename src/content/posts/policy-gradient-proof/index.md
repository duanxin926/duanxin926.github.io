---
title: 策略梯度定理推导
description: 从策略梯度到PPO算法的详细推导
date: 2025-09-08
tags:
  - Reinforcement Learning
categories:
  - 学习笔记
# image: "[[cover.png]]"
# imageAlt: A beautiful multi-tiered waterfall cascading into a turquoise lake surrounded by lush green forest.
imageOG: false
hideCoverImage: false
hideTOC: false
targetKeyword: ""
draft: false
---
## 1. Policy Gradient 推导
首先我们的**目标是最大化累计奖励期望**，期望是当前策略下，每一条轨迹的概率乘上每条轨迹的奖励。
为什么我们能对累计奖励期望这个常数求导？是因为我们假设策略是一个随机分布，在训练过程中，它的参数会不断变化，每条轨迹的概率发生变化，则累计奖励期望会随着改参数的变化而变化。我们实际是对这个参数求导。那么自然就想到如果能找出梯度上升的方向，就能找到最大化累计奖励期望的策略分布参数。

那么我们首先写出累计奖励回报的公式：
$$
J(\theta) = E_{\tau \sim p_\theta(\tau)} [R(\tau)]
$$

其中：
*   $\theta$ 是策略网络 $\pi_\theta(a|s)$ 的参数。
*   $\tau$ 代表一个完整的轨迹（trajectory），即一连串的状态和动作序列：$\tau = (s_0, a_0, s_1, a_1, \dots, s_T, a_T)$。
*   $R(\tau) = \sum_{t=0}^{T} r(s_t, a_t)$ 是这条轨迹的总回报。
*   $p_\theta(\tau)$ 是在策略 $\pi_\theta$ 下，这条轨迹发生的概率。

对累计奖励回报求导：
$$
\nabla_\theta J(\theta) = \nabla_\theta E_{\tau \sim p_\theta(\tau)} [R(\tau)]
$$
我们可以将期望写成积分（或求和）的形式，根据莱布尼兹法则可以把求导符号移到积分内部：    
$$
\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau) R(\tau) d\tau=  \int \nabla_\theta (p_\theta(\tau) R(\tau) )d\tau = \int \nabla_\theta p_\theta(\tau) R(\tau) d\tau 
$$

对于任意可微函数 $f(x)$，我们有：$\nabla_x \log f(x) = \frac{\nabla_x f(x)}{f(x)}$
移项可得：$\nabla_x f(x) = f(x) \nabla_x \log f(x)$
我们将这个技巧应用到我们的轨迹概率 $p_\theta(\tau)$上：
$$
\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)
$$

代入原式：
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \int \left( p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) \right) R(\tau) d\tau \\
&= \int p_\theta(\tau) \left( \nabla_\theta \log p_\theta(\tau) R(\tau) \right) d\tau
\end{aligned}
$$
此时我们发现可以将改公式写回期望的形式，于是我们就可以通过采样估计期望：
$$
\nabla_\theta J(\theta) = E_{\tau \sim p_\theta(\tau)} [\nabla_\theta \log p_\theta(\tau) R(\tau)]
$$
我们只需要从当前策略 $\pi_\theta$ 采样一批轨迹 $\tau_i$，然后计算 $\nabla_\theta \log p_\theta(\tau_i) R(\tau_i)$ 的平均值即可近似得到梯度。

现在我们来求$\nabla_\theta \log p_\theta(\tau)$.
由于$p_\theta(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)$，取对数后，连乘变成连加：$\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T-1} \left( \log \pi_\theta(a_t|s_t) + \log p(s_{t+1}|s_t, a_t) \right)$。
则对 $\theta$ 求梯度：$\nabla_\theta \log p_\theta(\tau) = \nabla_\theta \left( \log p(s_0) + \sum_{t=0}^{T-1} (\log \pi_\theta(a_t|s_t) + \log p(s_{t+1}|s_t, a_t)) \right)$

非常重要的一点是，环境动力学 $p(s_{t+1}|s_t, a_t)$ 和初始状态分布 $p(s_0)$ 都不依赖于我们的策略参数 $\theta$。所以它们的梯度为零。
$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$
**于是对轨迹求导变成了对策略求导**。

再次代入期望公式中：
$$
\nabla_\theta J(\theta) = E_{\tau \sim p_\theta(\tau)} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \right) R(\tau) \right]
$$
其中 $R(\tau) = \sum_{t=0}^{T} r_t$。这个公式就是 Policy Gradient 定理的一种形式。

## 2. 为什么轨迹的奖励回报可以替换为 未来累计奖励折扣回报？
$R(\tau)$ 如何转变为 $\sum_{t=0}^{T} \gamma^t r_t$?
本质是我们**定义了更好的求解目标**。

在评估当前动作奖励时，用未来的累计奖励回报比用从 t=0 时刻开始的所有奖励回报要合理，因为过去时刻获得的奖励回报已经拿到了，与当前动作的选择无关。

对于 Policy Gradient，我们可以选择折扣奖励回报期望 $G_t$ 作为更合理的优化目标。主要是三个原因：
- 便于处理无限长度的任务，总回报可以收敛，而不是无限增长。
- 降低方差，强调即时奖励。
- 符合数学基础，贝尔曼方程依赖折扣因子，马尔科夫性也说明未来状态只与当前状态有关。

那么重新回顾策略梯度的公式：
$$
\nabla_\theta J(\theta) = E_{\tau \sim p_\theta(\tau)} [ \nabla_\theta \log p_\theta(\tau) \cdot \text{SomeValue}(\tau) ]
$$

*   **对于无折扣目标**，$\text{SomeValue}(\tau) = R(\tau) = \sum r_t$。
    $$
    \nabla_\theta J_{\text{undiscounted}}(\theta) = E_{\tau \sim p_\theta(\tau)} \left[ \nabla_\theta \log p_\theta(\tau) \left( \sum_{t=0}^{T} r_t \right) \right]
    $$

*   **对于折扣目标**，$\text{SomeValue}(\tau) = G_0 = \sum \gamma^t r_t$。
    $$
    \nabla_\theta J_{\text{discounted}}(\theta) = E_{\tau \sim p_\theta(\tau)} \left[ \nabla_\theta \log p_\theta(\tau) \left( \sum_{t=0}^{T} \gamma^t r_t \right) \right]
    $$

## 3. 为什么可以减去一个 baseline？即为什么可以用 Reward-to-Go 代替 Gt？
在我们已经确立了求解目标的基础上，我们希望优化求解过程。

现在的 $E_{\tau \sim p_\theta(\tau)} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \right) G_t \right]$ 是通过采样一整条轨迹得到的无偏估计，面临的问题就是方差过大，更新不稳定。

我们希望能够找到方差更小的无偏估计，一个很简单的方法就是降低 $G_t$ 的绝对值大小，即减去一个 baseline。这和采用 Reward-to-go 代替整条轨迹的奖励回报的推理是同样的思路。

首先我们**直观理解**为什么可以用 Reward-to-go 代替 Gt？
**因为一个在时间点  做出的动作 ，不可能影响它之前的奖励（$r_0, r_1, \dots, r_{t-1}$）。**
在策略梯度中加入过去的奖励，对期望值的计算没有贡献，但会给单词采样的梯度估计增加不必要的噪声，从而增大方差。

### 数学证明
我们要证明，用  替换 (即 $R(\tau)$) 是无偏的。这意味着：
$$
E_{\pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t \right] = E_{\pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) G_0 \right]
$$
等价于证明它们差值的期望为零：
$$
E_{\pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) (G_0 - G_t) \right] = \boldsymbol{0}
$$

令 $C_t = G_0 - G_t=\sum_{k=0}^{t-1} \gamma^k r_k$,我们来看求和中的任意一项 $E_{\pi_\theta} [ \nabla_\theta \log \pi_\theta(a_t|s_t) C_t ]$。

期望可以写成对状态和动作的积分（或求和）形式:
$$
E_{s \sim d^\pi, a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) C_t] = \int_s d^\pi(s) \int_a \pi_\theta(a|s) \left[ \nabla_\theta \log \pi_\theta(a|s) C_t \right] da ds
$$
因为$C_t$只依赖于状态 $s$，不依赖于动作 $a$，所以可以把它提出来：
    $$
    \int_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) C_t da = C_t \int_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) da
    $$
**再次使用 Log-Derivative Trick**
    我们知道 $\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$。代入上式：
    $$
    C_t \int_a \pi_\theta(a|s) \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} da = C_t \int_a \nabla_\theta \pi_\theta(a|s) da
    $$
由于积分域（所有动作空间）不依赖于 $\theta$，我们可以交换积分和梯度算子：
    $$
    =C_t \nabla_\theta \int_a \pi_\theta(a|s) da = C_t \nabla_\theta(1) = \boldsymbol{0}
    $$

**Baseline 的证明同理，只要和 a 或$\theta$没有关系，梯度都可以被提到积分号外面。**

> 很多教科书和原始论文在证明策略梯度定理时，推导出的就是使用**完整轨迹回报 $R(\tau)$** 的形式。reward-to-go 和 baseline 都可以看作后续的实践技巧和理论优化。

## 4. 如何选择最优 Baseline？
即 Reward-to-go 怎么变成$A_t$?

既然任何只依赖于状态的 $b(s)$ 都可以，那么选择什么样的 $b(s)$ 最好呢？理论上可以证明，能够最大程度减小方差的最优基线是**状态价值函数 (State-Value Function)**：
$$
b(s_t) = V^{\pi_\theta}(s_t) = E[G_t | S_t=s_t]
$$
当我们用 $V(s_t)$作为基线时，我们得到的差值有了一个专门的名字：**优势函数 (Advantage Function)**。
$$
A(s_t, a_t) = G_t - V(s_t)
$$
（在实际应用中，通常用 $Q(s_t, a_t)$ 的估计来代替 $G_t$，即 $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$）。

优势函数 $A(s_t, a_t)$ 直观地衡量了在状态 $s_t$ 下，采取动作 $a_t$ 相对于采取平均动作要好多少。这正是我们进行策略更新最想要的、最纯净的信号。这也是所有**Actor-Critic**算法（如A2C, A3C）的核心思想。

综上所述，减去一个基线在数学上是无偏的，并且在实践中通过将回报中心化，显著降低了梯度方差，是Policy Gradient方法不可或缺的一项关键技术。

至此, 经过这四个步骤, 我们得到了策略梯度定理的一种常见形式
$$
\nabla_\theta J(\theta) = E_{\tau \sim p_\theta(\tau)} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \right) A(s_t, a_t)  \right]
$$

更多可参考[从策略梯度定理到 PPO Actor Loss](posts/from-policy-gradient-to-ppo)


## Reference
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)