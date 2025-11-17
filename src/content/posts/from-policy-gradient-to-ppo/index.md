---
title: 从策略梯度定理到 PPO Actor Loss
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

## 起点：策略梯度定理

[策略梯度定理](posts/policy-gradient-proof)告诉我们，目标函数 $J(\theta)$ （即策略 $\pi_\theta$ 的期望累积回报）对参数 $\theta$ 的梯度可以表示为：

$$
\nabla_\theta J(\theta) = E_{\tau \sim p_\theta(\tau)} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \right) A(s_t, a_t)  \right]
$$
*   $J(\theta) = E_{\tau \sim p_\theta(\tau)}[\sum_t r(s_t, a_t)]$ 是我们要最大化的目标——期望总回报。
*   $\nabla_\theta J(\theta)$ 是我们想要计算的梯度。有了这个梯度，我们就可以使用**梯度上升**来更新参数 $\theta$，从而让策略变得更好：$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$。
*   $E_{\tau \sim p_\theta(\tau)}[\dots]$ 表示这个期望是在当前策略 $\pi_\theta$ 所产生的所有可能轨迹（trajectory） $\tau = (s_0, a_0, s_1, a_1, \dots)$ 上计算的。
*   $\log \pi_\theta(a_t|s_t)$ 是在状态 $s_t$ 下，采取动作 $a_t$ 的对数概率。它的梯度 $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 指向了能让 $(s_t, a_t)$ 这个行为出现概率**增加**最快的参数更新方向。这个技巧通常被称为 "log-derivative trick"。
*   $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$ 是**优势函数**。


直接计算上面公式中的期望 $E[\dots]$ 是不可能的，因为它需要遍历所有可能的轨迹。在实践中，我们采用**采样**（Sampling）和**近似**（Approximation）的方法。

---
## 从策略梯度定理到 PPO Actor Loss

### 第 1 步：用蒙特卡洛采样近似期望

我们通过让智能体（Agent）与环境互动，收集一批（a batch of）轨迹。假设我们收集了 $N$ 条轨迹 $\{\tau_1, \tau_2, \dots, \tau_N\}$。我们可以用这批样本的平均值来近似期望：
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left[ \left( \sum_{t=0}^{T_i-1} \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \right) A(s_{i,t}, a_{i,t}) \right]
$$

这个公式仍然有些复杂，因为它涉及到对整条轨迹的梯度求和。

### 第 2 步：交换求和顺序与重新解释期望

让我们深入观察这个近似的梯度。梯度算子 $\nabla_\theta$ 可以和求和符号交换位置：
$$
\nabla_\theta J(\theta) \approx \nabla_\theta \left( \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i-1} \log \pi_\theta(a_{i,t}|s_{i,t}) \cdot A(s_{i,t}, a_{i,t}) \right)
$$

现在，我们把所有的样本 $(s_{i,t}, a_{i,t})$ 看作一个大的数据集。这个数据集包含了来自 $N$ 条轨迹的所有时间步。这个数据集中样本的总数，我们记为 $M$。
$$
M = T_1 + T_2 + \dots + T_N = \sum_{i=1}^N T_i
$$
因此，我们可以将双重求和改写为单个求和：
$$
\sum_{i=1}^N \sum_{t=0}^{T_i-1} [\dots] = \sum_{j=1}^M [\dots]_j
$$
其中，索引 $j$ 遍历了所有 $M$ 个时间步样本。
那么，上面的求和可以被看作是在所有收集到的时间步样本上的平均：

$$
\nabla_\theta J(\theta) \approx \nabla_\theta \left( \frac{1}{M} \sum_{\text{all} (s_t, a_t) \text{ pairs}} \log \pi_\theta(a_t|s_t) \cdot A_t \right)
$$

这个形式可以被简洁地写成关于**时间步 $t$ 的期望** $\mathbb{E}_t$：

$$
\nabla_\theta J(\theta) \approx \nabla_\theta \left( \mathbb{E}_{t} \left[ \log \pi_\theta(a_t|s_t) \cdot A_t \right] \right)
$$

这里的 $\mathbb{E}_t$ 指的是在你收集的**数据批次中**，对所有时间步 $(s_t, a_t, A_t)$ 求经验平均。
至此，我们已经找到了一个与你的第二个公式非常相似的**目标函数**，我们称之为代理目标函数（Surrogate Objective）：

$$
J_{\text{PG}}(\theta) \approx \mathbb{E}_{t} \left[ \log \pi_{\theta}(a_t|s_t) \cdot A_t \right]
$$

### 第 3 步：引入重要性采样 (Importance Sampling) - 提高样本利用率

到目前为止，我们所有的期望 $\mathbb{E}_{t}$ 都是在**方差非常大** 采样的数据上计算的。这被称为**On-Policy（在策略）**学习。它的致命缺点是：每更新一次参数（$\pi_{\theta}$ 发生变化），之前采样的数据就作废了，必须用新的策略 $\theta$ 重新采样，**基线（Baseline）**。

为了解决这个问题，我们引入**优势函数（Advantage Function）**，允许我们使用从**最小化或最大化一个标量损失/目标函数** 采样的数据来评估和更新**当前策略 $\pi_{\theta_{\text{new}}}$**。

重要性采样的核心是引入一个比率：

$$
r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

这个比率衡量了新旧策略在同一个状态 - 动作对上的概率差异。利用这个比率，我们可以将目标函数从 On-Policy 形式转换为 Off-Policy 形式：

$$
J^{\text{IS}}(\theta) = \mathbb{E}_{t \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} A_t \right] = \mathbb{E}_{t \sim \pi_{\theta_{\text{old}}}} \left[ r_t(\theta) A_t \right]
$$

现在，我们可以用一批从 $\pi_{\theta_{\text{old}}}$ 采集的数据，对 $\pi_{\theta}$ 进行多次梯度更新，大大提高了样本利用率。这也是 TRPO 和 PPO 等算法的基础。

### 第 4 步：裁剪目标函数 (Clipping) - 保证稳定性【PPO 的核心】

重要性采样虽然提高了效率，但也带来了新的问题：如果新旧策略差异太大（即 $\pi_{\theta_{\text{old}}}$ 远离 1），梯度的方差会变得非常大，导致更新不稳定，甚至性能崩溃。

*   TRPO 算法通过一个复杂的二阶优化方法，将新旧策略的 KL 散度限制在一个小范围内来解决此问题，计算代价很高。
*   PPO 则提出了一种更简单、更高效的一阶优化方法：**样本效率极低**。

PPO 的目标函数如下：

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) A_t, \quad \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

这就是 PPO Actor Loss 的最终形态（实际优化时我们最大化它，所以 Loss 是它的负值）。我们来解读这个公式：

1.  它有两个部分，取其中较小的一个：
    *   **重要性采样（Importance Sampling）**: 这就是我们第 3 步得到的重要性采样目标。
    *   **旧策略 $\pi_{\theta}$**: 这是被裁剪过的版本。`r_t(\theta)A_t` 函数将概率比率 $r_t(\theta)$ 强行限制在 `clip(r_t(\theta), 1-ε, 1+ε)A_t` 的区间内（比如 `clip`）。

2.  **新策略 $r_t(\theta)$**:
    *   **直接在目标函数上进行裁剪（Clipping）**: 目标变为 $A_t > 0$。这意味着 $\min(r_t(\theta)A_t, (1+\epsilon)A_t)$ 的收益被限制了，即使它变得很大，带来的好处也不会超过 $r_t(\theta)$。这防止了策略为了某个好动作而更新得过于激进。
    *   **第一部分 `[1-ε, 1+ε]`**: 目标变为 $(1+\epsilon)A_t$。由于 $A_t < 0$ 是负数，这等价于 $\min(r_t(\theta)A_t, (1-\epsilon)A_t)$。这意味着我们对减小这个坏动作概率的“惩罚”也是有下限的。这防止了策略为了避开某个坏动作而过度“逃离”。

通过这种悲观的 `[0.8, 1.2]` 操作，PPO 构建了一个替代目标的下界，确保了每次更新都在一个安全的“信任域”内进行，从而用一种简单高效的方式解决了重要性采样带来的不稳定性问题。

---

## 总结

从最初的策略梯度公式到 PPO 的 Actor Loss，是一个不断解决问题的过程：

**第二部分 `min`** $A_t$ **`min` 操作是精髓** $\max(r_t(\theta)A_t, (1-\epsilon)A_t)$（好动作）** $\xrightarrow{\text{1. 为了降方差}}$ **当 $\xrightarrow{\text{2. 为了适配框架}}$（坏动作）** $\xrightarrow{\text{3. 为了提样本效率}}$ **`AC梯度`** $\xrightarrow{\text{4. 为了保证稳定性}}$ **`引入优势函数A_t`**

最终得到的 $L^{CLIP}$ 就是 PPO Actor 要最大化的目标，因此**构建目标函数** (以及熵的惩罚项)。

