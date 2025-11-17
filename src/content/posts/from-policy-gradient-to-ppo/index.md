---
title: 从策略梯度定理到 PPO Actor Loss
description: 从策略梯度到PPO算法中 Actor Loss的详细推导
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
*   $E_{\tau \sim p_\theta(\tau)}[\dots]$ 表示这个期望是在当前策略 $\pi_\theta$ 所产生的**所有可能轨迹** $\tau = (s_0, a_0, s_1, a_1, \dots)$ 上计算的。
*   $\log \pi_\theta(a_t|s_t)$ 是在状态 $s_t$ 下，采取动作 $a_t$ 的对数概率。它的梯度 $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 指向了能让 $(s_t, a_t)$ 状态-动作对出现概率**增加**最快的参数更新方向。
*   $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$ 是**优势函数**。


我们无法直接计算期望 $E[\dots]$，因为它需要遍历所有可能的轨迹。在实践中，我们采用**采样**（Sampling）和**近似**（Approximation）的方法。

---
## 从策略梯度定理到 PPO Actor Loss

### 第 1 步：用蒙特卡洛采样近似期望

我们通过让智能体（Agent）与环境互动，收集一批轨迹。假设我们收集了 $N$ 条轨迹 $\{\tau_1, \tau_2, \dots, \tau_N\}$。我们可以用这批样本均值近似中括号内的式子的期望,**样本均值近似期望**,这就是蒙特卡洛采样的思想：
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left[ \left( \sum_{t=0}^{T_i-1} \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \right) A(s_{i,t}, a_{i,t}) \right]
$$

这个公式仍然有些复杂，因为它涉及到对整条轨迹的梯度求和。

### 第 2 步：交换求和顺序与重新解释期望

由于优势$A$与$\theta$无关(根据第 1 步采集的样本计算),梯度算子 $\nabla_\theta$ 与求和符号交换位置：
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

这里的 $\mathbb{E}_t$ 指的是在收集的**数据批次**中，对所有时间步 $(s_t, a_t, A_t)$ 求经验平均。
至此，我们在上式的括号内已经找到了一个与第二个公式非常相似的**目标函数**，我们称之为**替代目标函数**（Surrogate Objective）：
$$
J_{\text{PG}}(\theta) \approx \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta}} \left[ \log \pi_{\theta}(a_t|s_t) \cdot A_t \right]
$$
这个代理目标函数和原始的$J_(\theta)$的梯度相同.

### 第 3 步：引入重要性采样 (Importance Sampling) - 提高样本利用率

到目前为止，我们的流程都是采集一批数据,进行更新,丢弃,再循环,即**在线学习**（On-policy Learning）。它的缺点是：每更新一次参数（$\pi_{\theta}$ 发生变化），之前采样的数据就作废了，必须用新的策略 $\theta$ 重新采样，导致样本利用率很低。

为了提高样本利用率，PPO算法希望采集一批数据进行多次更新，这就导致策略和数据的不匹配，即用**旧策略的数据**更新**新策略**的参数$\theta$。

**为什么旧策略的数据不能直接用来更新策略**？我们期望最大化的是$J_{PG}(\theta)$，如果用的数据分布不匹配，导致梯度估计有偏，策略就无法朝着正确的方向优化。所以引入了重要性采样。

重要性采样的核心是**引入一个重要性比率**,这个比率衡量了新旧策略在同一个状态 - 动作对上的概率差异：
$$
r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$
根据
$$
\mathbb{E}_{x \sim p}[f(x)] = \int p(x)f(x)dx = \int q(x) \frac{p(x)}{q(x)} f(x)dx = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)}f(x)\right]
$$
我们可以将替代目标函数进行转换：
$$
J^{\text{IS}}(\theta) = \mathbb{E}_{t \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} A_t \right] = \mathbb{E}_{t \sim \pi_{\theta_{\text{old}}}} \left[ r_t(\theta) A_t \right]
$$

现在，我们可以用一批从 $\pi_{\theta_{\text{old}}}$ 采集的数据，对 $\pi_{\theta}$ 进行多次梯度更新，大大提高了样本利用率。这也是 TRPO 和 PPO 等算法的基础。

### 第 4 步：裁剪目标函数 (Clipping) - 保证PPO稳定性的核心

重要性采样虽然提高了效率，但也带来了新的问题：当新策略$\pi_{\theta}$和旧策略$\pi_{\theta_{\text{old}}}$差异很大时，比率$r_t(\theta)$可能会变得非常大或非常接近于0。
* 如果$r_t(\theta)$非常大，即使$A_t$不大，它们的乘积也会导致一次非常大的梯度更新，可能会“冲垮”当前已经学得不错的策略，导致训练不稳定。
* 如果$r_t(\theta)$非常接近0，那么这个样本对梯度的贡献就几乎消失了。

解决方案：
*   TRPO 算法通过一个复杂的二阶优化方法，将新旧策略的 KL 散度限制在一个小范围内来解决此问题，计算代价很高。
*   PPO 则提出了一种更简单、更高效的方法:Clipped Surrogate Objective。

PPO-Clip的目标函数如下：
$$
L^{CLIP}(\theta) = \mathbb{E}_{t} \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$
这里的$\epsilon$是一个超参数（通常为0.1或0.2）.


---

## 总结

从最初的策略梯度公式到 PPO 的 Actor Loss，是一个不断解决问题的过程：
1. 为了降方差，引入**优势函数**
2. 为了适配框架，构建**替代目标函数**
3. 为了提样本效率，引入**重要性采样**
4. 为了保证稳定性，引入 **clipping**

最终得到的 $L^{CLIP}$ 就是 PPO Actor 要最大化的目标。

## 参考
[强化学习文档](https://hugging-face.cn/learn/deep-rl-course/unit8/clipped-surrogate-objective)