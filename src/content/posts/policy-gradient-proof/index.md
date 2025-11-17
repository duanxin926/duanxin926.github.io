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
hideTOC: true
targetKeyword: ""
draft: false
---
首先我们的**目标是最大化累计奖励期望**，期望是当前策略下，每一条轨迹的概率乘上每条轨迹的奖励。
为什么我们能对累计奖励期望求导，一般情况下他不是常数吗？是因为我们假设策略是一个随机分布，在训练过程中，它的参数会不断变化，每条轨迹的概率发生变化，则累计奖励期望会随着改参数的变化而变化。我们实际是对这个参数求导。那么自然就想到如果能找出梯度上升的方向，就能找到最大化累计奖励期望的策略分布参数。

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


# Reference
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)