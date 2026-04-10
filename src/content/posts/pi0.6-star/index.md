---
title: 论文理解-Pi0.6-star
description: 利用 advantage conditioning 做真机微调的思路
date: 2025-09-08
tags:
  - Reinforcement Learning
categories:
  - 论文理解
# image: "[[cover.png]]"
# imageAlt: A beautiful multi-tiered waterfall cascading into a turquoise lake surrounded by lush green forest.
imageOG: false
hideCoverImage: false
hideTOC: false
targetKeyword: ""
draft: false
---
## 快速总结
为了做真机后训练, 用一个预训练好的 VLA($\pi_{0.6}$)收集自主尝试和人为纠错数据, 再用所有数据训一个 value function; 根据value 对数据做简单的正负标记("Advantage: positive" 或 "Advantage: negative"), 这个标记作为一个额外的条件输入, 进行**Conditioned Training**.
部署 → 收集数据（自主尝试 + 人工纠正） → 训练价值函数（学会判断好坏） → 训练策略（学会根据“好”的指令行动）→ 部署更强的模型...

这里给的故事背景是 当人类在学习新技能的时候，通常会经过三个步骤:
1. 我们会先了解大概有些什么样的策略和技巧，有什么常见的错误; 
2. 会有一个老师 不只是告诉我们怎么做，还会带我们一起训练，纠正我们犯的错误; 
3. practice makes perfect, 我们通过大量的练习来逐渐完善这个技能.
4. 

---
### Why is imitation learning not enough?
