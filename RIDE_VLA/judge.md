 ---
  审稿报告：RIDE-VLA (aaai.tex 当前版本)

  目标会议：AAAI-27 主会（Artificial Intelligence / Robotics track）

  ---
  I. 综合评估与推荐

  1. 核心贡献

  论文提出 RIDE-VLA：在 VLA 训练中，用 EMA clean-teacher 对 perturbed-student 在 action-query hidden states 位置施加余弦一致性损失，同时保留 flow-matching action head
  的模仿监督，推理时零额外开销。贡献叙事已从早版本有所调整，新增 Contribution 2："where invariance must be enforced"——即通过 Proposition 1 和 distillation-locus ablation 论证
  action-query interface 优于 visual features 和 output-level consistency。方向是正确的。

  2. 针对 AAAI-27 的契合度评估

  方向契合，但当前版本距离 AAAI-27 可投稿状态仍有根本性差距。论文在三个层面存在致命问题：（A）稿件尚未完成（大量 placeholder 和 XX.X
  数据），（B）核心实验结果中存在可重复验证的算术错误，（C）最核心的新意（locus ablation）所依赖的实验图表全部是占位符。

  3. 推荐意见

  Reject — 当前版本不具备 AAAI-27 可审稿性。

  这不是方向问题，而是执行和完成度的问题。如时间线允许，AAAI-27 截止前需完成所有实验并完整修改；若时间不足，建议先以当前进度向 CoRL/ICRA/ICLR robotics track
  投稿，积累更强实验后冲击顶会。

  ---
  II. 必须解决的核心问题

  问题 1：稿件实质上仍处于草稿状态——5张图全是 placeholder，3张表全是 XX.X

  这是最直接的不可审稿问题。具体未完成内容如下：

  ┌────────────────────────────────┬────────────────────────────────────────────────────┐
  │              位置              │                        问题                        │
  ├────────────────────────────────┼────────────────────────────────────────────────────┤
  │ \gainComb{XX.X}（第 104 行）   │ 核心声明未填                                       │
  ├────────────────────────────────┼────────────────────────────────────────────────────┤
  │ \dropOurs{XX.X}（第 105 行）   │ 核心声明未填                                       │
  ├────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Figure 3 (fig_combined)        │ "Placeholder (run scripts/...)"                    │
  ├────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Figure 4 (fig_representation)  │ "Placeholder (run scripts/...)"                    │
  ├────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Figure 5 (fig_locus)           │ "Placeholder (run scripts/...)"                    │
  ├────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Figure 6 (fig_robustness_drop) │ "Placeholder (run scripts/...)"                    │
  ├────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Figure 7 (fig_lambda)          │ "Placeholder (run scripts/...)"                    │
  ├────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Table 3（tab:consistency）     │ 全部 XX.XXX                                        │
  ├────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Table 4（tab:ablation）        │ 除 "visual perturb. only / Clean=93.4" 外全部 XX.X │
  ├────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Table 5（tab:simplerenv）      │ 全部 XX.X                                          │
  └────────────────────────────────┴────────────────────────────────────────────────────┘

  Introduction 第 4 段（第 174-175 行）写道："Further diagnostic analysis shows that \method{} substantially reduces both representation distance and action
  discrepancy..."——但这些分析所在的 Table 3 和 Figure 4 是空的。这句话在当前稿中是无根据的声明。同样地，Contribution 2 声称"empirically (a distillation-locus ablation)"——但 Figure 5 是
  placeholder，该贡献目前没有实验支撑。AAAI 审稿人看到 "Placeholder (run scripts/...)" 会立即拒稿，不经讨论。

  最低要求：所有 XX.X / Placeholder 必须在投稿前清零；所有图表替换为真实数据。

  ---
  问题 2：Table 1 存在系统性算术错误，严重损害数据可信度

  按当前表中的逐列数值重算简单平均值：

  ┌────────────────────┬───────────┬────────────────┬──────┐
  │       Method       │ 表中 Avg. │ 按列重算（÷7） │ 差值 │
  ├────────────────────┼───────────┼────────────────┼──────┤
  │ Base VLA           │ 62.2      │ 65.0           │ −2.8 │
  ├────────────────────┼───────────┼────────────────┼──────┤
  │ Aug. only          │ 74.6      │ 76.2           │ −1.6 │
  ├────────────────────┼───────────┼────────────────┼──────┤
  │ Action Consistency │ 78.2      │ 79.6           │ −1.4 │
  ├────────────────────┼───────────┼────────────────┼──────┤
  │ RIDE-VLA           │ 79.2      │ 80.5           │ −1.3 │
  └────────────────────┴───────────┴────────────────┴──────┘

  （Base VLA 各列：29.5+34.0+77.6+92.6+94.8+48.1+78.6 = 455.2，÷7 = 65.03）

  四行数据全部不一致，且差值方向一致（表中 Avg 均偏低），说明平均方式与简单均值不同。论文中没有任何说明。直接后果：

  - Introduction 中 "\gainBase{17.0} points" 声明不成立：若用重算均值，RIDE-VLA(80.5) vs Base VLA(65.0) 是 +15.5，而非 17.0。
  - "\gainAug{4.6} points" 也不成立：RIDE-VLA(80.5) vs Aug. only(76.2) 是 +4.3。
  - RIDE-VLA vs Action Consistency 差距仅约 0.9 点（80.5 vs 79.6），这个边距在无统计检验的情况下几乎不可区分。

  必须修复：若使用加权平均（按任务数量），必须在脚注或正文中明确说明任务数量和加权方式，并提供原始公式。若使用简单平均，所有 Avg.
  数值必须与列值吻合。当前状态下，审稿人会怀疑数据是手工填写或来自不同 checkpoint。

  ---
  问题 3：新颖性的最强 claim（"where invariance matters"）在当前版本仍缺乏实证支撑

  论文的 Contribution 2（第 179-181 行）明确声称：

  ▎ "We show, both formally (Proposition 1) and empirically (a distillation-locus ablation), that...robustness should be enforced at the action-conditioning representation."

  这是全文最有辨识度的贡献——但 Figure 5（locus ablation scatter plot）是 placeholder。Proposition 1 的 part (i) 本质上是定义性结论（$p_\phi(\tau|h)$ 只依赖 $h$，因此 $h$
  相同时分布相同），没有实质理论含量；part (ii) 关于 "measure-zero slice" 的论证在直觉上合理，但 proof sketch 不形式化（缺少 velocity field regularity
  条件、函数类假设、度量空间定义）。在 locus ablation 图表缺失的情况下，Contribution 2 的"实证"部分完全是空声明。

  更根本的问题：在当前版本中，论文的核心差异化 claim 与 RobustVLA/RoVLA（均施加某种 clean-to-perturbed consistency 约束）的区别，必须完全依赖 locus ablation
  来支撑。如果这张图不存在，论文与同期工作的区分度就消失了。

  ---
  问题 4：RIDE-VLA vs Action Consistency 的边距不足以支撑核心论点，且完全缺乏统计支撑

  即使用表中自己声明的数字：RIDE-VLA(79.2) vs Action Consistency(78.2) = +1.0 点。结合以下事实：

  - 完全没有 random seed 信息
  - 没有每类扰动的 episode 数量
  - 没有置信区间或任何统计检验
  - Clean performance: Base VLA(96.9) vs RIDE-VLA(96.7)，RIDE-VLA 反而略低

  1.0 点的差距在 AAAI 标准下极易被判定为 "within noise"。论文的核心论点是 "representation-level invariance is strictly stronger than output-level consistency"，但 1.0 点的 success rate
  差距根本无法支撑 "strictly stronger" 这一措辞。

  必须补充：至少 3 个 random seeds；每扰动类别的 episode 数量；bootstrap confidence intervals 或 paired t-test；并将措辞从 "strictly stronger" 降级为 "consistently outperforms"。

  ---
  问题 5：关键实现细节缺失，当前不可复现

  方法部分（Section 3）遗漏以下在 AAAI reproducibility checklist 中被明确要求的信息：

  - action-query tokens 的具体来源：K 值是多少？从哪一层取 h？是否经过 layer norm？
  - EMA momentum schedule 的具体形式（"gradually increases" 描述不够）
  - λS, λT, λD 的具体值及调参方式
  - flow time s 的采样分布
  - visual perturbation 的具体强度范围（color jitter 的参数、crop ratio 等）
  - paraphrase bank 的规模、生成方式、语义等价性验证方法
  - teacher action loss 是否会导致 action head 对 clean teacher representation 产生 bias（这是一个开放的设计问题，论文没有讨论）

  此外，第 299 行关于"robotics-safe augmentations that preserve task semantics"的陈述是一个未经验证的前提——论文没有提供任何 perturbation quality control 的说明（随机抽样人工核查、自动
  object visibility check、对颜色相关任务的特殊处理等）。

  ---
  问题 6：Related Work 仍未明确区分与 RoVLA/RobustVLA 的差异

  第 190 行将 RobustVLA 和 RiPT-VLA 合并在一个引用列表末尾（~\cite{...,ript_vla,robustvla}），但没有专门的段落分析这些工作与 RIDE-VLA 的关系。RobustVLA 同样研究多模态扰动下的 VLA
  鲁棒性并施加一致性约束；如果不在 Related Work 中明确解释二者的关键区别（例如：RIDE-VLA 专注于 action-query interface 而非 visual features 或 output），AAAI
  审稿人会认为贡献差异不清晰。

  必须增加：对 RobustVLA、RoVLA（如已发表）的逐点比较段落，明确说明"other works enforce consistency at X or Y, while RIDE-VLA enforces it at Z, and locus ablation shows Z is better"。

  ---
  III. 其他改进建议

  1. Title 仍然过于泛化。 "Representation Invariance Distillation for Robust VLA" 缺乏辨识度。建议改为聚焦核心贡献的标题，例如 "Where to Enforce Invariance in Flow-Matching VLA
  Policies: The Action-Conditioning Interface"。这样标题本身就是论点，有助于 AAAI 审稿人快速理解贡献的独特性。
  2. Abstract 过度承诺。 第 142 行 "show that drift is a distinct and measurable source" 需要实验证据；Table 3 是空的，目前无法支撑。改为 "identify representation drift as a measurable
  correlate of brittleness"，待实验完成后再升级措辞。
  3. Clean performance 表述有误导性。 Table 2 (LIBERO clean) 中 RIDE-VLA(96.7) 实际低于 Base VLA(96.9)。Section 4.3 的措辞 "RIDE-VLA maintains or improves clean-task performance"
  不准确——Long: 93.6 vs 94.2（Base），Spatial: 93.6 vs 94.8（Base），两项都下降了。应改为 "RIDE-VLA does not significantly degrade clean performance, with differences within 1.2
  points"，不要用 "maintains or improves"。
  4. Proposition 1 的措辞应降级。 Part (i) 的结论是定义性的，不应称为 "Proposition"；建议改为 "Observation" 或融入正文段落。Part (ii) 若要保留 formal
  形式，需补充函数类和度量空间的假设。同时，正文第 245 行 "strictly stronger" 用于描述 representation-level invariance 时是绝对断言；改为 "a sufficient condition for full
  distributional invariance, stronger than output-level consistency in a formal sense" 更严谨。
  5. 外部 baselines 的使用策略需要调整。 Table 1 将 OpenVLA-OFT(69.6) 与 RIDE-VLA(79.2) 放在同一表中，正文第 400 行的措辞暗示 RIDE-VLA 优于外部 baselines。但 OpenVLA-OFT 与 RIDE-VLA 在
  model scale、pretraining data、action decoder 上均不同，这个比较不公平。建议主结论完全锚定在 controlled baselines；external rows 仅用于说明任务难度和鲁棒性水平的绝对参照，并在
  caption 中明确标注"not directly comparable"。
  6. 语言表达需要通篇降温。 多处过强表述需修改："correct interface" → "a suitable interface"；"guarantees invariance" → "encourages invariance"；"stems from" → "is associated
  with"（在机制尚未完全建立前）；"precisely when perturbations compound" → "especially when perturbations compound"。

  ---
  最核心修改路线（总结）

  论文的方向已经调整到正确轨道——以"where to enforce invariance"作为核心问题，而非泛泛的 robust VLA framework。这是正确的。但要让这条路走通，最高优先级的任务只有一件：

  ▎ 跑完所有实验，生成所有图表和数值，清零所有 placeholder。

  这不是润色问题，是可审稿性的门槛。在此之上，Table 1 的算术错误必须修复（补充加权说明或重新计算），statistical testing 必须加入，Related Work 必须增加对 RobustVLA/RoVLA 的显式区分。

  如果 locus ablation（Figure 5）的结果能清楚地显示 action-query interface 在 Robust Avg 和 Dh 两个维度都优于 visual features、all hidden states 和 output
  consistency，那么这篇论文是有机会成为一个有清晰观点的 AAAI submission 的。但前提是实验必须跑完，数据必须正确，统计必须严格。