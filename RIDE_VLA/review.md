Peer Review Report
I. 综合评估与推荐
1. 核心贡献

稿件提出 RIDE-VLA：在 VLA 训练中使用 clean EMA teacher 和 perturbed student，对齐二者在 action-query / action-conditioning interface 处的表示，同时保留 flow-matching action head 的动作监督。推理时只使用 student 分支，不引入额外运行时模块。论文声称该方法能减少 clean-to-perturbed representation drift，并提升 LIBERO-Plus 中视觉、语言及组合扰动下的鲁棒性。

这个问题本身是有价值的：VLA 在语义保持扰动下的不稳定性确实是近期机器人学习中的重要问题。LIBERO-Plus 明确展示了 VLA 在 camera viewpoint、robot initial state、language instruction、lighting、background、sensor noise、object layout 等七类扰动下的脆弱性。 VLATest 也从 fuzzing 角度指出当前 VLA 在复杂场景、光照、相机姿态、未见物体和指令变异下存在鲁棒性缺陷。

2. 针对 AAAI-27 的契合度评估

方向契合，但论文质量与新颖性目前不达标。 从主题上看，robust VLA、representation invariance、flow-matching action head 都属于 AAAI 主会可能接受的 AI/robotics 方向。但 AAAI-27 对 novelty、soundness、clarity、reproducibility 的要求很高，而当前稿件存在三类致命问题：

第一，稿件明显未完成。正文中存在大量 ?、???、XX.X、placeholder figures、未填实验表格和未补引用。这不是小问题，而是直接破坏可审稿性。

第二，新颖性边界没有站住。RoVLA、RobustVLA、STRONG-VLA 等近期工作已经覆盖了 VLA 多模态扰动鲁棒性、一致性训练、输入/输出扰动鲁棒优化等关键空间。RIDE-VLA 目前只是在“对齐哪个表示层”上做了相对窄的定位，但稿件没有充分证明 action-query representation 是一个足够强、足够新的切入点。

第三，实验与论证尚不足以支撑核心 claim。论文声称“representation drift 是 VLA brittleness 的来源”，但当前实验主要是相关性式的 consistency analysis，而且 Table 3、Table 4、Figure 4、Figure 5 仍是占位符；这不足以支撑因果层面的机制解释。

3. 推荐意见

Recommendation: Reject / Not ready for AAAI-27 submission in current form.

这不是因为方向差，而是因为当前稿件在 AAAI 标准下存在明显硬伤：未完成、结果不一致、相关工作威胁强、理论命题不严谨、机制证据缺失。若要冲 AAAI-27，必须进行实质性重写和补实验；如果时间不足，更现实的策略是先完成为 workshop / arXiv 版本，再积累更强实验后投 CoRL / ICRA / ICLR robotics-oriented track 或 AAAI 后续版本。

II. 必须解决的核心问题
1. 新颖性受到 RoVLA 等同期工作的直接威胁，当前贡献边界不清

这是最大问题。稿件把核心贡献包装为“clean-to-perturbed self-distillation + representation-level invariance + zero-overhead inference”。但 RoVLA 已经提出面向 robust VLA 的多一致性约束，覆盖 instruction reformulation、observation perturbation 和 flow-matching/action evolution 层面的 consistency，并且同样强调视觉变化、指令改写和复合扰动。 RobustVLA 也已经研究多模态扰动，并对输入语义保持变化施加一致性，对输出扰动做鲁棒优化。

你现在必须把论文从“又一个 robust VLA consistency/distillation 方法”改写成一个更窄但更硬的 claim：

不是提出 VLA 鲁棒性的一致性训练框架，而是证明在 flow-matching VLA 中，action-conditioning/action-query interface 是比视觉特征、全 hidden states、输出 velocity 更有效的 invariance locus。

也就是说，真正可能成立的新意不是 teacher-student，也不是 perturbation consistency，而是 where to enforce invariance。因此，Figure 5 的 distillation-locus ablation 不能再是 appendix-style diagnostic，而必须成为主实验核心。否则这篇论文会被审稿人认为与 RoVLA/RobustVLA 过度相似。

2. 稿件仍处于未完成草稿状态，无法作为 AAAI 投稿

论文中有大量明显未完成内容。例如 Related Work 中存在 ?、???、Black et al. 2024; ?、VAT (?)、FixMatch (?)、DINO (?)、DrQ (?) 等未补引用；实验部分 Figure 3、Figure 4、Figure 5、Figure 6、Figure 7 仍是 placeholder；Table 3、Table 4、Table 5 包含大量 XX.X；正文甚至保留了 “Placeholder (run scripts/...)” 这样的内部提示。

这在 AAAI 审稿中会被视为不可审稿，不是“需要补充细节”。AAAI reproducibility checklist 明确要求说明超参数范围、代码、随机种子、硬件软件、运行次数、统计显著性、评价指标动机等信息。 当前稿件不仅没有满足这些要求，连关键实验结果都没有填完。

最低修改要求：所有 ? / ??? / XX.X / Placeholder 必须清零；所有图表必须替换为真实图；所有实验都要说明 seeds、任务数、episode 数、训练数据、checkpoint、评测协议、是否使用官方 split、是否使用相同训练数据。

3. Table 1 的数值存在算术错误，严重损害可信度

Table 1 中 controlled baselines 的 Avg. 与逐列数值不一致。按表中七个扰动类别重新计算：

Method	表中 Avg.	按列重算 Avg.
Base VLA	62.2	65.0
Aug. only	74.6	76.2
Action Consistency	78.2	79.6
RIDE-VLA	79.2	80.5

因此，文中“RIDE-VLA improves over Base VLA by 17.0 points”和“over Aug. only by 4.6 points”都不成立。若按重算均值，RIDE-VLA 相对 Base 是约 +15.5，相对 Aug. only 是约 +4.3，相对 Action Consistency 只有约 +0.9。

这会触发审稿人的强烈怀疑：结果是手工填的？是否来自不同 runs？是否平均方式不同？如果平均方式加权，为什么没有说明权重？如果是不同任务数量加权平均，必须给出每类任务数和加权公式。当前版本会被视为实验不可信。

4. 核心机制证据不足：你没有真正证明 representation drift 是“原因”

论文反复声称 representation drift 是 VLA brittleness 的来源，但目前的证据不足。即使 RIDE-VLA 降低了 Dh 并提升了 success rate，也只能说明二者相关，不能说明 drift 是失败原因。更严重的是，Table 3 仍是占位符，Figure 4 仍是占位符，所以机制 claim 目前没有实证支撑。

必须补充至少三类证据：

Failure-level correlation：在 Base VLA 上，clean-to-perturbed Dh 是否能预测 episode failure？需要报告成功/失败样本的 Dh 分布、AUC 或 logistic regression。
Intervention evidence：只改变 h 的扰动程度，观察 action discrepancy 或 success rate 是否随 h drift 单调恶化。否则“drift causes brittleness”只是叙事。
Locus ablation：visual/projector、all hidden states、action output、action-query 四个位置必须在同一 backbone、同一数据、同一扰动、同一 loss weight 下比较，并报告 robust avg、clean avg、Dh、Da、effective rank。

没有这些证据，论文最好把表述降级为：“representation drift is associated with brittleness and can be reduced by our training objective”，不要写成“stems from”。

5. Proposition 1 过度包装，理论贡献目前不严谨

Proposition 1 的第一部分基本是定义性结论：如果两个 h 完全相同，且 action head 只依赖 h，那么 action distribution 相同。这并不是一个有实质理论含量的 proposition。第二部分说 output-level consistency 只约束 finite sampled flow times/noise，因此存在 h≠h̃ 在这些点上一致但 elsewhere 不一致；这个方向直觉上合理，但当前 proof sketch 不够形式化，也没有说明函数类、head injectivity、velocity field regularity、sampling distribution、度量空间等条件。

更关键的是，论文实际训练用的是 cosine loss，不保证 hS = hT；因此 proposition 中的“exact invariance implies distribution invariance”与实际算法之间存在断裂。建议改为一个更有用、更诚实的 bound：

若 flow-matching head 对 h 是 L-Lipschitz，则 velocity discrepancy 或生成分布距离可由 representation distance 上界控制。

然后用实验估计 Dh 与 Da 的关系。这样理论与实验能闭合，而不是现在这种“形式上显得有 theorem，但实际支撑很弱”的状态。

6. 与 Action Consistency 的差距过小，不能支撑“representation-level clearly better”的强 claim

按 Table 1 当前数值重算，RIDE-VLA 相比 Action Consistency 的平均提升只有 0.9 points。这非常脆弱，尤其没有 seeds、confidence intervals、statistical test。AAAI 审稿人很可能会问：这是否只是随机波动？是否值得提出一个新框架？

因此必须补充多种统计证据：至少 3 seeds；每个 perturbation category 的 episode 数；bootstrap confidence intervals；paired significance test。还要报告 clean performance 的 variance，因为 Table 2 中 RIDE-VLA 的 clean avg 是 96.7，Base 是 96.9，并没有“improves clean performance”，最多是“不显著下降”。

7. 外部 baseline 比较不严谨，容易被认为不公平

Table 1 把 OpenVLA、OpenVLA-OFT、π0、π0-FAST 与 controlled baselines 放在同一表中，但这些模型的 scale、pretraining data、fine-tuning recipe、action decoder、training data 都不同。稿件虽然说 external rows are reference systems，但正文又用这些数字暗示 RIDE-VLA 优于 strong public systems。

这在审稿中很危险。你要么只把外部 baselines 放到“context only”，明确不做 superiority claim；要么必须在相同 protocol 下复现。更好的 AAAI 写法是：主结论完全基于 controlled baselines；external baselines 只用于说明任务难度和当前鲁棒性水平。

8. 方法细节不足，当前不可复现

RIDE-VLA 的关键实现没有讲清楚：action-query tokens 是 learnable tokens 还是 special tokens？K 取多少？从哪一层取 h？是否 layer norm？teacher 如何初始化？EMA momentum schedule 具体是什么？λS、λT、λD 如何选？flow time s 的分布是什么？visual perturbation 的强度范围是什么？paraphrase bank 如何生成和过滤？如何验证 paraphrase 不改变 task semantics？teacher action loss 是否真的必要，是否会造成 head 对 clean teacher bias？

AAAI reproducibility checklist 明确要求说明超参数范围、最终超参数、随机性、硬件软件、评价指标动机、运行次数和统计显著性。 当前方法部分停留在概念层，缺少足够工程细节。

9. “semantics-preserving perturbation” 是全篇地基，但没有验证

论文大量依赖“visual perturbation 和 language paraphrase 不改变任务语义”这一前提。但在机器人任务里，这个前提并不自动成立。裁剪可能移除目标物；颜色扰动可能改变颜色条件任务；背景扰动可能影响空间关系；paraphrase 可能改变 object reference 或 relational semantics。LIBERO-Plus 也指出 VLA 对语言变化的敏感性和语言忽略现象本身很复杂。

必须增加 perturbation quality control：随机抽样人工检查；自动 object visibility check；paraphrase semantic equivalence filter；对颜色/空间/左右关系任务禁用某些增强；报告过滤比例。否则 distillation target 可能是噪声目标。

III. 其他改进建议
题目需要更聚焦。 当前标题像一个泛泛的 robust VLA 方法。更有辨识度的标题应突出 “action-conditioning interface” 或 “where to enforce invariance”。
Abstract 过度承诺。 “representation drift stems in part...” 可以保留，但 “show that drift is a distinct source” 需要实验证据支撑；否则改成更保守的 “we identify representation drift as a measurable correlate”。
Related Work 必须重写。 现在是堆文献，而且大量引用缺失。必须单独讨论 RoVLA、RobustVLA、STRONG-VLA、LIBERO-Plus、VLATest、StarVLA，并明确 RIDE-VLA 与它们的差异。StarVLA 是模块化 VLA 代码库，支持 VLM backbone 和 action head 解耦，并集成 LIBERO、SimplerEnv、RoboTwin 2.0 等 benchmark；你可以说自己基于其基础设施实现，但不要让读者误以为 StarVLA 是一个传统 baseline。
Figure 1/2 可以保留，但要降低“宣传图”味道。 Figure 1 的概念清楚，但缺少真实数据支撑；建议加入一个小的 empirical inset，比如 Base 与 RIDE 的 Dh 分布或 clean/perturbed action discrepancy。
实验部分应重排。 目前主实验、机制分析、ablation 的逻辑顺序还可以，但核心应改成：先证明 drift 存在且与失败相关；再证明 action-query locus 最好；最后报告鲁棒性提升。现在顺序是先报成功率，再补机制，显得像事后解释。
语言上避免过强表达。 “correct interface”“strictly stronger”“guarantees invariance”“stems from” 都太绝对。AAAI 审稿人会抓这些词。建议改成 “a suitable interface”“provides a stronger sufficient condition”“encourages invariance”“is associated with”。
最核心的修改路线

这篇稿子不是不能救，但必须换主线。不要再主打“我们提出一个 robust VLA distillation framework”，因为这个空间已经拥挤且 RoVLA 直接压住你。应改成：

RIDE-VLA 的核心问题：在 flow-matching VLA 中，语义保持扰动下的 invariance 应该施加在哪里？

围绕这个问题，你需要把 distillation-locus ablation + representation drift causal evidence 提升为主贡献。只要你能证明 action-query/action-conditioning interface 在鲁棒性、clean performance、non-collapse、action discrepancy 上都优于 visual features、all hidden states 和 output consistency，那么论文还有机会成为一篇有清晰观点的 AAAI submission。否则，当前版本在 AAAI-27 下会被判为：incremental, under-supported, unfinished, and insufficiently distinguished from concurrent work.
