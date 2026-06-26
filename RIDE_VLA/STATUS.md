# RIDE-VLA 实验进度（更新于 2026-06-21）

所有实验统一使用 IntentVLA framework + FlowmatchingActionHead。
train_variant 参数区分四种训练目标（见 intent_vla.py）。

---

## LIBERO 实验

### 训练状态

| run_id | 脚本 | train_variant | 状态 | checkpoint |
|---|---|---|---|---|
| LIBERO_base | libero_1_base.sh | base | **训练+eval完成** | final_model ✓ |
| LIBERO_lang_only | libero_2_lang_only.sh | ridevla (lang only) | **训练+eval完成** | final_model ✓ |
| LIBERO_visual_only | libero_3_visual_only.sh | ridevla (visual only) | **训练+eval完成** | final_model ✓ |
| LIBERO_full | libero_4_full.sh | ridevla | **训练+eval完成** | final_model ✓ |
| LIBERO_aug_only | libero_5_aug_only.sh | aug_only | **训练+eval完成** | final_model ✓ |
| LIBERO_output_consistency | libero_6_output_consistency.sh | output_consistency | **训练+standard+plus eval全部完成** | final_model ✓ |
| LIBERO_full_low_distill | libero_4b_full_low_distill.sh | ridevla (w_distill=0.1) | **待运行** | 无 |

> 注：LIBERO_base 原脚本用 QwenPI（不同 action head），已于 2026-06-06 改为 IntentVLA + train_variant=base。

### Standard LIBERO eval 结果（50 trials/task）

| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| LIBERO_base | 94.8% | 99.8% | 98.8% | 94.2% | **96.9%** |
| LIBERO_lang_only | 94.0% | 99.6% | 99.2% | 92.6% | **96.4%** |
| LIBERO_visual_only | 93.4% | 99.8% | 97.8% | 94.0% | **96.3%** |
| LIBERO_full | 93.6% | 100.0% | 99.4% | 93.6% | **96.7%** |
| LIBERO_aug_only | 93.8% | 99.8% | 97.4% | 93.2% | **96.1%** |
| LIBERO_output_consistency | 95.6% | 99.8% | 97.6% | 90.2% | **95.8%** |


### LIBERO-plus eval 结果（per-category breakdown）

> eval 脚本：`RIDE_VLA/eval/eval_libero_plus.sh <ckpt_path>`
> 查看分类分数表：在 `chd/` 根目录执行下方 Python 片段，修改 `run_id` 即可。

```python
import json, os

classification_path = "/nfs/ofs-llm-ssd/user/shengrenren_i/research/LIBERO-plus/libero/libero/benchmark/task_classification.json"
suite_list = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
run_id = "LIBERO_full"   # ← 修改这里
results_dir = f"/nfs/ofs-llab-hdd/users/shengrenren_i/IntentVLA/{run_id}/logs/plus"
folder_name = "pytorch_model"

with open(classification_path) as f:
    classification = json.load(f)
task_to_category = {t["name"]: t["category"] for suite, tasks in classification.items() for t in tasks}

ORDER = ["Camera Viewpoints", "Robot Initial States", "Language Instructions",
         "Light Conditions", "Background Textures", "Sensor Noise", "Objects Layout"]
DISPLAY = {"Camera Viewpoints":"Camera","Robot Initial States":"Robot",
           "Language Instructions":"Language","Light Conditions":"Light",
           "Background Textures":"Background","Sensor Noise":"Noise","Objects Layout":"Layout"}

def load_suite(suite):
    fpath = os.path.join(results_dir, suite, f"{folder_name}_tasks.json")
    if not os.path.exists(fpath): return None
    with open(fpath) as f: d = json.load(f)
    s, n = {c:0.0 for c in ORDER}, {c:0 for c in ORDER}
    for task, rate in d.get("tasks",{}).items():
        cat = task_to_category.get(task)
        if cat in s: s[cat]+=rate; n[cat]+=1
    return s, n

data = {suite: load_suite(suite) for suite in suite_list}
cw = 11
headers = [DISPLAY[c] for c in ORDER] + ["Total"]
print("Suite".ljust(16) + "".join(h.ljust(cw) for h in headers))
print("-" * (16 + cw * len(headers)))
agg_s, agg_n = {c:0.0 for c in ORDER}, {c:0 for c in ORDER}
for suite in suite_list:
    if data[suite] is None:
        print(suite.ljust(16) + "N/A (still running)"); continue
    s, n = data[suite]
    row = [f"{s[c]/n[c]*100:.1f}" if n[c]>0 else "N/A" for c in ORDER]
    ts = sum(s[c] for c in ORDER if n[c]>0); tn = sum(n[c] for c in ORDER if n[c]>0)
    row.append(f"{ts/tn*100:.1f}" if tn>0 else "N/A")
    print(suite.ljust(16) + "".join(v.ljust(cw) for v in row))
    for c in ORDER:
        if n[c]>0: agg_s[c]+=s[c]; agg_n[c]+=n[c]
print("-" * (16 + cw * len(headers)))
tot = [f"{agg_s[c]/agg_n[c]*100:.1f}" if agg_n[c]>0 else "N/A" for c in ORDER]
ts2 = sum(agg_s[c] for c in ORDER if agg_n[c]>0); tn2 = sum(agg_n[c] for c in ORDER if agg_n[c]>0)
tot.append(f"{ts2/tn2*100:.1f}" if tn2>0 else "N/A")
print("Total".ljust(16) + "".join(v.ljust(cw) for v in tot))
```

#### 各 run 汇总（2026-06-21）

| Run | Camera | Robot | Language | Light | Background | Noise | Layout | **Total** |
|---|---|---|---|---|---|---|---|---|
| LIBERO_base | 29.5 | 34.0 | 77.6 | 92.6 | 94.8 | 48.1 | 78.6 | **62.2** |
| LIBERO_lang_only | — | — | — | — | — | — | — | **待跑** |
| LIBERO_visual_only | — | — | — | — | — | — | — | **待跑** |
| LIBERO_full | 62.2 | 60.1 | 89.6 | 91.6 | 95.8 | 82.3 | 81.9 | **79.2** |
| LIBERO_aug_only | 52.7 | 48.1 | 85.0 | 92.4 | 94.5 | 82.0 | 78.9 | **74.6** |
| LIBERO_output_consistency | 59.3 | 57.7 | 84.7 | 95.1 | 94.5 | 84.3 | 81.6 | **78.2** |

#### LIBERO_full（2026-06-09，全部完成）

| Suite | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|---|---|---|---|---|---|---|---|---|
| libero_spatial | 73.1 | 64.3 | 92.8 | 91.1 | 96.5 | 87.5 | 94.5 | 85.3 |
| libero_object | 64.4 | 48.0 | 90.7 | 99.3 | 98.8 | 95.5 | 85.4 | 81.6 |
| libero_goal | 74.8 | 70.9 | 82.0 | 88.2 | 94.0 | 88.4 | 66.1 | 79.4 |
| libero_10 | 37.9 | 57.3 | 93.5 | 87.2 | 94.5 | 60.6 | 83.3 | 70.9 |
| **Total** | **62.2** | **60.1** | **89.6** | **91.6** | **95.8** | **82.3** | **81.9** | **79.2** |

---

## SimplerEnv 实验

### 训练状态

| run_id | 脚本 | train_variant | 状态 | checkpoint |
|---|---|---|---|---|
| SIMPLEENV_base | simpleenv_1_base.sh | base | **训练中断（40000/50000步，6/4停止）** | steps_40000 |
| SIMPLEENV_lang_only | simpleenv_2_lang_only.sh | ridevla (lang only) | **待运行** | 无 |
| SIMPLEENV_visual_only | simpleenv_3_visual_only.sh | ridevla (visual only) | **待运行** | 无 |
| SIMPLEENV_full | simpleenv_4_full.sh | ridevla | **待运行** | 无 |
| SIMPLEENV_aug_only | simpleenv_5_aug_only.sh | aug_only | **待运行** | 无 |
| SIMPLEENV_output_consistency | simpleenv_6_output_consistency.sh | output_consistency | **待运行** | 无 |

> SIMPLEENV_base GPU 空闲，需要恢复训练（从 steps_40000 续跑）。

### SimplerEnv eval 结果

| Method | Spoon | Carrot | Stack | Eggplant | Avg |
|---|---|---|---|---|---|
| SIMPLEENV_base | — | — | — | — | — |
| SIMPLEENV_lang_only | — | — | — | — | — |
| SIMPLEENV_visual_only | — | — | — | — | — |
| SIMPLEENV_full | — | — | — | — | — |
| SIMPLEENV_aug_only | — | — | — | — | — |
| SIMPLEENV_output_consistency | — | — | — | — | — |

---

## Locus 消融实验（新增 2026-06-12）

> 脚本 7/8/9 是 locus ablation，测试不同蒸馏目标位置的影响。LIBERO 三个 run standard+plus eval 全部完成。

| run_id | 脚本 | 描述 | 状态 | checkpoint |
|---|---|---|---|---|
| LIBERO_locus_visual | libero_7_locus_visual.sh | visual projector 输出蒸馏 | **standard+plus eval全部完成** | final_model ✓ |
| LIBERO_locus_all_hidden | libero_8_locus_all_hidden.sh | 全序列 hidden state 蒸馏 | **standard+plus eval全部完成** | final_model ✓ |
| LIBERO_locus_output | libero_9_locus_output.sh | action head velocity 输出蒸馏 | **standard+plus eval全部完成** | final_model ✓ |
| SIMPLEENV_locus_visual | simpleenv_7_locus_visual.sh | (同上，SimplerEnv) | **待运行** | 无 |
| SIMPLEENV_locus_all_hidden | simpleenv_8_locus_all_hidden.sh | | **待运行** | 无 |
| SIMPLEENV_locus_output | simpleenv_9_locus_output.sh | | **待运行** | 无 |

### Standard Locus eval 结果（50 trials/task）

| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| LIBERO_locus_visual | 96.0% | 100.0% | 98.2% | 92.2% | **96.6%** |
| LIBERO_locus_all_hidden | 94.8% | 99.8% | 96.0% | 94.2% | **96.2%** |
| LIBERO_locus_output | 96.2% | 98.8% | 98.6% | 94.2% | **97.0%** |

> 对比参考：LIBERO_full = 96.7%，LIBERO_output_consistency = 95.8%

### LIBERO-plus eval 结果（Locus runs，2026-06-21 全部完成）

| Run | Camera | Robot | Language | Light | Background | Noise | Layout | **Total** |
|---|---|---|---|---|---|---|---|---|
| LIBERO_locus_visual | 54.6 | 54.6 | 82.8 | 91.9 | 94.7 | 84.0 | 83.5 | **76.6** |
| LIBERO_locus_all_hidden | 57.0 | 54.6 | 85.7 | 92.5 | 97.0 | 83.8 | 81.0 | **77.3** |
| LIBERO_locus_output | 55.6 | 59.2 | 85.7 | 93.1 | 96.7 | 85.4 | 82.0 | **78.2** |

> 对比参考：LIBERO_full = 79.2，LIBERO_output_consistency = 78.2

---

## 优先级队列

### 🔴 分析实验（论文核心）

1. **$D_h$ / $D_a$ / eRank 一致性分析（tab:consistency）**
   - 脚本：`RIDE_VLA/analysis/compute_consistency_metrics.py`
   - 运行：`CUDA_VISIBLE_DEVICES=0 python RIDE_VLA/analysis/compute_consistency_metrics.py`
   - 覆盖：base / aug_only / output_consistency / full + **locus_visual / locus_all_hidden / locus_output**（共 7 个 variant）
   - 已有结果（4 variant）：
     - base: D_h=0.0721, eRank=1038
     - aug_only: D_h=0.0310, eRank=1283
     - output_consistency: D_h=0.0246, eRank=1447
     - **full: D_h=0.0003, eRank=53 ← 坍缩告警**
   - ⚠️ **eRank 坍缩问题**：full 的 D_h≈0 + eRank=53 表示 action-query 表示空间严重坍缩，不能作为"表示不变性"的有效证据。待补 locus variants 的 eRank 数据以定位问题来源。

2. **Locus eval（fig:locus）** ✅
   - standard eval **已完成**：locus_visual 96.6% / locus_all_hidden 96.2% / locus_output 97.0%
   - plus eval **已完成**：locus_visual 76.6% / locus_all_hidden 77.3% / locus_output 78.2%
   - 可直接画图：4 个点（visual→all_hidden→output→full）+ output_consistency 参考线

3. **eRank 坍缩诊断：低蒸馏权重实验**
   - 脚本：`RIDE_VLA/train/libero_4b_full_low_distill.sh`
   - 变更：`w_distill=0.1`（原 0.5），其余完全相同
   - 判定规则：
     - eRank 恢复~1000+ 且 Plus ≈79 → 坍缩是调参问题，论文故事成立
     - eRank 恢复但 Plus 降到~78 → 增益和坍缩同源，方法本质是正则化
     - eRank 仍低 → action-query locus 的结构问题
   - 状态：**待运行**

4. **LIBERO-plus eval: lang_only / visual_only** — lang_only 只有 libero_spatial（崩溃，ImageMagick缺失）；visual_only 无 plus 结果，均需重跑

### 🟠 P1（审稿必查）

4. **LIBERO_output_consistency standard eval** ✅ — 全部完成：Spatial 95.6% / Object 99.8% / Goal 97.6% / Long 90.2% → Avg **95.8%**
5. **w/o EMA teacher / w/o teacher action loss** — 新训练 run，待开始
6. **多 seed 方差** — 至少主对照（base vs full）各 3 seed
7. **训练时间/显存测一下** — Implementation Details

### 🟡 P2（附录）

8. **SIMPLEENV_base 续跑** — 从 steps_40000 恢复，还剩 10000 步
9. **SIMPLEENV 系列** — base 完成后，按 base → full → 其余顺序训练

---

## 说明

- 所有 LIBERO 脚本位于 `RIDE_VLA/train/libero_*.sh`
- 所有 SimplerEnv 脚本位于 `RIDE_VLA/train/simpleenv_*.sh`
- eval 脚本位于 `RIDE_VLA/eval/`
- 结果日志路径：`/nfs/ofs-llab-hdd/users/shengrenren_i/IntentVLA/{run_id}/logs/`
- LIBERO-plus 环境配置记录见 `exp.md` § 11.5
