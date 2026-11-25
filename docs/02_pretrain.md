# Pretrain 原理：让 MiniMind 学习预测下一个 token

源码核对版本：[`858008eb98da`](https://github.com/jingyaogong/minimind/tree/858008eb98da)。

阅读顺序：[模型结构](01_model.md) → 本文 → [Pretrain Notebook](../notebooks/01_pretrain.ipynb)。本节的“预训练”是从随机初始化参数出发，在纯文本上做**因果语言建模**；它不是指令微调，也不需要奖励模型。

## 1. 学习目标：条件概率

给定 token 序列 `x₀, x₁, …, xₙ`，模型在位置 `t` 只看到 `x₀ … xₜ`，预测 `xₜ₊₁`。整条序列的训练目标是最大化真实后续 token 的条件概率，等价于最小化有效位置的平均交叉熵：

```text
L = - (1 / N) Σ_t log pθ(xₜ₊₁ | x₀, …, xₜ)
```

这里 `N` 是参与损失的目标 token 数。假设 token ID 为 `[10, 20, 30, 40]`，计算时会把前三个位置的 logits 与 `[20, 30, 40]` 对齐。模型不能通过“看到答案”作弊，因为自注意力有因果掩码。

上游 `MiniMindForCausalLM.forward` 对 logits 取 `[..., :-1, :]`，对 labels 取 `[..., 1:]`，然后调用 `cross_entropy(..., ignore_index=-100)`。这就是代码中的**移位预测**。loss 通常以自然对数计算；单独一个 loss 数值不能直接说明模型会不会聊天。

## 2. 从 JSONL 到训练样本

ModelScope 的 [`pretrain_t2t_mini.jsonl`](https://modelscope.cn/datasets/gongjy/minimind_dataset) 每行是包含 `text` 的 JSON 对象：

```json
{"text": "这里是一段用于学习语言规律的文本。"}
```

上游 `PretrainDataset` 的处理顺序是：

1. 读取 `text` 并用 MiniMind tokenizer 编码，至多保留 `max_seq_len - 2` 个正文 token。
2. 在前后加入 BOS、EOS；长度不足时用 PAD 补齐到 `max_seq_len`。
3. 复制 `input_ids` 作为 labels，把 PAD 的 label 改为 `-100`，使 padding 不参与交叉熵。
4. 模型内部再将 logits 与 labels 错开一位，形成“已知前文、预测后文”的监督对。

在课程 Notebook 中，`max_seq_len=340`。如果一行文本超过这个长度，后半段会被截断；如果很短，则有 padding。**340 是本次实验的训练序列长度**，不能由此推断模型在更长文本上的效果。

## 3. 一次参数更新怎样发生

训练循环按以下顺序工作：取 batch → 模型前向得到 loss → 反向传播梯度 → 梯度裁剪 → AdamW 更新参数 → 清空梯度。课程使用混合精度：支持时采用 BF16，否则上游脚本会回退到 FP16。混合精度降低部分计算与激活的显存开销，但优化器状态仍会占用显存。

本课程短实验取 **512 条**、`batch_size=8`、`accumulation_steps=4`、`epochs=1`：

```text
512 条 / 每微批 8 条 = 64 个微批
64 个微批 / 累积 4 次 = 16 次优化器更新
有效 batch = 8 × 4 = 32 条
```

梯度累积会将每个微批的 loss 除以 4 后反向传播，累计四次再更新一次参数。它能在较小瞬时显存下模拟更大的有效 batch，但一次更新仍需要处理四个微批，总计算量不会凭空减少。若 batch 从 8 降到 4 而想保持有效 batch 32，累积次数应改为 8。

默认初始学习率为 `5e-4`，`get_lr` 使用可选 warmup 加余弦衰减，最终约为初始值的十分之一。训练循环还会裁剪梯度范数。短实验只有 16 次更新，loss 下降可说明程序在拟合这批数据，不能代表通用语言能力。

## 4. 显存、检查点与训练结果

显存主要来自权重、梯度、AdamW 状态和中间激活。增大 batch 或序列长度通常会提高显存需求；完整数据集变大主要增加训练时间，不会按文件大小直接把 GPU 显存占满。当前课程的 512 条样本短实验在 GPU 6 的 A100 上实测整卡峰值约 **3.06 GiB**，这是该配置的观测值，不保证其他 GPU、PyTorch 版本或更长序列完全相同。

训练脚本在 epoch 结束时保存 `course_pretrain_768.pth`，其中是可供下一节 SFT 加载的模型权重；还会在上游源码的 `checkpoints/` 目录保存续训所需状态。**模型权重**用于推理或作为下阶段起点；**完整训练检查点**还需优化器、步数等信息，才能较完整地接续训练。Notebook 的输出目录是 `work/runs/pretrain/`，不会覆盖用户已有的 `pretrain_768.pth`。

512 条样本只用来走通数据、前向、反向和保存链路。要评价语言建模能力，应使用未参与训练的验证集，比较同一 tokenizer 下的交叉熵或 perplexity；要评价指令响应，还要经过 SFT 并做独立评测。训练集 loss 下降不等于回答质量提高。

## 源码对照与思考题

- [`dataset/lm_dataset.py`](https://github.com/jingyaogong/minimind/blob/858008eb98da/dataset/lm_dataset.py)：`PretrainDataset` 中的 BOS、EOS、PAD、`-100`。
- [`model/model_minimind.py`](https://github.com/jingyaogong/minimind/blob/858008eb98da/model/model_minimind.py)：`MiniMindForCausalLM.forward` 中的 logits/labels 移位。
- [`trainer/train_pretrain.py`](https://github.com/jingyaogong/minimind/blob/858008eb98da/trainer/train_pretrain.py)：训练循环、梯度累积、优化器与保存。
- 思考：如果把 `max_seq_len` 从 340 增到 680，数据和显存会分别怎样变化？
- 思考：为什么训练集 loss 下降不能证明模型已经会回答新问题？
