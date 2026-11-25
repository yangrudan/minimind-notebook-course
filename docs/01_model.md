# MiniMind 模型：从 token 到下一个 token 的概率

源码核对版本：[`858008eb98da`](https://github.com/jingyaogong/minimind/tree/858008eb98da)。

阅读顺序：本文 → [预训练原理](02_pretrain.md) → [Pretrain Notebook](../notebooks/01_pretrain.ipynb)。本文以课程使用的 **Dense 64M 配置**为准；上游源码也支持 MoE，但本课程的训练命令设置 `use_moe=0`。

## 1. 模型究竟接收和输出什么

文本先由 tokenizer 编成整数序列 `input_ids`。例如“你好”可能被拆成一个或多个 token；token 不等于汉字，也不固定等于一个词。模型接收形状为 `[batch, seq_len]` 的 token ID，返回每个位置对整个词表的打分 `logits`，形状为 `[batch, seq_len, vocab_size]`。对最后一个位置的 logits 做 softmax，就得到下一个 token 的概率分布。

MiniMind 是**仅解码器（decoder-only）、自回归**语言模型。第 `t` 个位置只能使用自己与更早位置的信息，不能看未来 token；推理时把新生成的 token 接到末尾，再预测下一个。训练时整段序列可并行计算，但因果掩码仍保证每个位置看不到未来。

```text
文本 → tokenizer → token embedding → 8 个 Transformer block → RMSNorm
     → 线性输出层 → 每个位置的词表 logits → 下一个 token 的概率
```

## 2. 本课程默认配置

下表来自上游 `MiniMindConfig` 的默认值和课程 Notebook 的启动参数；`head_dim = hidden_size / num_attention_heads`，中间层宽度由源码公式取整。

| 项目 | 本课程值 | 作用 |
| --- | ---: | --- |
| 词表大小 | 6,400 | 每个位置输出 6,400 个 token 的打分 |
| 隐藏维度 | 768 | 每个 token 在主干网络中的表示宽度 |
| Transformer block | 8 | 顺序堆叠的层数 |
| Query / KV 头数 | 8 / 4 | 分组查询注意力（GQA）；两个 Query 头共享一组 KV |
| 每头维度 | 96 | `768 / 8` |
| 前馈中间维度 | 2,432 | `ceil(768 × π / 64) × 64` |
| 位置编码 | RoPE | 在 Query、Key 上注入相对位置信息 |
| 输出权重 | 与 embedding 共享 | 减少独立参数量 |
| 课程训练长度 | 340 token | `max_seq_len`；长样本截断，短样本补齐 |

源码配置中的 `max_position_embeddings=32768` 是位置编码缓存的上限，**不等于**这节课已经在 32K 长度上训练或验证。模型能否可靠处理长上下文，仍取决于训练数据的长度分布、训练方法和实际评测。

## 3. 一个 Transformer block 做什么

每层先用 RMSNorm 归一化，再做因果自注意力，与输入相加；随后再做 RMSNorm 和前馈网络，并再次加残差。简写为：

```text
h₁ = h + Attention(RMSNorm(h))
h₂ = h₁ + FFN(RMSNorm(h₁))
```

**自注意力**将表示投影为 Query、Key、Value。每个 Query 与允许看到的 Key 计算相似度，softmax 后对 Value 加权求和。因果掩码把未来位置的注意力分数屏蔽掉。这里有 8 个 Query 头、4 个 KV 头，所以 KV 可以被多个 Query 头复用；在生成时，这通常减少 KV cache 的占用。实现还对 Query、Key 分别做 RMSNorm，并在它们上应用 RoPE。

**前馈网络**在每个位置独立变换表示。MiniMind 使用门控形式：`down_proj(SiLU(gate_proj(x)) × up_proj(x))`。注意力负责在位置之间交换信息，前馈网络负责对每个位置的特征进行非线性变换。

**残差连接**把每个子层的输入加回输出，使深层网络更容易优化。最后一层后还有 RMSNorm，随后输出层把隐藏向量映射到词表 logits。Embedding 与输出层共享权重；因此直接把 `state_dict` 中各张量的元素数相加，可能重复计算同一组共享参数。课程训练日志给出的独立参数量约 **63.91M**，Notebook 对保存字典逐项求和约 **68.83M**，两者口径不同。

## 4. 训练与生成的区别

训练时提供整段已知文本，模型在每个位置预测下一 token，再与真实 token 比较。这叫 *teacher forcing*。生成时没有真实的后续文本，只能逐步采样或选取预测 token；KV cache 可复用先前 token 的 Key、Value 来减少重复计算。

模型结构只规定了“怎样计算概率”。模型是否会遵循指令、回答是否可靠，还取决于数据和训练阶段。预训练首先学习语言建模；之后的 SFT 才用带角色的对话样本专门训练指令回复。

## 源码对照与思考题

- [`model/model_minimind.py`](https://github.com/jingyaogong/minimind/blob/858008eb98da/model/model_minimind.py)：`MiniMindConfig`、`Attention`、`FeedForward`、`MiniMindBlock`、`MiniMindForCausalLM`。
- [`dataset/lm_dataset.py`](https://github.com/jingyaogong/minimind/blob/858008eb98da/dataset/lm_dataset.py)：文本如何变成 token ID 与 label。
- 思考：若 Query 头数保持 8、KV 头数从 4 改为 8，生成时 KV cache 通常会怎样变化？
- 思考：为什么配置写着 32768，也不能直接宣称模型有可靠的 32K 上下文能力？
