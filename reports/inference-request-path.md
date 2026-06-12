# LLM Inference Request Path

token是大模型推理的原子单元

## 1. 请求路径

tokenizer -> prefill -> KV Cache write -> decode loop -> sampling -> detokenizer -> stream output

用户输入 prompt 
-> tokenizer 编码成 input_ids
-> token embedding + position embeeding
-> prefill
-> 将每层 prompt tokens 的 K、V 写入 KV Cache
-> 得到 first token logits
-> sampling 选出第一个 token 
-> decode loop：
   上一步 token id
   -> embedding
   -> 单 token forward
   -> 每层读取 KV Cache
   -> 追加当前 token 的 K/V 到 KV Cache
   -> 得到 next token logits
   -> sampling 选出 next token
-> detokenizer
-> stream / final output

## 2. Prefill vs Decode

| 阶段      | 输入                        | 输出                            | 计算特征            | 主要瓶颈               |
| ------- | ------------------------- | ----------------------------- | --------------- | ------------------ |
| Prefill | prompt tokens             | first token logits + KV Cache | 并行度高，矩阵-矩阵计算    | 计算 / attention     |
| Decode  | previous token + KV Cache | next token logits             | 逐 token，矩阵-向量倾向 | 显存带宽 / KV Cache 读取 |

## 3. TTFT

TTFT = 排队时间 + prefill 时间 + 第一次 sampling / detokenize 时间

## 4. TPOT

TPOT = 每个 decode step 的平均耗时，主要受 batch size、sequence length、KV Cache 读取、scheduler 开销影响。

## 5. 问题记录

### 5.1 为什么 decode 更 memory-bound？

decode 的计算是 matrix-vector，耗时取决于喂 GPU 数据（权重、KV Cache、激活）的效率

### 5.2 KV Cache 在每层存什么？

每层存的是的 token 隐向量对 K 和 V 投影结果。

这用于 Attention 的计算：

Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V

其中历史 token 的 K/V 是固定的，Cache 下来以免重复计算。可通过计算流程理解：

1）prefill 阶段的 token 经过 embedding 生成隐向量 x 矩阵

2）x 分别与 Wq Wk Wv 做线性变换，得到 Q、K、V，将 K、V 放入 KV Cache 中

3）计算 attention，得到 next token 的概率分布

4）进入 decode 循环：

a）计算 next token 的 Q_new、K_new、V_new，将 K_new、V_new append 到 KV Cache

b）用 Q_new 计算 attention，得到 next token 的概率分布

### 5.3 长 prompt 更影响 TTFT 还是 TPOT？

prompt 作为 prefill 阶段的输入，输入长度会影响 TTFT

### 5.4 sampling是什么，起到什么作用？

decode 后产生 logit 分布，sampling 策略决定怎么选取 next token，下面是常见的一些 sampling 策略

| Sampling           | 采样策略                      | 特点               |
| ------------------ | ------------------------- | ---------------- |
| greedy             | 每次选取概率最高的                 | 稳定，确定；容易单调       |
| temperature        | 调节采样随机性                   | 越小，分布越尖锐越大，分布越平 |
| top-k              | 只从概率最高的 k 个采样             | 候选集大小不变          |
| top-p              | 只从累计概率达到 p 的那一小批 token 采样 | 候选集大小会变          |
| repetition penalty | 降低已经出现过的 token 再次出现的概率    | 减少复读             |
