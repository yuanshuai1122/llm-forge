# 动手实践transformer

项目地址: 

```
[transformer-from-scratch](../../projects/transformer-from-scratch)
```

在本文中，我们将从头开始实现一个类似GPT的Transformer。我们将按照我之前文章中描述的步骤，逐步编码每个部分。

ok, 开始吧。

1. ## 环境准备

安装以下依赖（使用uv）
```shell
uv add numpy requests torch tiktoken matplotlib pandas
```

导入依赖

```python
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import math
import tiktoken
import torch
import torch.nn as nn
```

2. ## 设置超参数 

超参数是模型的外部配置，无法在训练过程中从数据中学习。它们在训练开始前设定，对控制训练算法的行为和训练模型的性能起着至关重要的作用。

```toml
# 超参数
batch_size = 4  # 每个训练步骤的批次数量
context_length = 16  # 每个批次的标记块长度
d_model = 64  # 模型标记嵌入的大小
num_blocks = 8  # Transformer 块的数量
num_heads = 4  # 多头注意力中的头数
learning_rate = 1e-3  # 学习率 0.001
dropout = 0.1  # Dropout 比率
max_iters = 5000  # 总训练迭代次数 <- 测试时可改为较小的数字
eval_interval = 50  # 评估频率
eval_iters = 20  # 用于评估的平均迭代次数
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果有 GPU 则使用 GPU
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)  # 设置随机种子
```

3. ## 准备数据集

在我们的示例中，我们将使用一个小数据集进行训练。该数据集是一个包含销售教科书内容的文本文件。我们将使用该文本文件来训练一个能够生成销售文本的语言模型。

```python
# 加载训练数据
if not os.path.exists('data/sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    with open('data/sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)  # 下载并保存数据

with open('data/sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()  # 读取文本数据
```

环境准备完成，下面开始实操。

### 第一步: Tokenization

`https://github.com/openai/tiktoken`

我们将使用 tiktoken 库对数据集进行分词。该库是一个快速且轻量级的分词器，可用于将文本分词成标记（tokens）。

```python
# 使用 TikToken（与 GPT3 相同）对源文本进行分词
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)  # 将文本编码为标记
vocab_size = len(set(tokenized_text)) # 单词的数量为 3,771
max_token_value = max(tokenized_text) + 1  # 分词后的最大值
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)  # 将分词文本转为张量

print(f"Tokenized text size: {len(tokenized_text)}")
print(f"Vocabulary size: {vocab_size}")
print(f"The maximum value in the tokenized text is: {max_token_value}")
```

打印输出

```
Tokenized text size: 77919
Vocabulary size: 3771
The maximum value in the tokenized text is: 100070
```

### 第二步：Word Embedding

我们将数据集分为训练集和验证集。训练集将用于训练模型，验证集将用于评估模型的性能。、

```python
# 划分训练集和验证集
split_idx = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:split_idx]  # 训练数据
val_data = tokenized_text[split_idx:]  # 验证数据

# 为训练批次准备数据
data = train_data
idxes = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
x_batch = torch.stack([data[idx:idx + context_length] for idx in idxes])
y_batch = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxes])
print(x_batch.shape, x_batch.shape)
```

打印输出（训练输入 x 和 y 的形状）：

```
torch.Size([4, 16]) torch.Size([4, 16])
```

### 第三步：Positional Encoding﻿