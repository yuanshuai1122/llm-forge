import os
import requests
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

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

# 加载训练数据
if not os.path.exists('data/sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    with open('data/sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)  # 下载并保存数据

with open('data/sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()  # 读取文本数据

# 使用 TikToken（与 GPT3 相同）对源文本进行分词
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)  # 将文本编码为标记
max_token_value = max(tokenized_text) + 1  # 分词后的最大值
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)  # 将分词文本转为张量

# 划分训练集和验证集
split_idx = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:split_idx]  # 训练数据
val_data = tokenized_text[split_idx:]  # 验证数据

# 定义前馈神经网络
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),  # 线性层
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),  # 线性层
            nn.Dropout(dropout),  # Dropout 层
        )

    def forward(self, x):
        return self.ffn(x)  # 前向传播

# 定义缩放点积注意力
class Attention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout

        self.key_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)  # 键层
        self.query_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)  # 查询层
        self.value_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)  # 值层
        self.register_buffer('tril', torch.tril(
            torch.ones((self.context_length, self.context_length))))  # 下三角掩码
        self.dropout_layer = nn.Dropout(self.dropout)  # Dropout 层

    def forward(self, x):
        B, T, C = x.shape  # 批次大小，时间步（当前上下文长度），通道数（维度）
        assert T <= self.context_length
        assert C == self.d_model
        q = self.query_layer(x)  # 查询
        k = self.key_layer(x)  # 键
        v = self.value_layer(x)  # 值

        # 缩放点积注意力：Q @ K^T / sqrt(d_k)
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 应用掩码注意力
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(input=weights, dim=-1)  # 应用 softmax
        weights = self.dropout_layer(weights)  # 应用 dropout

        # 应用点积注意力：weights @ V
        out = weights @ v
        return out

# 定义多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        self.heads = nn.ModuleList([Attention(head_size=self.head_size) for _ in range(self.num_heads)])  # 多头注意力
        self.projection_layer = nn.Linear(in_features=self.d_model, out_features=self.d_model)  # 投影层
        self.dropout_layer = nn.Dropout(dropout)  # Dropout 层

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # 拼接多头输出
        out = self.projection_layer(out)  # 投影
        out = self.dropout_layer(out)  # 应用 dropout
        return out

# 定义 Transformer 块
class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model  # 头大小应可被 d_model 整除
        self.num_heads = num_heads
        self.dropout = dropout

        self.multi_head_attention_layer = MultiHeadAttention(head_size=self.head_size)  # 多头注意力层
        self.feed_forward_layer = FeedForward()  # 前馈层
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.d_model)  # 层归一化
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.d_model)  # 层归一化

    def forward(self, x):
        # 注意：操作顺序与原始 Transformer 论文不同
        # 此处顺序为：层归一化 -> 多头注意力 -> 层归一化 -> 前馈
        x = x + self.multi_head_attention_layer(self.layer_norm_1(x))  # 残差连接
        x = x + self.feed_forward_layer(self.layer_norm_2(x))  # 残差连接
        return x

# 定义 Transformer 语言模型
class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_value
        # 设置标记嵌入查找表
        self.token_embedding_lookup_table = nn.Embedding(num_embeddings=self.max_token_value + 1, embedding_dim=self.d_model)

        # 运行所有 Transformer 块
        # 与原始论文不同，此处我们在所有块后添加了最终层归一化
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_blocks)] +
                [nn.LayerNorm(self.d_model)]
        ))
        self.language_model_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=self.max_token_value)  # 输出线性层

    def forward(self, idx, targets=None):
        B, T = idx.shape
        """
        # 设置位置嵌入查找表
        # 遵循原始 Transformer 论文的方法（正弦和余弦函数）
        """
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)  # 正弦位置编码
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)  # 余弦位置编码
        # 将 position_encoding_lookup_table 从 (context_length, d_model) 更改为 (T, d_model)
        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        x = self.token_embedding_lookup_table(idx) + position_embedding  # 标记嵌入与位置嵌入相加
        x = self.transformer_blocks(x)  # 通过 Transformer 块
        # “logits”是模型在应用 softmax 前的输出值
        logits = self.language_model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)  # 计算交叉熵损失
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx 是当前上下文中 (B,T) 的索引数组
        for _ in range(max_new_tokens):
            # 裁剪 idx 以匹配位置嵌入表的最大大小
            idx_crop = idx[:, -self.context_length:]
            # 获取预测
            logits, loss = self(idx_crop)
            # 从 logits 中获取最后一个时间步，logits 维度为 (B,T,C)
            logits_last_timestep = logits[:, -1, :]
            # 应用 softmax 获取概率
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # 从概率分布中采样
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # 将采样得到的索引 idx_next 追加到 idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 初始化模型
model = TransformerLanguageModel()
model = model.to(device)  # 将模型移到设备（GPU 或 CPU）

# 获取输入嵌入批次
def get_batch(split: str):
    data = train_data if split == 'train' else val_data  # 选择训练或验证数据
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))  # 随机选择索引
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)  # 输入批次
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)  # 目标批次
    return x, y

# 计算损失
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # 设置模型为评估模式
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()  # 计算平均损失
    model.train()  # 恢复训练模式
    return out

# 使用 AdamW 优化器
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))  # 打印训练和验证损失

    xb, yb = get_batch('train')  # 获取训练批次
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)  # 清空梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

# 保存模型状态字典
torch.save(model.state_dict(), 'model-ckpt.pt')

# 生成
model.eval()  # 设置模型为评估模式
start = 'The salesperson'  # 起始文本
start_ids = encoding.encode(start)  # 编码起始文本
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])  # 转为张量
y = model.generate(x, max_new_tokens=100)  # 生成文本
print('---------------')
print(encoding.decode(y[0].tolist()))  # 解码并打印生成结果
print('---------------')