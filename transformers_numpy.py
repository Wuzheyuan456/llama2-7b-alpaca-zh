"""
nano-transformer.py  ——  手推 Transformer（Encoder+Decoder）<200行
依赖：numpy 1.20+
"""
import numpy as np

# ------------------ 工具 ------------------
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=axis, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ------------------ 基础层 ------------------
class Embedding:
    def __init__(self, vocab_size, d_model):
        # 初始化权重：[vocab_size, d_model]
        self.W = np.random.randn(vocab_size, d_model) / np.sqrt(vocab_size)

    def __call__(self, x):
        # x: [B, T] of integers (token IDs)
        # 返回: [B, T, d_model]
        return self.W[x]  # 关键：用索引取行，不是矩阵乘法！


class Linear:
    def __init__(self, inp, out):          # Xavier
        scale = np.sqrt(1 / inp)
        self.W = np.random.randn(inp, out) * scale
        self.b = np.zeros(out)
    def __call__(self, x):
        return x @ self.W + self.b

class LayerNorm:
    def __init__(self, d, eps=1e-6):
        self.gamma = np.ones(d) # 可学习的缩放参数 (scale)
        self.beta  = np.zeros(d)# 可学习的偏移参数 (shift)
        self.eps = eps# 防止除以0的小常数
    def __call__(self, x):                 # x:[B,T,d]
        mean = x.mean(-1, keepdims=True) # 沿最后一个维度求均值 → [B, T, 1]
        var  = x.var(-1, keepdims=True) # 沿最后一个维度求方差 → [B, T, 1]
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta#(x - mean) / np.sqrt(var + self.eps) 标准化

# ------------------ Attention ------------------
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.n, self.d = n_heads, d_model // n_heads
        self.qkv = Linear(d_model, 3 * d_model)
        self.o   = Linear(d_model, d_model)
    def __call__(self, x, mask=None):      # x:[B,T,d]
        B, T, _ = x.shape
        qkv = self.qkv(x)                  # [B,T,3d]
        q, k, v = np.split(qkv, 3, -1)
        q = q.reshape(B, T, self.n, self.d).transpose(0,2,1,3)  # [B,n,T,d]
        k = k.reshape(B, T, self.n, self.d).transpose(0,2,1,3)
        v = v.reshape(B, T, self.n, self.d).transpose(0,2,1,3)
        scores = q @ k.transpose(0,1,3,2) / np.sqrt(self.d)      # [B,n,T,T]
        if mask is not None:
            scores = scores + mask        # mask:-inf
        attn = softmax(scores)
        out = attn @ v                     # [B,n,T,d]
        out = out.transpose(0,2,1,3).reshape(B, T, -1)
        return self.o(out)

# ------------------ FeedForward ------------------
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.l1 = Linear(d_model, d_ff)
        self.l2 = Linear(d_ff, d_model)
    def __call__(self, x):
        return self.l2(np.maximum(0, self.l1(x)))   # ReLU

# ------------------ 子层封装 ------------------
class EncoderBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff   = FeedForward(d_model, d_ff)
        self.norm1, self.norm2 = LayerNorm(d_model), LayerNorm(d_model)
    def __call__(self, x):
        x = x + self.attn(self.norm1(x))   # 残差
        return x + self.ff(self.norm2(x))

class DecoderBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.mask_attn = MultiHeadAttention(d_model, n_heads)
        self.enc_dec   = MultiHeadAttention(d_model, n_heads)
        self.ff        = FeedForward(d_model, d_ff)
        self.norm1, self.norm2, self.norm3 = [LayerNorm(d_model) for _ in range(3)]
    def __call__(self, x, enc_out, mask):
        x = x + self.mask_attn(self.norm1(x), mask)
        x = x + self.enc_dec(self.norm2(x))   # Q=dec, K/V=enc
        return x + self.ff(self.norm3(x))

# ------------------ 位置编码 ------------------
def positional_encoding(T, d_model):
    pe = np.zeros((T, d_model)) # 创建 T x d_model 的矩阵
    pos = np.arange(T)[:, None] # 位置索引: [T, 1]
    div = 10000 ** (np.arange(0, d_model, 2) / d_model) # 频率因子: [d_model//2,]  np.arange是NumPy库中的一个函数，用于创建等差数列Q。它接受二个参数:起始值、终止值和步长。
    pe[:, 0::2] = np.sin(pos / div)  # 偶数列用 sin
    pe[:, 1::2] = np.cos(pos / div)  # 奇数列用 cos
    return pe

# ------------------ 整体模型 ------------------
class Transformer:
    def __init__(self, vocab, d_model=128, n_heads=4, d_ff=512, N=1):
        self.embed = Embedding(vocab, d_model) # 词嵌入层
        self.enc   = EncoderBlock(d_model, n_heads, d_ff)  # 编码器（多个 Transformer Block）
        self.dec   = DecoderBlock(d_model, n_heads, d_ff) # 解码器（多个 Transformer Block）
        self.ln    = LayerNorm(d_model) # 最终的层归一化
        self.head  = Linear(d_model, vocab) # 输出头（从 d_model 映射到 vocab_size）
        self.d_model = d_model # 模型维度
    def encode(self, src):
        B, T = src.shape  # src: [B, T]，输入的 token IDs
        x = self.embed(src) + positional_encoding(T, self.d_model)  # 词嵌入: [B, T] -> [B, T, d_model]  # 加上位置编码
        return self.enc(x)    # 经过编码器 → [B, T, d_model]
    def decode(self, tgt, enc_out):
        B, T = tgt.shape
        mask = np.triu(-np.inf * np.ones((T, T)), 1) # # 上三角掩码
        x = self.embed(tgt) + positional_encoding(T, self.d_model)
        x = self.dec(x, enc_out, mask)
        return softmax(self.head(self.ln(x)))
    def __call__(self, src, tgt):
        return self.decode(tgt, self.encode(src))


# ------------------ 测试 ------------------
if __name__ == "__main__":
    vocab = 20
    model = Transformer(vocab, d_model=64, n_heads=4, d_ff=128, N=1)
    src = np.random.randint(0, vocab, (2, 5))   # [B=2, T=5]
    tgt = np.random.randint(0, vocab, (2, 7))
    out = model(src, tgt)                       # [2,7,20]
    print("output shape:", out.shape)