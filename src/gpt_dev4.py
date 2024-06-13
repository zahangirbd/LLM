# data
# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

import torch
from torch import nn, optim
import torch.nn.functional as F

text = ''
with open('../data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab = sorted(list(set(text)))
vocab_size = len(vocab)

stoi = {s: i for i, s in enumerate(vocab)}
itos = {i: s for i, s in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]  # gets a string and generate array of integers
decode = lambda l: ''.join([itos[i] for i in l]) # gets array of integer and generate a string

# generate data tensor first from the input text
data = torch.tensor(encode(text), dtype=torch.long)

# separating train and test data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
# print(train_data[:64])


# parameters --------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4  # 16 # how many independent sequences will we process in parallel?
block_size = 8  # 32 # what is the maximum context length for predictions?
n_embd = 16  # 64
n_head = 2  # 4
dropout = 0.0
n_layer = 2  # 4
# -----------------
torch.manual_seed(1337)


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i: i + block_size] for i in ix])
    y = torch.stack([d[i + 1: i + block_size + 1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y


# xb, yb = get_batch('train')
# print(xb)
# print(yb)


class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super(Head, self).__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, : T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttn(nn.Module):
    def __init__(self, n_embd, num_heads, head_size):
        super(MultiHeadAttn, self).__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super(Block, self).__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttn(n_embd, n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size, n_head):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[Block(embed_size, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embed_size)  # layer norm
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = token_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_no_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_no_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]  # its like sliding bloc_size token in each iteration
            # get the predictions
            logits, loss = self(idx_cond)  # (B,T,C)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def print_generated_tokens(in_model, max_tokens=100):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # (1, 1)
    gen_idx = in_model.generate(idx=context, max_no_tokens=max_tokens)  # (1, 101)
    gen_idx = gen_idx[0].tolist()  # array of 101
    print(decode(gen_idx))


my_model = BigramLanguageModel(vocab_size, embed_size=n_embd, block_size=block_size, n_head=n_head)
m = my_model.to(device)

# create a PyTorch optimizer
optimizer = optim.AdamW(m.parameters(), lr=1e-3)

for epoch in range(5000):

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print_generated_tokens(m, 500)
