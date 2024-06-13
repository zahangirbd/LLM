# data
# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

import torch
from torch import nn, optim
import torch.nn.functional as F

text = ''
with open('../data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
vocab = sorted(list(set(text)))
print(''.join(vocab))
vocab_size = len(vocab)

# create a mapping from characters to integers
stoi = {s: i for i, s in enumerate(vocab)}
itos = {i: s for i, s in enumerate(vocab)}

# create encode and decode
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode('hi there'))
print(decode(encode('hi there')))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# preparing batch
torch.manual_seed(1337)
batch_size = 4 #16
block_size = 8 #32
embed_size = 16 #64
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i: i + block_size] for i in ix])
    y = torch.stack([d[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        self.ln_f = nn.LayerNorm(embed_size)  # layer norm
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = token_emb + pos_emb  # (B,T,C)
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


xb, yb = get_batch('train')
my_model = BigramLanguageModel(vocab_size, embed_size=embed_size, block_size=block_size)
m = my_model.to(device)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)


def print_generated_tokens(max_tokens=100):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # (1, 1)
    gen_idx = m.generate(idx=context, max_no_tokens=max_tokens)  # (1, 101)
    gen_idx = gen_idx[0].tolist()  # array of 101
    print(decode(gen_idx))


print_generated_tokens(100)


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

print_generated_tokens(500)
