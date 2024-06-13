import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)

a1 = torch.ones(3, 3)  # a 3 by 3 matrix with all 1
print('a1')
print(a1)
a2 = torch.tril(a1)  # means make matrix triangular by putting 0 in upper part of the diagonal
print('a2')
print(a2)
a3 = torch.sum(a2, dim=1, keepdim=True)  # sum of all data in each row
print('a3')
print(a3)
a = a2 / a3  # each positioned element of a2 is divided by the particular positioned element in a3
print('a')
print(a)
b = torch.randint(0, 10, (3, 2)).float()  # a 3 by 2 matrix
print('b')
print(b)
# rules of matrix multiplication
# - 1. A of (m x n) * B of (n * p) = AB of (m x p)
# - 2. the column of A must be equal to the row of B
c = a @ b  # (3, 2) - means the columns of matrix a must be equal to the rows of matrix b
print('c')
print(c)

# consider the following toy example:
torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
print(x.shape)
print(x)

# we want x[b,t] = mean_{i <= t} x[b,i]
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t+1]
        avg = torch.mean(xprev, 0)
        xbow[b, t] = avg


# version 2: using matrix multiply for a weighted aggregation
print('version 2: using matrix multiply for a weighted aggregation')
wei = torch.tril(torch.ones(T, T))
wei_sm = wei.sum(dim=1, keepdim=True)
wei = wei / wei_sm
# in (B, T, C) shaped, matrix multiplication are done with the last two (T, C) by keeping B aside
xbow2 = wei @ x  # (B, T, T) * (B, T, C) ---> (B, T, C), i.e., (T, T1) * (T1, C) ---> (T, C) here T1=T
r = torch.allclose(xbow, xbow2)
print(r)

# version 3: use Softmax
print('version 3: use Softmax')
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x  # (B, T, T) * (B, T, C) ---> (B, T, C), i.e., (T, T1) * (T1,C) ---> (T, C) here T1=T
r = torch.allclose(xbow, xbow3)
print(r)

# version 4: self-attention!
print('# version 4: self-attention!')
torch.manual_seed(1337)
B, T, C = 4, 8, 32  # batch, time, channels or batch, block_size, vocab_size
x = torch.randn(B, T, C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)  # (B, T, C) with (C, head_size) --> (B, T, 16)
q = query(x)  # (B, T, C) with (C, head_size) --> (B, T, 16)
wei = q @ k.transpose(-2, -1)  # (B, T, 16) * (B, 16, T) --> (B, T, T)
tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
v = value(x)  # (B, T, C) with (C, head_size) --> (B, T, 16)
out = wei @ v  # (B, T, T) * (B, T, 16) --> (B, T, 16)
print(out.shape)
print(wei[0])

torch.manual_seed(1337)
k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)
wei = q @ k.transpose(-2, -1) * head_size**-0.5
print(k.var())
print(q.var())
print(wei.var())

out = torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)
print(out)

out = torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]) * 8, dim=-1) # gets too peaky, converges to one-hot
print(out)


class LayerdNorm1d: # (used to be BatchNorm1d)
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(dim=1, keepdim=True)  # batch mean
        xvar = x.var(dim=1, keepdim=True)  # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


torch.manual_seed(1337)
module = LayerdNorm1d(100)
x = torch.randn(32, 100)  # batch size 32 of 100-dimensional vectors
x = module(x)
print(x.shape)

n_embd = 66
n_head = 4
head_size = n_embd // n_head
print(head_size)



