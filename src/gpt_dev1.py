# data
# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

import torch
from torch import nn, optim
import torch.nn.functional as F

CONTEXT_SIZE = 2  # 8
EMBEDDING_SIZE = 10  # 16

text = ""
with open('../data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))  # 1115394

vocab = sorted(list(set(text)))
vocab_size = len(vocab)

stoi = {s: i for i, s in enumerate(vocab)}
itoi = {i: s for i, s in enumerate(vocab)}

encode = lambda s: [stoi[c] for c in s]  # given string as input and generate array of integers
decode = lambda l: ''.join([itoi[i] for i in l])  # given an array of integer and generate a strings

trigrams = [([text[i], text[i + 1]], text[i + 2]) for i in range(len(text) - 2)]
print(trigrams[:3])


class MyBigramModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, context_size):
        super(MyBigramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)

        self.linear1 = nn.Linear(context_size * embedding_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, idx):
        out = self.embeddings(idx).view((1, -1))
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def generate(self, idx, max_no_tokens):
        for _ in range(max_no_tokens):
            # get the predictions
            probs = self(idx)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (1, T+1)
        return idx


losses = []
loss_function = nn.NLLLoss()
my_model = MyBigramModel(vocab_size, EMBEDDING_SIZE, CONTEXT_SIZE)
optimizer = optim.SGD(my_model.parameters(), lr=0.001)

for epoch in range(1):
    total_loss = 0
    for context, target in trigrams:
        context_idx = torch.tensor([stoi[w] for w in context], dtype=torch.long)
        my_model.zero_grad()
        log_probs = my_model(context_idx)
        loss = loss_function(log_probs, torch.tensor([stoi[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)

print(losses)


def print_generated_tokens(max_tokens=100):
    context_ix = torch.zeros((1, 1), dtype=torch.long)  # (1, 1)
    gen_idx = my_model.generate(idx=context_ix, max_no_tokens=max_tokens)  # (1, 101)
    gen_idx = gen_idx[0].tolist()  # array of 101
    print(decode(gen_idx))


print_generated_tokens(100)




