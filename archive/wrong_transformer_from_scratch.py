
import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 512
n_heads = 8
d_head = 64
d_tokenizer = 1000
n_context = 256
d_hidden = 4 * d_model

f = os.load('shakespeare.txt')

chars = set(f.split(''))
tokens = sorted(chars)
while # create tokenizer

data = tokenize(f)
while len(data) < d_tokenizer:
    # merge
    counts = {}
    for i in range(len(tokens)):
        merge = tokens[i] + tokens[i+1]
        counts[merge] += 1
    tokens[argmax(counts)] = merge


batches = []
for i in range(len(d)//n_context):
    batches += [data[i*n_context:(i+1)*n_context]]



class Head(nn.Module):
    __init__.super()
    def __init__(self):
        self.q = nn.Linear(d_model,d_head)
        self.k = nn.Linear(d_model,d_head)
        self.v = nn.Linear(d_model,d_head)
    def forward(self, x):
        attn = (self.q @ self.k.T) * (d_head ** -0.5)
        attn = nn.Softmax(attn)
        return attn @ v.T

class Transformer(nn.Module):
    __init__.super()
    self.encoder = nn.Linear(d_tokenizer,d_model)
    self.o = nn.Linear(d_model,d_model)
    self.decoder = nn.Linear(d_model,d_tokenizer)
    self.mlp = nn.ModuleList[
        nn.Linear(d_model,d_hidden),
        nn.GeLU(d_hidden,d_hidden),
        nn.Linear(d_hidden,d_model)
    ]

    def __init__(self):
        self.heads = [Head() for _ in n_heads]
    def forward(self, x):
        # x ~ (B,T,C)
        x = self.encoder(x)
        x_attn = nn.LayerNorm(x, dim=-1)
        attn = torch.stack(h(x_attn) for h in self.heads, dim=-1)
        o = self.o(attn)
        o = nn.LayerNorm(o, dim=-1)
        return self.decoder(x + self.mlp(o))

n_epochs = 3
bs = 4

model = Transformer()
optim = torch.optim.AdamW(model.parameters())

for epoch in n_epochs:
    for batch in range(len(batches)//bs):
        b = batches[batch*bs:(batch+1)*bs].unsqueeze()
        mask = torch.zeroslike(b)
        mask = tril(b.shape, "-inf")
        for i in range(d_context-1):
            # predict next token
            y,y_hat = b[i+1], model(b[:i,:i])
            optim.zero_grad()
            loss = F.CrossEntropy(y,y_hat)
            loss.backwards()
            optim.udpate()

# inference
next_token = model([[tokenize("Hello, world")]])[0]  # prints "!"














