# @title Hopfield memory
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hopfield(nn.Module):
    def __init__(self, in_dim, d_model=None, n_heads=1, beta=8):
        super().__init__()
        self.beta = beta # β
        d_model = d_model or in_dim
        self.n_heads = n_heads
        self.k = nn.Linear(in_dim, d_model, bias=False)
        self.v = nn.Linear(d_model, in_dim, bias=False)

    def forward(self, query): # [...,in]
        weights = F.softmax(self.beta * self.k(query).unflatten(-1, (self.n_heads,-1)), dim=-1) # [...,h,d]
        return self.v(weights.flatten(-2))

# Softmax(q @ k.T) @ v

# mamba(ssd):
# h = Ah + Bx : A*h + x@B = 1/ds*ds + d1@1s = ds
# y = Ch + Dx : h@C + D*x = ds@s1 + 1/d1*d1 = d1

# b,d,D,h = 2,2,16,2
b,d,D,h = 4,8,64,2
# b,d,D,h = 100,16,256,8
x = torch.randn(b,d)
print(x)
model = Hopfield(d, D, h, 8)
optim = torch.optim.AdamW(model.parameters(), lr=3e-1)

for i in range(100):
    loss = F.mse_loss(model(x), x)
    # print(loss)
    optim.zero_grad()
    loss.backward()
    optim.step()
print(loss)

s_=x[:].clone()
# s_[:,:-2]=0
s_[:,d//2:]=0
# s_=s_+torch.randn_like(s_)*.1
# print(s_)
out = model(s_)
for i in range(50):
    # out[:,d//2:] = s_[:,d//2:]
    out = model(out)
print(out)
# er=(x-out).detach().abs()
er=(x-out).detach()**2
print(er.max(), er.mean())
