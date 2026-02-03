# @title Hopfield
import torch
import torch.nn as nn

class Hopfield(nn.Module):
    def __init__(self, patterns, beta=8): # 8?
        super().__init__()
        self.patterns = patterns # X [N,d]
        self.beta = beta # β higher -> sharper
        self.num_patterns = patterns.shape[1]
        self.max_pattern_norm = torch.max(torch.norm(patterns, dim=-1))  # For energy constant

    def energy(self, state): # ξ xi
        # E = 1/2 * ||ξ||^2 - 1/β * logsumexp(β * patterns^T @ ξ) + 1/β * log(num_patterns) + 1/2 * max_pattern_norm^2
        lse = -1/self.beta*torch.logsumexp(self.beta*self.patterns.T@state, dim=0)
        quadratic = .5*state.T@state
        constant = 1/self.beta*torch.log(torch.tensor(self.num_patterns).float()) + .5*self.max_pattern_norm**2
        energy = lse + quadratic + constant
        return energy

    def forward(self, state): # [n,d] Ξ Xi update rule by Concave-Convex-Procedure (CCCP) # ξnew = X @ softmax(β * X^T @ ξ)
        new_state = torch.softmax(self.beta * state @ self.patterns.T, dim=-1) @ self.patterns # (nd @ dN) @ Nd
        return new_state

N,d = 3,8
pattern = torch.randn(N,d)
print(pattern)
model = Hopfield(pattern, 8)
s_=pattern[:2].clone()
s_[:,d//2:]=0
# print(s_)
out = model(s_)
print(out)
# er=(pattern-out).detach()**2
er=(pattern[:2]-out).detach()**2
print(er.max(), er.mean())
