import torch
import numpy as np
import pandas as pd

t1 = torch.tensor("4")
print(t1.shape)

x = torch.tensor(3.)
w = torch.tensor(4., requires_grad = True)
b = torch.tensor(5., requires_grad = True)

print(x,w,b)

y = w*x + b

print(y)

y.backward()
# derivative of y w.r.t to all vriables for which requires_grad is True
print("function is y = w*x + b, the equation of line")
print("dy/dx", x.grad)
print("dy/dw", w.grad)
print("dy/db", b.grad)
