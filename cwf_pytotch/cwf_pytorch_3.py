import ssl
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms


softmax= torch.nn.Softmax(dim=1)#这里dim可选0,1,即按入力张量哪个维度算softmax
x=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
print(softmax(x))