import ssl
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms

ssl._create_default_https_context = ssl._create_unverified_context
data_path="D:\\SVN_WORK\\cwf_pytotch\\dataset"

#返回不经过变换的数据集
#cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
#cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

#返回将图片变换成tensor的数据集
cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.ToTensor())
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.ToTensor())

print(len(cifar10))
print(len(cifar10_val))

img,label=cifar10[5000]
print(cifar10.classes[label])

#查看图片(不经过变换的数据集才能直接看图片)
#plt.imshow(img)
#plt.show()


plt.imshow(img.permute(1,2,0))#将张量的轴从C*H*W转为H*W*C
plt.show()

