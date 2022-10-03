
import torch
from torchvision.prototype.features import Image


x = torch.rand(1, 3, 32, 32)
print("-- Image(x)")
y = Image(x)

print("-")
print(y.shape)

print("-- add")
z = y + y

print("-- Image.new_like(y, z)")
u = Image.new_like(y, z)

print("--")
print(u.image_size)

print("-- torch.Tensor(u)")
t = torch.Tensor(u)


print("-- add inplace")
u.add_(1)
