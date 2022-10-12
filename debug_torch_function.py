import torch
from torchvision.prototype.features import Image


x = torch.rand(1, 3, 32, 32)
print("-- Image(x)")
y = Image(x)

print("-")
print(y.shape)

print("-- add")
z = y + y

print("-- Image.wrap_like(y, z)")
u = Image.wrap_like(y, z)

print("--")
print(u.image_size)

print("-- torch.Tensor(u)")
t = torch.Tensor(u)


print("-- add inplace")
u.add_(1)


x = torch.rand(1, 3, 32, 32)
print("-- Image(x)")
y = Image(x)

print("-- resize inplace")
z = y.resize_((1, 3, 12, 12))

print(y.shape, type(y))
print(z.shape, type(z))
