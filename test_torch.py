import time
import torch
import torch.utils.data as data

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from tqdm import tqdm

from model import Net

mnist_loader = data.DataLoader(
    MNIST('/Users/makora/data/mnist', train=False, download=False,
          transform=transforms.Compose([
              transforms.ToTensor(),
              # transforms.Normalize((0.1307,), (0.3081,))
          ])),
    batch_size=1, shuffle=False)

model = Net()
model.load_state_dict(torch.load('./mnist_cnn.pth.tar'))
model.eval()

correct = 0
pbar = tqdm(mnist_loader)
average_time = 0
with torch.no_grad():
    for image, target in pbar:
        start = time.time()
        output = model(image)
        average_time += (time.time() - start)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

print('Test set: Accuracy: {}/{} ({:.0f}%), Time: {}'.format(
        correct,
        len(mnist_loader.dataset),
        100. * correct / len(mnist_loader.dataset),
        average_time / len(mnist_loader.dataset)))
