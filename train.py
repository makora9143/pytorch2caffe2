import argparse

import torch
import torch.onnx
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from tqdm import tqdm

from model import Net

def train(args, model, loader, optimizer, epoch):
    model.train()
    losses = 0
    pbar = tqdm(loader)
    pbar.set_description('Epoch {}/{}'.format(epoch, args.epochs))
    for idx, (data, target) in enumerate(pbar):
        optimizer.zero_grad()
        preds = model(data)
        loss = F.nll_loss(preds, target)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
        losses += loss.item()
    print('Loss:{}'.format(losses/len(pbar)))

def test(args, model, loader):
    model.eval()
    losses = 0
    correct = 0
    pbar = tqdm(loader)
    pbar.set_description('Test')
    with torch.no_grad():
        for idx, (data, target) in enumerate(pbar):
            output = model(data)
            losses += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    losses /= len(loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            losses, correct, len(loader.dataset),
            100. * correct / len(loader.dataset)))





parser = argparse.ArgumentParser(description='CNN Training')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')

args = parser.parse_args()

torch.manual_seed(1234)

train_loader = data.DataLoader(
    MNIST('/Users/makora/data/mnist', train=True, download=False,
          transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
          ])),
    batch_size=args.batch_size, shuffle=True)

test_loader = data.DataLoader(
    MNIST('/Users/makora/data/mnist', train=False, download=False,
          transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
          ])),
    batch_size=args.batch_size, shuffle=False)

model = Net()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

for epoch in range(1, args.epochs+1):
    train(args, model, train_loader, optimizer, epoch)
    test(args, model, test_loader)

torch.save(model.state_dict(), 'mnist_cnn.pth.tar')

dummy_input = torch.randn(args.batch_size, 1, 28, 28, requires_grad=True)
torch.onnx.export(model, dummy_input, 'mnist_cnn.onnx', verbose=True)

