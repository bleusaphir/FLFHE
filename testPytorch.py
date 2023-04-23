from __future__ import print_function
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from typing import List, Tuple, Dict, Optional, Callable
from Pyfhel import Pyfhel

HE = Pyfhel()           # Creating empty Pyfhel object
ckks_params = {
    'scheme': 'CKKS',   # can also be 'ckks'
    'n': 2**15,         # Polynomial modulus degree. For CKKS, n/2 values can be
                        #  encoded in a single ciphertext. 
                        #  Typ. 2^D for D in [10, 16]
    'scale': 2**30,     # All the encodings will use it for float->fixed point
                        #  conversion: x_fix = round(x_float * scale)
                        #  You can use this as default scale or use a different
                        #  scale on each operation (set in HE.encryptFrac)
    'qi_sizes': [60, 30, 30, 30, 60] # Number of bits of each prime in the chain. 
                        # Intermediate values should be  close to log2(scale)
                        # for each operation, to have small rounding errors.
}
HE.contextGen(**ckks_params)  # Generate context for bfv scheme
HE.keyGen()             # Key Generation: generates a pair of public/secret keys
HE.rotateKeyGen()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def get_parameters2(net) -> List[np.ndarray]:
    #Renvoie une liste avec les poids puis les biais de chaque couche
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def get_EncryptedParameters2(net) -> List[np.ndarray]:
    #Renvoie une liste avec les poids puis les biais de chaque couche
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def encrypt(weights):
    #Fonction récursive pour encrypter chaque node ? 
    #---- Shape (1,3,3) --> 1 array 3*3
    #---- Shape (3,3) --> encrypter chaque array de la liste
    encryptedWeights = []
    for w in weights:
        if len(w.shape) > 2 :
            encryptedWeights.append(encrypt(w))
        else:
            encryptedWeights.append([HE.encrypt(node) for node in w])
        
    return encryptedWeights

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def aggregate(N,aW):

    aggregateLayers = []
    if isinstance(aW[0][0], list):
        aggregateLayers.append(aggregation(N,len(aW[0]),list(zip(*aW))))
    else:
        aggregateLayers.append([sum([aW[k][n] for n in range(N)]) for k in range(len(aW))])

    return aggregateLayers




def aggregation(N,nLayers,weights):
    moy = []
    #Dans ce cas-ci : 7
    for l in weights :
        aW = []
    #tous les Models ont la même structure
        if isinstance(l[0][0], list):
            moy.append(aggregation(N,len(l[0]), list(zip(*l))))

        else:
            moy.append([sum([l[n][k] for n in range(N)]) for k in range(len(l[0]))])
    return moy



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        param = get_parameters2(model)
        encryptedWeights = encrypt(param)
        otherW = encryptedWeights.copy()
        weights = []
        weights.append(otherW)
        weights.append(encryptedWeights)
        weights = list(zip(*weights))
        N = 2 #[ModelW1, ModelW2,..., ModelWn] avec N = nClients
        nLayers = len(weights)
        moy = aggregation(N, nLayers,weights)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()