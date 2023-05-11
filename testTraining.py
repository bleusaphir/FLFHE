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
import torchmetrics
import time

CKKSN = 2**15
HE = Pyfhel()           # Creating empty Pyfhel object
ckks_params = {
    'scheme': 'CKKS',   # can also be 'ckks'
    'n': CKKSN,         # Polynomial modulus degree. For CKKS, n/2 values can be
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
import torchvision

class Net(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(Net, self).__init__()
        print("Number of classes : ", num_classes)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def get_vgg():
    temp = torchvision.models.vgg16()
    temp.load_state_dict(temp.state_dict())
    return temp


import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F



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

def get_parameters3(state_dict) -> List[np.ndarray]:
    #Renvoie une liste avec les poids puis les biais de chaque couche
    return [val.cpu().numpy() for _, val in state_dict.items()]

def get_EncryptedParameters2(net) -> List[np.ndarray]:
    #Renvoie une liste avec les poids puis les biais de chaque couche
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def encrypt(weights):

    

    i = 0
    #Fonction récursive pour encrypter chaque node ? 
    #---- Shape (1,3,3) --> 1 array 3*3
    #---- Shape (3,3) --> encrypter chaque array de la liste
    encryptedWeights = []
    for w in weights:
        if len(w.shape) > 2 :
            i+=1
            print(f"Récursivité {i}")
            encryptedWeights.append(encrypt(w))
        elif len(w.shape) == 1:
            encryptedWeights.append(HE.encrypt(w))
        else:

                #Une limitation de taille de CKKSN / 2 par array chiffré est imposée par le module d'encryption.
                #Afin de contourner ce modèle, on découpe l'array en sous array qu'on crypte par la suite, avant de les réunir à la fin
                
                #NB : Ne fonctionne pas car pas possible de refusionner les 2 arrays chiffrés comme des normaux
                #_n = lambda x: np.split(x, list(range(0,len(node),CKKSN//2))[1:])
                
                
                 encryptedWeights.append([HE.encrypt(node) for node in w])
        
    return encryptedWeights

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)



def decrypt(weight, shape):
    #Fonction récursive pour encrypter chaque node ? 
    #---- Shape (1,3,3) --> 1 array 3*3
    #---- Shape (3,3) --> encrypter chaque array de la liste
    encryptedWeights = []
    
    if not isinstance(weight, list):
        encryptedWeights.append(HE.decryptFrac(weight)[:shape[-1]])

    else :
        for w in weight :
            if isinstance(w, list):
                encryptedWeights.append(decrypt(w, shape))
                
            else:
                _r = lambda x: np.round(x, decimals=8)
                encryptedWeights.append(_r(HE.decryptFrac(w)[:shape[-1]]))
            
           
        
    return encryptedWeights

def aggregation(N,weights):

    """
    Applique la mise en commun des modèles de chaque client.

    Parameters:
    l (tuple) : tuple contenant les poids de la couche de chaque client.
    N (int) : Nombre de client participant au FL.

    Returns:
    list: moyenne des poids des clients


    """
    moy = []
    #Dans ce cas-ci : 7
    for l in weights :

        if not isinstance(l[0], list):
            #Poids à une seule dimension
            moy.append(sum(l))
    #tous les Models ont la même structure
        else :
            if isinstance(l[0][0], list):
                #Encore 2 dimensions de couches
                moy.append(aggregation(N, list(zip(*l))))

            else:
                #Une liste de poids cryptés
                moy.append(moy_agg(l,N))
    return moy

def moy_agg(l,N):
    """
    Fonction d'aggregation effectuant une moyenne simple des poids des clients.

    Parameters:
    l (tuple) : tuple contenant les poids de la couche de chaque client.
    N (int) : Nombre de client participant au FL.

    Returns:
    list: moyenne des poids des clients


    """
    return [sum(l[n][k] for n in range(N))/N for k in range(len(l[0]))]

def main():
    model = get_vgg()
    
    start_time = time.time()
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
    model = Net10().to(device)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)

        
        
        
        startEn = time.time()
        print("[STATUS] : Encryption des poids en cours...")
        param = get_parameters2(model)
        encryptedWeights = encrypt(param)
        print(f"[SUCCESS] : poids encryptés en {time.time() - startEn}")
        otherW = encryptedWeights.copy()
        weights = []
        weights.append(otherW)
        weights.append(encryptedWeights)
        weights = list(zip(*weights))
        N = 2 #[ModelW1, ModelW2,..., ModelWn] avec N = nClients

        startAg = time.time()
        print("[STATUS] : Agreggation des poids en cours...")
        moy = aggregation(N,weights)
        print(f"[SUCCESS] : aggregation en {time.time() - startAg}")

        startdec = time.time()
        print("[STATUS] : Decryption des poids en cours...")
        decrypted = []
        input_shapes = [p.shape for p in param]
        for weight, shape in zip(moy, input_shapes):
            decrypted.append(decrypt(weight,shape))
        
        newParam = []
        print(f"[SUCCESS] : decryption en {time.time() - startdec}")
        #4D : Conv
        #2D : Linear
        #1D : biais
        startcons = time.time()
        print("[STATUS] : Construction du nouveau modèle en cours...")
        for i in range(len(decrypted)) :
            
            if isinstance(decrypted[i], np.ndarray):
                #Reshape les biais 
                newParam.append(np.array(decrypted[i][:param[i].size]))
            else:
                #Reshape les poids
                newParam.append(np.array(decrypted[i], dtype="float32").reshape(param[i].shape))
        
        
        
        set_parameters(model, newParam)
        print(f"[SUCCESS] : nouveau modèle construit en {time.time() - startcons}")
        test(model, device, test_loader)


        print(f"Epoch {epoch} en {time.time() - start_time}")
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")





if __name__ == '__main__':
    main()

