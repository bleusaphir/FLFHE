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
import pandas as pd
from going_modular.common import *
import psutil
from psutil._common import bytes2human
device = "gpu"
DEVICE = torch.device(choice_device(device))

from going_modular import model_builder, data_setup, engine
CKKSN = 2**15 #Réduction à 2**13 car problème de mémoire. Passage à 2**15 pour le support de davantage de modèle, mais hausse de la consommation en mémoire
import sys
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
HE.contextGen(**ckks_params)  # Generate context for bfv scheme            # Key Generation: generates a pair of public/secret keys
HE.rotateKeyGen()
HE.keyGen()
import torchvision

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



def get_vgg():
    temp = torchvision.models.vgg16()
    temp.load_state_dict(temp.state_dict())
    return temp

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

import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F
import psutil


def test(model, device, test_loader, save_results, matrix_path, roc_path):
    model.eval()

    test_loss = 0
    correct = 0
    loss, accuracy, y_pred, y_true, y_proba = engine.test(model, test_loader,
                                                              loss_fn=torch.nn.CrossEntropyLoss(), device=DEVICE)
    CLASSES = (
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    )
    if save_results:
            os.makedirs(save_results, exist_ok=True)
            if matrix_path:
                save_matrix(y_true, y_pred, save_results + matrix_path, CLASSES)

            if roc_path:
                save_roc(y_true, y_proba, save_results + roc_path, 10)
    

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



def main(N):
    model = get_vgg()
    
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
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
        n_couches = len(get_parameters2(model))//2
        start_time = time.time()
        train(args, model, device, train_loader, optimizer, epoch)

        
        
        
        startEn = time.time()
        print("[STATUS] : Encryption des poids en cours...")
        try :
            param = get_parameters2(model)

            encryptedWeights = encrypt(param)
            print(f"[SUCCESS] : poids encryptés en {time.time() - startEn}")
            ram_enc = bytes2human(psutil.virtual_memory().used)
            total_ram_enc = bytes2human(psutil.swap_memory().used + psutil.virtual_memory().used)
            encTime = time.time() - startEn
            weights = []
            for i in range(N):
                weights.append(encryptedWeights.copy())
            weights = list(zip(*weights))

            ram_serv = bytes2human(psutil.virtual_memory().used)
            total_ram_serv = bytes2human(psutil.swap_memory().used + psutil.virtual_memory().used)
            # Getting % usage of virtual_memory ( 3rd field)
            startAg = time.time()
            print("[STATUS] : Agreggation des poids en cours...")
            moy = aggregation(N,weights)
            ram_agg = bytes2human(psutil.virtual_memory().used)
            total_ram_agg = bytes2human(psutil.swap_memory().used + psutil.virtual_memory().used)
            print(f"[SUCCESS] : aggregation en {time.time() - startAg}")

            aggTime = time.time() - startAg

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
            decryptTime = time.time() - startdec
            startcons = time.time()
            print("[STATUS] : Construction du nouveau modèle en cours...")
            for i in range(len(decrypted)) :
                
                if isinstance(decrypted[i], np.ndarray):
                    #Reshape les biais 
                    newParam.append(np.array(decrypted[i][:param[i].size]))
                else:
                    #Reshape les poids
                    newParam.append(np.array(decrypted[i], dtype="float32").reshape(param[i].shape))
            
            ram_dec = bytes2human(psutil.virtual_memory().used)
            total_ram_dec = bytes2human(psutil.swap_memory().used + psutil.virtual_memory().used)
            set_parameters(model, newParam)
            print(f"[SUCCESS] : nouveau modèle construit en {time.time() - startcons}")
            test(model, device, test_loader, f"results{1}/", "matrix.png", "roc.png")
            success = "SUCCESS"
        except Exception as e:
            print(f"[FAILED] : {e}, Probably memory error.")
            success = "Failure"

        epochTime = time.time() - start_time
        print(f"Epoch {epoch} en {epochTime}")
        scheduler.step()

    total_time = time.time() - start_time
    data = {
        'n_clients' : [N],
        'n_couches' : [n_couches],
        'epochs' : [args.epochs],
        'learning_rate' : [args.lr],
        "Encryption_time" : [encTime],
        "Aggregation_time" : [aggTime],
        "Decryption_Time" : [decryptTime],
        "Epoch_time" : [epochTime],
        "total_time" : [total_time],
        "RAM_enc" : [ram_enc],
        "RAM_serv" : [ram_serv],
        "RAM_agg" : [ram_agg],
        "RAM_dec" : [ram_dec],
        "Total_Mem_enc" : [total_ram_enc],
        "Total_Mem_serv" : [total_ram_serv],
        "Total_Mem_agg" : [total_ram_agg],
        "Total_Mem_dec" : [total_ram_dec],
        "CKKSN" : [CKKSN],
        'Success' : [success]

    }
    
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


    df = pd.DataFrame(data)

    # Étape 5 : Exportez le DataFrame vers un fichier Excel
    nom_fichier_excel = f"resultats_entrainement{N}.xlsx"
    df.to_excel(nom_fichier_excel, index=False)


if __name__ == '__main__':
    n = 16 # Nombre clients

    main(n)

