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
from numcompress import compress, decompress
CKKSN = 2**13 #Réduction à 2**13 car problème de mémoire. Passage à 2**15 pour le support de davantage de modèle, mais hausse de la consommation en mémoire
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
HE.contextGen(**ckks_params)  # Generate context for bfv scheme
HE.keyGen()             # Key Generation: generates a pair of public/secret keys
HE.rotateKeyGen()
import torchvision


class Client():
    def __init__(self, model, ) -> None:
        self.model = model
        
        pass

    def encrypt(self):
        pass
    def getShapes(self, param):
        return [p.shape for p in param]
    def get_parameters(self) -> List[np.ndarray]:
        #Renvoie une liste avec les poids puis les biais de chaque couche
        return [val.cpu().numpy().astype('float32') for _, val in net.state_dict().items()]

def get_vgg():
    temp = torchvision.models.googlenet()
    temp.load_state_dict(temp.state_dict())
    return temp


def get_parameters2(net) -> List[np.ndarray]:
    #Renvoie une liste avec les poids puis les biais de chaque couche
    return [val.cpu().numpy().astype('float32') for _, val in net.state_dict().items()]



def get_parameters3(state_dict) -> List[np.ndarray]:
    #Renvoie une liste avec les poids puis les biais de chaque couche
    return [val.cpu().numpy() for _, val in state_dict.items()]
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


def encrypt(weights):

    i = 0
    #Fonction récursive pour encrypter chaque node ? 
    #---- Shape (1,3,3) --> 1 array 3*3
    #---- Shape (3,3) --> encrypter chaque array de la liste
    encryptedWeights = []
    if len(weights.shape) == 1:
        encryptedWeights.append(np.array(HE.encrypt(weights)))

    else:
        for w in weights:
            if len(w.shape) > 2 :
                i+=1
                print(f"Récursivité {i}")
                
                encryptedWeights.append(np.array(encrypt(w)))
            elif len(w.shape) == 1:

                encryptedWeights.append(np.array(HE.encrypt(w)))
            else:

                    #Une limitation de taille de CKKSN / 2 par array chiffré est imposée par le module d'encryption.
                    #Afin de contourner ce modèle, on découpe l'array en sous array qu'on crypte par la suite, avant de les réunir à la fin
                    
                    #NB : Ne fonctionne pas car pas possible de refusionner les 2 arrays chiffrés comme des normaux
                    #_n = lambda x: np.split(x, list(range(0,len(node),CKKSN//2))[1:])
                    
                    
                    encryptedWeights.append(np.array([HE.encrypt(node) for node in w]))

    return encryptedWeights

def decrypt(weight, shape):

    #Fonction récursive pour encrypter chaque node ? 
    #---- Shape (1,3,3) --> 1 array 3*3
    #---- Shape (3,3) --> encrypter chaque array de la liste
    decryptedWeights = []
    
    if not isinstance(weight, np.ndarray):
        
        decryptedWeights.append(HE.decryptFrac(weight)[:shape[-1]])

    else :
        if len(weight.shape) == 0:
            #Fix a bug when np.load() create à 0d-array.
            decryptedWeights.append(HE.decryptFrac(weight.item())[:shape[-1]])
        else: 
            for w in weight :
                
                if isinstance(weight, np.ndarray):
                    decryptedWeights.append(decrypt(w, shape))
                    
                else:
                    
                    _r = lambda x: np.round(x, decimals=8)
                    decryptedWeights.append(_r(HE.decryptFrac(w)[:shape[-1]]))

    
            
           
        
    return decryptedWeights

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

        if not isinstance(l[0], np.ndarray):
            #Poids à une seule dimension
            moy.append(sum(l)/2)
    #tous les Models ont la même structure
        else :
            try:
                if l[0].shape == 0:
                    moy.append(sum(weights)/N)
            except AttributeError:
                if isinstance(l[0], np.ndarray):
                    #Poids à une seule dimension
                    moy.append(sum(l)/N)
                if isinstance(l[0][0], np.ndarray):
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

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)



def main():
    model = Net()


    startEn = time.time()

    print("[STATUS] : Encryption des poids en cours...")
    param = get_parameters2(model)
    input_shapes = [p.shape for p in param]


    #print(param[1][0].nbytes, sys.getsizeof(HE.encrypt(param[1])))



    enc = [np.array(encrypt(p)) for p in param]

    print(f"[SUCCESS] : poids encryptés en {time.time() - startEn}")

    otherW = param.copy()
    weights = []
    weights.append(otherW)
    weights.append(param)
    weights = list(zip(*weights))

    N = 2 #[ModelW1, ModelW2,..., ModelWn] avec N = nClients

    startAg = time.time()
    print("[STATUS] : Agreggation des poids en cours...")
    moy = aggregation(N,enc)
    print(f"[SUCCESS] : aggregation en {time.time() - startAg}")

    startdec = time.time()
    print("[STATUS] : Decryption des poids en cours...")

    decrypted = []
    for weight, shape in zip(moy, input_shapes):
        decrypted.append(decrypt(weight,shape))
        
    


    print(f"[SUCCESS] : decryption en {time.time() - startdec}")
    #4D : Conv
    #2D : Linear
    #1D : biais
    startcons = time.time()


    print(set_parameters(model, param))
    print(f"[SUCCESS] : nouveau modèle construit en {time.time() - startcons}")

if __name__ == "__main__":
    main()