from __future__ import print_function
import argparse
from collections import OrderedDict
from io import BytesIO
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
import psutil
 

import time
from flwr.common import serde
from flwr.common.typing import NDArray, NDArrays, Parameters
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
HE.contextGen(**ckks_params)  # Generate context for bfv scheme            # Key Generation: generates a pair of public/secret keys
HE.rotateKeyGen()
HE.load_public_key('public_key.key')
HE.load_secret_key('private_key.key')

HE1 = Pyfhel()           # Creating empty Pyfhel object
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
HE1.contextGen(**ckks_params)  # Generate context for bfv scheme
HE1.rotateKeyGen()
HE1.load_public_key('public_key.key')
HE1.load_secret_key('private_key.key')
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


def get_parameters2(net) -> List[np.ndarray]:
    #Renvoie une liste avec les poids puis les biais de chaque couche
    return [val.cpu().numpy().astype('float32') for _, val in net.state_dict().items()]



def get_parameters3(state_dict) -> List[np.ndarray]:
    #Renvoie une liste avec les poids puis les biais de chaque couche
    return [val.cpu().numpy() for _, val in state_dict.items()]


def decrypt(weight, shape, HE):

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
                    decryptedWeights.append(decrypt(w, shape, HE))
                    
                else:
                    
                    _r = lambda x: np.round(x, decimals=8)
                    decryptedWeights.append(_r(HE.decryptFrac(w)[:shape[-1]]))

    
            
           
        
    return decryptedWeights


def encrypt(weights, HE):

    i = 0
    #Fonction récursive pour encrypter chaque node ? 
    #---- Shape (1,3,3) --> 1 array 3*3
    #---- Shape (3,3) --> encrypter chaque array de la liste
    encryptedWeights = []
    if len(weights.shape) == 1:
        encryptedWeights.append(HE.encrypt(weights))
    else :
        for w in weights:
            if len(w.shape) > 2 :

                encryptedWeights.append(encrypt(w, HE))
            elif len(w.shape) == 1:

                encryptedWeights.append(HE.encrypt(w))
            else:

                    #Une limitation de taille de CKKSN / 2 par array chiffré est imposée par le module d'encryption.
                    #Afin de contourner ce modèle, on découpe l'array en sous array qu'on crypte par la suite, avant de les réunir à la fin
                    
                    #NB : Ne fonctionne pas car pas possible de refusionner les 2 arrays chiffrés comme des normaux
                    #_n = lambda x: np.split(x, list(range(0,len(node),CKKSN//2))[1:])
                    
                    
                    encryptedWeights.append([HE.encrypt(node) for node in w])

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

    if isinstance(weights[0][0], np.ndarray):
        for i in range(len(weights[0])):
            moy.append(np.array(aggregation(N, [w[i] for w in weights])))
    else:
        return moy_agg([w for w in weights], N)
    return moy

def moy_agg(l,N):
    """
    Fonction d'aggregation effectuant une moyenne simple des poids des clients.

    Parameters:
    l (tuple) : tuple contenant les poids de la couche de chaque client.
    N (int) : Nombre de client participant au FL.

    Returns:
    list: moyenne des poids des clients

    List[num_clients][taille_couche]
    """
    return [sum(l[n][k] for n in range(N))/N for k in range(len(l[0]))]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    print("[INFO] Serialization...")
    tensors = [ndarray_to_bytes(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    print("[INFO] Deserialization...")
    """Convert parameters object to NumPy ndarrays."""
    return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]


def ndarray_to_bytes(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(bytes_io, ndarray, allow_pickle=True)  # type: ignore
    return bytes_io.getvalue()


def bytes_to_ndarray(tensor: bytes):
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    ndarray_deserialized = np.load(bytes_io, allow_pickle=True)  # type: ignore
    return ndarray_deserialized


def main():
    model = Net()
    model1 = Net()

    startEn = time.time()

    print("[STATUS] : Encryption des poids en cours...")
    param = get_parameters2(model)
    param1 = get_parameters2(model1)
    input_shapes = [p.shape for p in param]


    encryptedWeights = []
    encryptedWeights1 = []
    #print(param[1][0].nbytes, sys.getsizeof(HE.encrypt(param[1])))

    for layer in param:
        encryptedWeights.append(encrypt(layer, HE))
    
    for layer in param1:
        encryptedWeights1.append(encrypt(layer, HE1))

    

    print(f"[SUCCESS] : poids encryptés en {time.time() - startEn}")


    encryptedWeights = ndarrays_to_parameters(encryptedWeights)
    encryptedWeights1 = ndarrays_to_parameters(encryptedWeights1)

    encryptedWeights = parameters_to_ndarrays(encryptedWeights)
    encryptedWeights1 = parameters_to_ndarrays(encryptedWeights1)


    weights = []
    weights.append(encryptedWeights)
    weights.append(encryptedWeights1)
    

    N = 2 #[ModelW1, ModelW2,..., ModelWn] avec N = nClients

    startAg = time.time()
    print("[STATUS] : Agreggation des poids en cours...")
    moy = ndarrays_to_parameters(aggregation(N,weights))
    print(f"[SUCCESS] : aggregation en {time.time() - startAg}")

    

    startdec = time.time()
    print("[STATUS] : Decryption des poids en cours...")
    newParam = []
    moy = parameters_to_ndarrays(moy)
    for i in range(0,len(moy)):
            foo = decrypt(moy[i], input_shapes[i], HE1)
            newParam.append(np.array(foo, dtype="float32").reshape(input_shapes[i]))
    


    print(f"[SUCCESS] : decryption en {time.time() - startdec}")
    #4D : Conv
    #2D : Linear
    #1D : biais
    startcons = time.time()


    print(set_parameters(model, param))
    print(f"[SUCCESS] : nouveau modèle construit en {time.time() - startcons}")
