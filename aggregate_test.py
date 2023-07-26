from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

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
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy


from client import decrypt, encrypt, get_parameters2, set_parameters
N = 10

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


def aggregate_fit(
    fit_res, 
) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

    # Convert results
    weights_results = [
        (parameters_to_ndarrays(res), np.random.randint(1000,1250))
        for res in fit_res
    ]
    parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))



    return parameters_aggregated



def main():
    model = Net()

    nEcrypt = 1 #Nombres de couches à encrypter
    fe = 0.2 #Fraction of the layer to encrypt

    startEn = time.time()

    print("[STATUS] : Encryption des poids en cours...")
    param = get_parameters2(model)
    input_shapes = [p.shape for p in param]


    encryptedWeights = encrypt(param)

    #print(param[1][0].nbytes, sys.getsizeof(HE.encrypt(param[1])))

    

    print(f"[SUCCESS] : poids encryptés en {time.time() - startEn}")

    otherW = encryptedWeights.copy()
    weights = []
    weights.append(otherW)
    weights.append(encryptedWeights)

    N = 2 #[ModelW1, ModelW2,..., ModelWn] avec N = nClients

    startAg = time.time()
    print("[STATUS] : Agreggation des poids en cours...")
    moy = aggregate_fit(ndarrays_to_parameters(weights))
    print(f"[SUCCESS] : aggregation en {time.time() - startAg}")

    startdec = time.time()
    print("[STATUS] : Decryption des poids en cours...")

    for i in range(0,len(param),2):
        if i > nEcrypt:
            param[i] = moy[i]
        else :
            foo = decrypt(param[i], input_shapes[i])
            #Ne décrypte que les couches chiffrées --> les nEncrypt entrées paires de param
            param[i] = np.array(foo, dtype="float32").reshape(input_shapes[i])
    


    print(f"[SUCCESS] : decryption en {time.time() - startdec}")
    #4D : Conv
    #2D : Linear
    #1D : biais
    startcons = time.time()


    print(set_parameters(model, param))
    print(f"[SUCCESS] : nouveau modèle construit en {time.time() - startcons}")

main()