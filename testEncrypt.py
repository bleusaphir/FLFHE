from io import BytesIO
import itertools
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from glob import glob
from going_modular.common import *

from going_modular import model_builder, data_setup, engine
import torchvision
# Plot tool
import matplotlib.pyplot as plt
import torchvision

import numpy as np
from Pyfhel import Pyfhel

from sklearn.metrics import classification_report
device = "gpu"  #@param ["cpu", "cuda", "mps","gpu"] {type:"string"}
number_clients = 3  #@param {type:"slider", min:3, max:10, step:1}
num_workers = -1
epochs = 3  #@param {type:"slider", min:1, max:50, step:1}
batch_size = 8 #@param [1, 2, 4, 8, 16, 32, 64, 128, 256] {type:"raw"}
data_path = "./Clients/"
model_save = "model1.pt"
matrix_path = "confusion_matrix.png"  # None
roc_path = "roc.png"  # None
save_results = "results/classic/"  #@param ["", "results/FL/", "results/classic/"] {type:"string"}
dataset = "cifar"  #@param ["cifar", "animaux", "breast"] {type:"string"}
split = 10  #@param {type:"slider", min:5, max:100, step:5}
seed = 42
length = 32
lr = 0.01  # 0.05
max_grad_norm = 1.2  # Tuning max_grad_norm is very important : Start with a low noise multiplier like 0.1, this should give comparable performance to a non-private model. Then do a grid search for the optimal max_grad_norm value. The grid can be in the range [0.1, 10].
epsilon = 50.0  # You can play around with the level of privacy, epsilon : Smaller epsilon means more privacy, more noise -- and hence lower accuracy. One useful technique is to pre-train a model on public (non-private) data, before completing the training on the private training data.
delta = 1e-5
dataset_clients=["Client_1","Client_2","Client_3"]
DEVICE = torch.device(choice_device(device))  # Try "cuda" to train on GPU
print(f"Training on {DEVICE} using PyTorch {torch.__version__}")
CLASSES = classes_string(dataset)
print(CLASSES)
NUM_CLASSES = 10



class Client:
    id_iter = itertools.count()
    def __init__(self, weights) -> None:
        self.id = next(Client.id_iter)
        self.weights = weights
        self.encryptedWeights = None


    def encrypt(self):
        self.encryptedWeights = []
        for w in self.weights:
            encrW = [HE.encrypt(node) for node in w] 
            self.encryptedWeights.append(encrW)
        

def initialize_parameters(layer_dims, nSeed):
    """
    Cette fonction initialise les poids et les biais pour un réseau de neurones profonds.
    
    Arguments:
    layer_dims -- une liste qui contient les dimensions de chaque couche du réseau (input layer inclus)
    
    Returns:
    parameters -- un dictionnaire Python contenant les poids "W1", "b1", ..., "WL", "bL":
                    Wl -- matrice de poids de forme (layer_dims[l], layer_dims[l-1])
                    bl -- vecteur de biais de forme (layer_dims[l], 1)
    """
    
    np.random.seed(nSeed) # Pour conserver la même seed pour chaque initialisation
    
    weights = []
    bias = []
    L = len(layer_dims) # nombre de couches dans le réseau
    
    for l in range(1, L):
        weights.append(np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01)
        bias.append(np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01)
        
    return weights, bias

startTime = time.time()
weights1 = initialize_parameters([120,80,70, 60, 50, 40, 30, 20, 10], 25)[0]
weights2 = initialize_parameters([120,80,70, 60, 50, 40, 30, 20, 10], 1)[0]
weights3 = initialize_parameters([120,80,70, 60, 50, 40, 30, 20, 10], 5)[0]
weights4 = initialize_parameters([120,80,70, 60, 50, 40, 30, 20, 10], 63)[0]
weights5 = initialize_parameters([120,80,70, 60, 50, 40, 30, 20, 10], 77)[0]
clients = []

clients.append(Client(weights=weights1))
clients.append(Client(weights=weights2))
clients.append(Client(weights=weights3))
clients.append(Client(weights=weights4))
clients.append(Client(weights=weights5))
N = len(clients)

from going_modular import model_builder, data_setup, engine
trainloaders, valloaders, testloader = data_setup.load_datasets(num_clients=1, batch_size=batch_size,
                                                                resize=length, seed=seed,
                                                                num_workers=num_workers, splitter=split, data_path=data_path)


# Define the architecture
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




HE = Pyfhel()           # Creating empty Pyfhel object
ckks_params = {
    'scheme': 'CKKS',   # can also be 'ckks'
    'n': 2**14,         # Polynomial modulus degree. For CKKS, n/2 values can be
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

def aggregation(weights, nLayers=3):
    


            
    encryptedWeights = []
    for i in range(nLayers):
        w = [client.encryptedWeights[i] for client in clients] #Récupère les poids respectifs de chaque client pour chaque couche
        nNodes = len(w[0]) #Récupère le nombre de noeuds de la couche
        nodes = []
        for k in range(nNodes): 
            nodes.append(sum([w[n][k] for n in range(N)]) / 2) #Effectue la moyenne des poids de chaque client

        encryptedWeights.append(nodes)
        #mean = sum([client.encryptedWeights[i] for client in clients])

    return encryptedWeights


def decrypt(encryptedWeights):
    input_shapes = [120,80,70, 60, 50, 40, 30, 20, 10] #Trouver comment automatiser ça
    n = 0
    for layer in encryptedWeights:
        for i in range(len(layer)):
            _r = lambda x: np.round(x, decimals=5)
            layer[i] = _r(HE.decryptFrac(layer[i]))[:input_shapes[n]]
        n = n +1
    return encryptedWeights


from flwr.common.typing import NDArray, NDArrays, Parameters


#Génerer des poids aléatoires pour n clients


for client in clients:
    #On encrypte tous les poids de chaque client
    client.encrypt()

def ndarray_to_sparse_bytes(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()

    if len(ndarray.shape) > 1:
        # We convert our ndarray into a sparse matrix
        ndarray = torch.tensor(ndarray).to_sparse_csr()

        # And send it by utilizng the sparse matrix attributes
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
        np.savez(
            bytes_io,  # type: ignore
            crow_indices=ndarray.crow_indices(),
            col_indices=ndarray.col_indices(),
            values=ndarray.values(),
            allow_pickle=False,
        )
    else:
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
        np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()

newEncryptedWeights = aggregation(clients) #Moyenne des poids

import pickle

input_shapes = [120,80,70, 60, 50, 40, 30, 20, 10]
cipher = HE.encrypt(input_shapes)
dump = pickle.dumps(cipher)
newWeights = decrypt(newEncryptedWeights) #Decryption des nouveaux poids
print("Duration time : ", time.time() - startTime)
print('Yep')
