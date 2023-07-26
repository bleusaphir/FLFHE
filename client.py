import sys
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
from typing import List, Tuple, Dict, Optional, Callable, cast
import flwr as fl
import torch.nn.functional
from flwr.common import Metrics
from flwr.common.typing import NDArray
from collections import OrderedDict
from flwr.common.logger import log
from logging import INFO, WARN
from Pyfhel import Pyfhel

from test import ndarrays_to_parameters, parameters_to_ndarrays
print("flwr", fl.__version__)
print("numpy", np.__version__)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
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




rounds =10  #@param {type:"slider", min:1, max:50, step:1}
frac_fit = 1  #@param {type:"slider", min:0.1, max:1.0, step:0.1}
frac_eval = 0.5  #@param {type:"slider", min:0.1, max:1.0, step:0.1}
min_fit_clients = 3  #@param {type:"slider", min:3, max:10, step:1}
min_eval_clients = 3  #@param {type:"slider", min:3, max:10, step:1}
min_avail_clients = 3  #@param {type:"slider", min:3, max:10, step:1}

trainloaders, valloaders, testloader = data_setup.load_datasets(num_clients=3, batch_size=batch_size,
                                                                resize=length, seed=seed,
                                                                num_workers=num_workers, splitter=split, data_path=data_path)


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
HE.load_public_key('public_key.key')
HE.load_secret_key('private_key.key')
HE.rotateKeyGen()

def get_parameters2(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_imbalance_weights(ds):
    labels = [x[1] for x in ds.train]
    _, label_counts = np.unique(labels, return_counts=True)
    weights = torch.DoubleTensor((1/label_counts)[labels])
    return weights


# weighted averaging function to aggregate the accuracy metric we return from evaluate
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Function will be by Flower called after every round : Centralized Evaluation (or server-side evaluation)
def evaluate2(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    #
    net = model_builder.Net(num_classes=NUM_CLASSES).to(DEVICE)
    net.to(DEVICE)


    valloader = valloaders[0]
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy, y_pred, y_true, y_proba = engine.test(net, testloader, loss_fn=torch.nn.CrossEntropyLoss(),
                                                          device=DEVICE)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


def get_on_fit_config_fn(epoch=2, lr=0.01, batch_size=32) -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(server_round: int) -> Dict[str, str]:
        """
        Return training configuration dict for each round with static batch size and (local) epochs.
        Perform two rounds of training with one local epoch, increase to two local epochs afterwards.
        """
        config = {
            "learning_rate": str(lr),
            "batch_size": str(batch_size),
            "server_round": server_round,  # The current round of federated learning
            "local_epochs": 1 if server_round < 2 else epoch,
        }
        print(server_round)
        return config

    return fit_config

class FlowerClient(fl.client.NumPyClient):
    """
    # The only real difference between Client and NumPyClient is
    that NumPyClient takes care of serialization and deserialization for you
    """

    def __init__(self, cid, net, trainloader, valloader, device, save_results, matrix_path, roc_path, privacy_engine):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.testloader=testloader
        self.device = device
        self.save_results = save_results
        self.matrix_path = matrix_path
        self.roc_path = roc_path
        self.privacy_engine = privacy_engine

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters2(self.net)

    def fit(self, parameters, config):
        # Read values from config
        server_round = config['server_round']
        local_epochs = config['local_epochs']
        lr = float(config["learning_rate"])

        # Use values provided by the config
        print(f'[Client {self.cid}, round {server_round}] fit, config: {config}')
        
        self.input_shapes = [p.shape for p in get_parameters2(self.net)]
        if server_round > 1 :
            #Passé le round 1, les poids sont chiffrés de manière permaente
            decrypted = []
            for weight, shape in zip(parameters, self.input_shapes):
                decrypted.append(np.array(decrypt(weight,shape)).squeeze())
            parameters = decrypted

        set_parameters(self.net, parameters)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01,
                                    momentum=0.9)
        # torch.optim.RMSprop(net.parameters(), lr=lr)

        # Start training
        results = engine.train(self.net, self.trainloader, self.valloader, optimizer=optimizer, loss_fn=criterion,
                               epochs=local_epochs, device=self.device, diff_privacy=False, delta=delta,
                               max_physical_batch_size=int(batch_size / 4), privacy_engine=self.privacy_engine)


        
        # Save results
        if self.save_results:
            os.makedirs(self.save_results, exist_ok=True)  # to create folders results
            # plot training curves (train and validation)
            plot_graph(
                [[*range(local_epochs)]] * 2,
                [results["train_acc"], results["val_acc"]],
                "Epochs", "Accuracy (%)",
                curve_labels=["Training accuracy", "Validation accuracy"],
                title="Accuracy curves",
                path=self.save_results + f"Accuracy_curves_Client {self.cid}")

            plot_graph(
                [[*range(local_epochs)]] * 2,
                [results["train_loss"], results["val_loss"]],
                "Epochs", "Loss",
                curve_labels=["Training loss", "Validation loss"], title="Loss curves",
                path=self.save_results + f"Loss_curves_Client {self.cid}")
        
        

        toEncrpyt = get_parameters2(self.net)
        #Encryption à l'envoi des résultats
        log(INFO, "Model encryption...")
        encryptedWeights = encrypt(toEncrpyt)
        log(INFO, "Model encryption ended.")


        #Pas de soucis au niveau de la décryption, ni de l'encryption.
        return encryptedWeights, len(self.trainloader), {}

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        print(f"[Client {self.cid}] evaluate, config: {config}")
        
        print("[STATUS] : Decryption des poids en cours...")
        decrypted = []
        params = get_parameters2(self.net)
        for weight, shape in zip(parameters, self.input_shapes):
            decrypted.append(np.array(decrypt(weight,shape)).squeeze())

        print(f"[SUCCESS] : decryption terminée. ")

        print("[STATUS] : Construction du nouveau modèle en cours...")

        
        set_parameters(self.net, decrypted)

        # Evaluate global model parameters on the local test data
        loss, accuracy, y_pred, y_true, y_proba = engine.test(self.net, self.testloader,
                                                              loss_fn=torch.nn.CrossEntropyLoss(), device=self.device)

        if self.save_results:
            os.makedirs(self.save_results, exist_ok=True)
            if self.matrix_path:
                save_matrix(y_true, y_pred, self.save_results + self.matrix_path, CLASSES)

            if self.roc_path:
                save_roc(y_true, y_proba, self.save_results + self.roc_path, NUM_CLASSES)

        # Return results, including the custom accuracy metric
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


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


def encrypt(weights):

    i = 0
    #Fonction récursive pour encrypter chaque node ? 
    #---- Shape (1,3,3) --> 1 array 3*3
    #---- Shape (3,3) --> encrypter chaque array de la liste
    encryptedWeights = []
    for w in weights:
        if len(w.shape) > 2 :

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
# The client-side execution (training, evaluation) from the server-side
def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load data
    """
    Note: each client gets a different trainloader/valloader, so each client will train and evaluate on their 
    own unique data
    """
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Load model
    net = model_builder.Net(num_classes=len(trainloader.dataset.dataset.dataset.classes)).to(DEVICE)
    net.to(DEVICE)

    
    # Envoyer le modèle sur GPU s'il est disponible

    # Create a  single Flower client representing a single organization
    return FlowerClient(cid, net, trainloader, valloader, device=DEVICE, matrix_path=matrix_path,
                        roc_path=roc_path, save_results=save_results, privacy_engine=None)

if __name__ == "__main__" :
    args = sys.argv[1:]
    id = args[0]
    fl.client.start_numpy_client(
        
            server_address='localhost'+ ':' + str(5002),
            client= client_fn(id),
    )


