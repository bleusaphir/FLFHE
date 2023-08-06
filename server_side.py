import json
import time
import flwr as fl
import sys
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
from psutil._common import bytes2human
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
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import FitIns, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
import torchvision
from typing import List, Tuple, Dict, Optional, Callable
import flwr as fl
import torch.nn.functional
from flwr.common import Metrics
from flwr.common.logger import log
from collections import OrderedDict
print("flwr", fl.__version__)
print("numpy", np.__version__)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
from sklearn.metrics import classification_report
device = "gpu"  #@param ["cpu", "cuda", "mps","gpu"] {type:"string"}
N_CLIENTS = 3  #@param {type:"slider", min:3, max:10, step:1}
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
lr = 0.001  # 0.05
max_grad_norm = 1.2  # Tuning max_grad_norm is very important : Start with a low noise multiplier like 0.1, this should give comparable performance to a non-private model. Then do a grid search for the optimal max_grad_norm value. The grid can be in the range [0.1, 10].
epsilon = 50.0  # You can play around with the level of privacy, epsilon : Smaller epsilon means more privacy, more noise -- and hence lower accuracy. One useful technique is to pre-train a model on public (non-private) data, before completing the training on the private training data.
delta = 1e-5
DEVICE = torch.device(choice_device(device))  # Try "cuda" to train on GPU
print(f"Training on {DEVICE} using PyTorch {torch.__version__}")
CLASSES = classes_string(dataset)
print(CLASSES)
NUM_CLASSES = 10
from functools import reduce
import psutil
from flwr.server.strategy.fedavg import FedAvg

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""
def aggregate1(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    from functools import reduce
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    #print(f"[DEBUG] : num_examples[0] = {results[0][1]}, supposed '15000' ")
    #print(f"[DEBUG] : results[0] = {results[0][0]}, supposed ndarray of Pyfhel objets ")

    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weights = [
        w for w, _ in results
    ]

    weights_prime = aggregation(len(results), weights)
    # Compute average weights of each layery

    
    # weights_prime: NDArrays = [
    #     reduce(np.add, layer_update s) / len(results)
    #     for layer_updates in zip(*weighted_weights)
    # ]
    return weights_prime

def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:

    process = psutil.Process()
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])
    

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    

    
    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    DATA["RAM_agg"].append(bytes2human(process.memory_info().rss))
    return weights_prime

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

    if not isinstance(weights[0], list) and len(weights[0].shape) == 0:
         moy.append(sum([w.item() for w in weights])/N)


    elif isinstance(weights[0][0], np.ndarray):
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






class CustomStrategy(FedAvg):

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)
    
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        startAgg = time.time()
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        endAgg = time.time() - startAgg
        DATA['actual_round'].append(server_round)
        DATA['time_agg'].append(endAgg)
        
        print(f"Temps d'aggregation avec {N_CLIENTS} clients = {endAgg} secondes")

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated 
rounds = 3   #@param {type:"slider", min:1, max:50, step:1}    
frac_fit = 1  #@param {type:"slider", min:0.1, max:1.0, step:0.1}
frac_eval = 0.5  #@param {type:"slider", min:0.1, max:1.0, step:0.1}
min_fit_clients = N_CLIENTS  #@param {type:"slider", min:3, max:10, step:1}
min_eval_clients = N_CLIENTS  #@param {type:"slider", min:3, max:10, step:1}
min_avail_clients = N_CLIENTS  #@param {type:"slider", min:3, max:10, step:1}
DATA = {
    "actual_round" : [],
    "time_agg" : [],
    "RAM_agg" : []
}
Config = {
    "n_clients" : [N_CLIENTS],
    "rounds" : [rounds],
    'learning_rate' : [lr]
}
trainloaders, valloaders, testloader = data_setup.load_datasets(num_clients=N_CLIENTS, batch_size=batch_size,
                                                                resize=length, seed=seed,
                                                                num_workers=num_workers, splitter=split, data_path=data_path)
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


def get_on_fit_config_fn(epoch=2, lr=0.001, batch_size=32) -> Callable[[int], Dict[str, str]]:
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
            "local_epochs": 3 if server_round < 2 else epoch,
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
        set_parameters(self.net, parameters)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(self.net.parameters(), lr=lr)

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

        return get_parameters2(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)

        # Evaluate global model parameters on the local test data
        loss, accuracy, y_pred, y_true, y_proba = engine.test(self.net, self.valloader,
                                                              loss_fn=torch.nn.CrossEntropyLoss(), device=self.device)

        if self.save_results:
            os.makedirs(self.save_results, exist_ok=True)
            if self.matrix_path:
                save_matrix(y_true, y_pred, self.save_results + self.matrix_path, CLASSES)

            if self.roc_path:
                save_roc(y_true, y_proba, self.save_results + self.roc_path, NUM_CLASSES)

        # Return results, including the custom accuracy metric
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


# The client-side execution (training, evaluation) from the server-side




strategy = CustomStrategy(
    fraction_fit=frac_fit,  # Train on frac_fit % clients (each round)
    fraction_evaluate=frac_eval,  # Sample frac_eval % of available clients for evaluation
    min_fit_clients=min_fit_clients,  # Never sample less than 10 clients for training
    min_evaluate_clients=min_avail_clients if min_eval_clients else 5,  # Never sample less than 5 clients for evaluation
    min_available_clients=min_avail_clients,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters2(model_builder.Net(num_classes=NUM_CLASSES).to(DEVICE))),  # prevents Flower from asking one of the clients for the initial parameters
    evaluate_fn=None,  # Pass the evaluation function
    on_fit_config_fn=get_on_fit_config_fn(epoch=epochs, lr=lr, batch_size=batch_size),  # Pass the fit_config function
)

"""
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=number_clients,
    config=fl.server.ServerConfig(num_rounds=rounds),
    strategy=strategy
)
"""


# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = '172.16.2.1:'+str(5002) ,
        config=fl.server.ServerConfig(num_rounds=rounds),
        grpc_max_message_length = 1024*1024*1024, #1Gb
        strategy = strategy
)
print(DATA)
df = pd.DataFrame(DATA)
df_config = pd.DataFrame(Config)

# Étape 5 : Exportez le DataFrame vers un fichier Excel
nom_fichier_excel = f"results/resultats_entrainement_server_{N_CLIENTS}_clients.xlsx"
df.to_excel(nom_fichier_excel, index=False)

df_config.to_excel("results/config_server.xlsx", index=False)
