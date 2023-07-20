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
import matplotlib.pyplot as plt
import torchvision
from typing import List, Tuple, Dict, Optional, Callable
import flwr as fl
import torch.nn.functional
from flwr.common import Metrics
from flwr.common import Metrics
from collections import OrderedDict
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
lr = 0.00-1  # 0.05
max_grad_norm = 1.2  # Tuning max_grad_norm is very important : Start with a low noise multiplier like 0.1, this should give comparable performance to a non-private model. Then do a grid search for the optimal max_grad_norm value. The grid can be in the range [0.1, 10].
epsilon = 50.0  # You can play around with the level of privacy, epsilon : Smaller epsilon means more privacy, more noise -- and hence lower accuracy. One useful technique is to pre-train a model on public (non-private) data, before completing the training on the private training data.
delta = 1e-5
dataset_clients=["Client_1","Client_2","Client_3"]
DEVICE = torch.device(choice_device(device))  # Try "cuda" to train on GPU
print(f"Training on {DEVICE} using PyTorch {torch.__version__}")
CLASSES = classes_string(dataset)
print(CLASSES)
NUM_CLASSES = 10




rounds = 25   #@param {type:"slider", min:1, max:50, step:1}
frac_fit = 1  #@param {type:"slider", min:0.1, max:1.0, step:0.1}
frac_eval = 0.5  #@param {type:"slider", min:0.1, max:1.0, step:0.1}
min_fit_clients = 3  #@param {type:"slider", min:3, max:10, step:1}
min_eval_clients = 3  #@param {type:"slider", min:3, max:10, step:1}
min_avail_clients = 3  #@param {type:"slider", min:3, max:10, step:1}




trainloaders, valloaders, testloader = data_setup.load_datasets(num_clients=number_clients, batch_size=batch_size,
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




strategy = fl.server.strategy.FedProx(
    proximal_mu=0.2,
    fraction_fit=frac_fit,  # Train on frac_fit % clients (each round)
    fraction_evaluate=frac_eval,  # Sample frac_eval % of available clients for evaluation
    min_fit_clients=min_fit_clients,  # Never sample less than 10 clients for training
    min_evaluate_clients=number_clients//2 if min_eval_clients else min_eval_clients,  # Never sample less than 5 clients for evaluation
    min_available_clients=min_avail_clients,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters2(model_builder.Net(num_classes=NUM_CLASSES).to(DEVICE))),  # prevents Flower from asking one of the clients for the initial parameters
    evaluate_fn=evaluate2,  # Pass the evaluation function
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
        #grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
)

