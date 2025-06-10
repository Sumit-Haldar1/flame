import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from flame.config import Config
from flame.mode.horizontal.syncfl.trainer import Trainer  
from torchvision import datasets, transforms
import argparse
import json
from flame.config import Config
logger = logging.getLogger(__name__)


class HorizontallySplitNet(nn.Module):
    def __init__(self, rank, world_size):
        super().__init__()
        assert world_size == 2
        self.rank = rank
        self.world_size = world_size

        self.conv1 = nn.Conv2d(1, 16, 3, 1)   # split of 32
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # split of 64
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608, 64)        # split of 128
        self.fc2 = nn.Linear(64, 10)          # shared or partially split

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
        return F.log_softmax(x, dim=1)


class HorizontalSplitTrainer(Trainer):
    def __init__(self, config: Config):
        self.config = config
        self.rank = self.config.hyperparameters.rank
        self.world_size = self.config.hyperparameters.world_size
       
        self.dataset_size = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = config.hyperparameters.epochs
        
        self.batch_size = self.config.hyperparameters.batch_size or 32
        self.model = HorizontallySplitNet(self.rank, self.world_size).to(self.device)
        #self.optimizer = optim.Adadelta(self.model.parameters())
        self.train_loader = None

        self.tmp_model = HorizontallySplitNet(self.rank, self.world_size).to(self.device)


        
    def initialize(self): pass

    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

        if self.rank == 0:
            indices = torch.arange(0, 2000)
        else:
            indices = torch.arange(2000, 4000)

        subset = data_utils.Subset(dataset, indices)
        self.train_loader = data_utils.DataLoader(subset, batch_size=self.batch_size, shuffle=True)

    def train(self, i = 0):

        self._update_model()
        
        self.optimizer = optim.Adadelta(self.model.parameters())
        for epoch in range(1, self.epochs + 1):
            #self._update_model()
            self._train_epoch(epoch)
            #self._update_weights()
        self.dataset_size = len(self.train_loader.dataset)

        self._update_weights()

        print("after training!!! conv1", self.weights['conv1.weight'][self.rank * 16])
        

    def _train_epoch(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                logger.info(f"epoch {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)}] loss: {loss.item():.6f}")
 
    def evaluate(self): pass

    def _update_model(self):
        self.model.load_state_dict(self._slice_weights(self.weights), strict=False)

    def _update_weights(self):
        # full_weights = {}
        # for name, param in self.model.state_dict().items():
        #     full_weights[name] = self._pad_tensor(name, param)
        # self.prev_weights = self.weights
        # self.weights = full_weights
        # print("lfro child")
        
        # print("conv1", self.weights['conv1.weight'][0])

        '''
        full_weights = {}
        for name, param in self.model.state_dict().items():
            full_weights[name] = self._pad_tensor(name, param)
        
      
        if not hasattr(self, "prev_weights") or self.prev_weights is None:
            print("*****************************")
            self.prev_weights = full_weights.copy()
        else:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            self.prev_weights = self.weights #self.weights

        self.weights = full_weights

        print("lfro child")
        print("conv1", self.weights['conv1.weight'][self.rank * 16])
        '''

        self.prev_weights = self._slice_weights(self.weights) 
        self.weights = self.model.state_dict()

        print(f"\n ======== self.weights: {self.weights}")

        print(f"\n +++++++++ self.prev_weights: {self.prev_weights}")

        self.tmp_model.load_state_dict(self.prev_weights, strict=False)
        full_prev_weights = {}
        for name, param in self.tmp_model.state_dict().items():
            full_prev_weights[name] = self._pad_tensor(name, param)

        self.prev_weights = full_prev_weights


        self.model.load_state_dict(self.weights, strict=False)
        full_weights = {}
        for name, param in self.model.state_dict().items():
            full_weights[name] = self._pad_tensor(name, param)

        self.weights = full_weights

        print(f"\n ======== self.weights: {self.weights}")
        print(f"\n +++++++++ self.prev_weights: {self.prev_weights}")

    def _pad_tensor(self, name, tensor):
        shape_map = {
            "conv1.weight": (32, 1, 3, 3),
            "conv1.bias": (32,),
            "conv2.weight": (64, 32, 3, 3),
            "conv2.bias": (64,),
            "fc1.weight": (128, 9216),
            "fc1.bias": (128,),
            "fc2.weight": (10, 128),
            "fc2.bias": (10,),
        }
        if name in shape_map:
            full_shape = shape_map[name]
            padded = torch.zeros(full_shape, device=tensor.device)

            if name == "conv1.weight":

                padded[self.rank * 16:(self.rank + 1) * 16] = tensor               
            elif name == "conv1.bias":
                padded[self.rank * 16:(self.rank + 1) * 16] = tensor
            elif name == "conv2.weight":
                padded[self.rank * 32:(self.rank + 1) * 32,
                       self.rank * 16:(self.rank + 1) * 16] = tensor
            elif name == "conv2.bias":
                padded[self.rank * 32:(self.rank + 1) * 32] = tensor
            elif name == "fc1.weight":
                padded[self.rank * 64:(self.rank + 1) * 64,
                       self.rank * 4608:(self.rank + 1) * 4608] = tensor
            elif name == "fc1.bias":
                padded[self.rank * 64:(self.rank + 1) * 64] = tensor
            elif name == "fc2.weight":
                padded[:, self.rank * 64:(self.rank + 1) * 64] = tensor
            elif name == "fc2.bias":
                padded = tensor

            
            return padded
           
        return tensor

    def _slice_weights(self, state_dict):
        sliced = {}
        for name, full_tensor in state_dict.items():
            if name == "conv1.weight":
                sliced[name] = full_tensor[self.rank * 16:(self.rank + 1) * 16]
            elif name == "conv1.bias":
                sliced[name] = full_tensor[self.rank * 16:(self.rank + 1) * 16]
            elif name == "conv2.weight":
                sliced[name] = full_tensor[self.rank * 32:(self.rank + 1) * 32,
                                           self.rank * 16:(self.rank + 1) * 16]
            elif name == "conv2.bias":
                sliced[name] = full_tensor[self.rank * 32:(self.rank + 1) * 32]
            elif name == "fc1.weight":
                sliced[name] = full_tensor[self.rank * 64:(self.rank + 1) * 64, :4608]
            elif name == "fc1.bias":
                sliced[name] = full_tensor[self.rank * 64:(self.rank + 1) * 64]
            elif name == "fc2.weight":
                sliced[name] = full_tensor[:, self.rank * 64:(self.rank + 1) * 64]
            elif name == "fc2.bias":
                sliced[name] = full_tensor
        return sliced


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.json")
    args = parser.parse_args()
    print("Parsed config file:", args.config)
    config = Config(args.config)
    t = HorizontalSplitTrainer(config)
    t.compose()
    t.run()


