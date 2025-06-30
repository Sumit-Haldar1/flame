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
        
        self.rank = rank
        self.world_size = world_size

        self.conv1 = nn.Conv2d(1, 32//world_size, 3, 1)   # split of 32
        self.conv2 = nn.Conv2d(32//world_size, 64//world_size, 3, 1)  # split of 64
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216//world_size, 128//world_size)        # split of 128
        self.fc2 = nn.Linear(128//world_size, 10)          # shared or partially split

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
        self.optimizer = optim.Adadelta(self.model.parameters())
        self.train_loader = None
        

        # self.tmp_model = HorizontallySplitNet(self.rank, self.world_size).to(self.device)


        

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = HorizontallySplitNet(self.rank, self.world_size).to(self.device)

    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)

        indices = torch.arange(self.rank*1000, (self.rank+1)*1000)

        

        subset = data_utils.Subset(dataset, indices)
        self.train_loader = data_utils.DataLoader(subset, batch_size=self.batch_size, shuffle=True)



    def train(self) -> None:
        """Train a model."""
        self.optimizer = optim.Adadelta(self.model.parameters(), lr= 0.01)

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        # save dataset size so that the info can be shared with aggregator
        self.dataset_size = len(self.train_loader.dataset)

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
                done = batch_idx * len(data)
                total = len(self.train_loader.dataset)
                percent = 100. * batch_idx / len(self.train_loader)
                logger.info(f"epoch: {epoch} [{done}/{total} ({percent:.0f}%)]"
                            f"\tloss: {loss.item():.6f}")

    def evaluate(self) -> None:
        """Evaluate a model."""
        # Implement this if testing is needed in trainer
        pass

   


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.json")
    args = parser.parse_args()
    config = Config(args.config)
    t = HorizontalSplitTrainer(config)
    t.compose()
    t.run()



