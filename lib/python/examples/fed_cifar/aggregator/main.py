import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.top_aggregator import TopAggregator
from torchvision import datasets, transforms
import torch.utils.data as data_utils

logger = logging.getLogger(__name__)


class Net(nn.Module):
    """Net class."""

    def __init__(self):
        """Initialize."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Forward."""
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


class PyTorchMnistAggregator(TopAggregator):
    """PyTorch Mnist Aggregator."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.model = None
        self.dataset: Dataset = None

        self.device = None
        self.test_loader = None

    def initialize(self):
        """Initialize role."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = Net().to(self.device)
        self.model.load_state_dict(torch.load('/home/cc/flame/lib/python/examples/mnist/aggregator/pretrained_weights.pth'))



    def load_data(self):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        try:
            dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            indices = torch.arange(0, 5000)
            subset = data_utils.Subset(dataset, indices)
            self.test_loader = torch.utils.data.DataLoader(subset)
            self.dataset = Dataset(dataloader=self.test_loader)
            
        except Exception as e:
            logger.error(f"Failed to load CIFAR-10 dataset: {e}")
            raise

    def train(self) -> None:
        """Train a model."""
        # Implement this if testing is needed in aggregator
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(
                    output, target,
                    reduction='sum').item()  # sum up batch loss
                pred = output.argmax(
                    dim=1,
                    keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        total = len(self.test_loader.dataset)
        test_loss /= total
        test_accuray = correct / total


    

      

        # Write results to a file instead of logging to terminal
        with open('evaluation_results.txt', 'a') as f:
            f.write(f"Test round: {self._round-1}\n")
            f.write(f"Test loss: {test_loss}\n")
            f.write(f"Test accuracy: {correct}/{total} ({test_accuray})\n")
            f.write("\n")  # Add a blank line for readability

        # update metrics after each evaluation so that the metrics can be
        # logged in a model registry.
        self.update_metrics({
            'test-loss': test_loss,
            'test-accuracy': test_accuray
        })
        
        # if self._round == self.config.hyperparameters.rounds:
        #     torch.save(self.model.state_dict(), '/home/cc/flame/lib/python/examples/mnist/aggregator/pretrained_weights.pth')
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PyTorchMnistAggregator(config)
    a.compose()
    a.run()
