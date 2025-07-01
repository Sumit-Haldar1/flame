# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""MNIST horizontal FL aggregator for PyTorch.

The example below is implemented based on the following example from pytorch:
https://github.com/pytorch/examples/blob/master/mnist/main.py.
"""

import logging
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.top_aggregator import TopAggregator
from torchvision import datasets, transforms
from flame.mode.message import MessageType
from flame.common.util import weights_to_model_device
from flame.optimizer.train_result import TrainResult
from flame.common.util import (MLFramework, get_ml_framework_in_use,
                               valid_frameworks, weights_to_device,
                               weights_to_model_device)

from flame.common.constants import DeviceType
from datetime import datetime
import torch.utils.data as data_utils

import os
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)

PROP_ROUND_START_TIME = "round_start_time"


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
        self.trainer_rank = {}
        self.trainer_count = 0
        self.world_size = self.config.hyperparameters.world_size
        self.all_available = False
        self.con_time = 0
                            

    def initialize(self):
        """Initialize role."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = Net().to(self.device)
        self.model.load_state_dict(torch.load('/home/cc/flame/lib/python/examples/mnist/aggregator/pretrained_weights.pth'))   


    def load_data(self) -> None:
        """Load a test dataset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])

        dataset = datasets.FashionMNIST('./data',
                                 train=False,
                                 download=True,
                                 transform=transform)

        self.test_loader = torch.utils.data.DataLoader(dataset)

        # store data into dataset for analysis (e.g., bias)
        self.dataset = Dataset(dataloader=self.test_loader)


    

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
            f.write(f"Concatenation Time: {self.con_time}\n")
            f.write(f"Test loss: {test_loss}\n")
            f.write(f"Test accuracy: {correct}/{total} ({test_accuray})\n")
            f.write("\n")  # Add a blank line for readability

        # Update metrics for model registry
        self.update_metrics({
            'test-loss': test_loss,
            'test-accuracy': test_accuray
        })


    def _aggregate_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return
        
      

        appended_weights = [0] * self.world_size
        count_total = 0
        individual_count = 0 

        w_received = 0

        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, timestamp = metadata
            
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            logger.info(f"Received data from {end}")
     

            weights = None
            count = 0


            if MessageType.WEIGHTS in msg:
                weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)
                

            if MessageType.DATASET_SIZE in msg:
                count = msg[MessageType.DATASET_SIZE]

            if MessageType.DATASAMPLER_METADATA in msg:
                self.datasampler.handle_metadata_from_trainer(
                    msg[MessageType.DATASAMPLER_METADATA], end, channel
                )

            logger.debug(f"{end}'s parameters trained with {count} samples")

            if weights is not None and count > 0:
                count_total += count
                w_received += 1
                appended_weights[self.trainer_rank[end]] = weights

            individual_count = count
        concated = {}      

        
        if w_received == self.world_size:
            start_time = time.time()
            weights = []
            for w in appended_weights:
                weights.append(w['conv1.weight'])
            concated['conv1.weight'] = torch.cat(weights, 0)
            weights = []
            for w in appended_weights:
                weights.append(w['conv1.bias'])
            concated['conv1.bias'] = torch.cat(weights, 0)

            # combined = torch.cat((appended_weights[alt1]['conv2.weight'], appended_weights[alt2]['conv2.weight']), dim=1)
            # concated['conv2.weight'] = torch.cat((combined, combined), dim=0)

           
            weights_list = [w['conv2.weight'] for w in appended_weights] 
            concated['conv2.weight'] = torch.cat(weights_list, dim=0)  
            if concated['conv2.weight'].shape[1] != 32:
                concated['conv2.weight'] = torch.cat([concated['conv2.weight']] * (32 // concated['conv2.weight'].shape[1]), dim=1)  

            weights = []
            for w in appended_weights:
                weights.append(w['conv2.bias'])
            concated['conv2.bias'] = torch.cat(weights, 0)
   
            weights_list = [w['fc1.weight'] for w in appended_weights]  
            concated['fc1.weight'] = torch.cat(weights_list, dim=0)  
            if concated['fc1.weight'].shape[1] != 9216:
                concated['fc1.weight'] = torch.cat([concated['fc1.weight']] * (9216 // concated['fc1.weight'].shape[1]), dim=1)

            weights = []
            for w in appended_weights:
                weights.append(w['fc1.bias'])
            concated['fc1.bias'] = torch.cat(weights, 0)


            weights = []
            for w in appended_weights:
                weights.append(w['fc2.weight'])
            concated['fc2.weight'] = torch.cat(weights, 1)

           
            weights_list = [w['fc2.bias'] for w in appended_weights] 
            concated['fc2.bias'] = torch.mean(torch.stack(weights_list), dim=0) 

            tres = TrainResult(concated, count_total)

            self.cache["concat"] = tres
            end_time = time.time()

            self.con_time = end_time - start_time

                     

        logger.debug(f"Received and collected weights from {len(channel.ends())} trainers")
        
        if count_total == individual_count*self.world_size:
            print('yes i am working')
            global_weights = self.optimizer.do(
                deepcopy(self.weights),
                self.cache,
                total=count_total,
                num_trainers=len(channel.ends()),
            )
            if global_weights is None:
                logger.info("Failed model aggregation")
                time.sleep(1)
                return
            self.weights = global_weights
            self._update_model()
    
            

    def _distribute_weights(self, tag: str) -> None:
        
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        while 1:
            list_ends = channel.all_ends()
            n_ends = len(list_ends)

            # print(f"_distribute_weights Joined ends: {n_ends}")
            time.sleep(0.5)

            if n_ends == self.world_size:
                print(f"_distribute_weights with {n_ends} ends")
                break

        # before distributing weights, update it from global model
        self._update_weights()

        selected_ends = channel.ends()

        datasampler_metadata = self.datasampler.get_metadata(self._round, selected_ends)

        print("end num: ", len(selected_ends))

        #Swapping logic
        if (self._round-1) % 2 == 0 and (self._round-1) != 0 and self.trainer_count == self.world_size:
                for key in self.trainer_rank:
                    self.trainer_rank[key] = (self.trainer_rank[key] + 1) % self.world_size


        # send out global model parameters to trainers
        for end in selected_ends:
            if end not in self.trainer_rank.keys():
                self.trainer_rank[end] = self.trainer_count
                self.trainer_count += 1

            logger.debug(f"sending weights to {end}")
            
            
            temp = self._slice_weights(self.weights, self.trainer_rank[end], self.world_size)
            channel.send(
                end,
                {
                    MessageType.WEIGHTS: weights_to_device(
                        temp, DeviceType.CPU
                    ),
                    MessageType.ROUND: self._round,
                    MessageType.DATASAMPLER_METADATA: datasampler_metadata,
                },
            )
            # register round start time on each end for round duration measurement.
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (round, datetime.now())
            )



    def _slice_weights(self, state_dict, rank, world_size):
        sliced = {}
        for name, full_tensor in state_dict.items():
            if name == "conv1.weight":
                slice_size = 32 // world_size
                sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif name == "conv1.bias":
                slice_size = 32 // world_size
                sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif name == "conv2.weight":
                out_slice_size = 64 // world_size
                in_slice_size = 32 // world_size
                sliced[name] = full_tensor[rank * out_slice_size:(rank + 1) * out_slice_size,
                                        rank * in_slice_size:(rank + 1) * in_slice_size]
            elif name == "conv2.bias":
                slice_size = 64 // world_size
                sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif name == "fc1.weight":
                out_slice_size = 128 // world_size
                in_slice_size = (64 // world_size) * 12 * 12
                sliced[name] = full_tensor[rank * out_slice_size:(rank + 1) * out_slice_size,
                                        :in_slice_size]
            elif name == "fc1.bias":
                slice_size = 128 // world_size
                sliced[name] = full_tensor[rank * slice_size:(rank + 1) * slice_size]
            elif name == "fc2.weight":
                slice_size = 128 // world_size
                sliced[name] = full_tensor[:, rank * slice_size:(rank + 1) * slice_size]
            elif name == "fc2.bias":
                sliced[name] = full_tensor  # Not split, same for all trainers
        return sliced


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PyTorchMnistAggregator(config)
    a.compose() 
    a.run()
