from torch_geometric.data import Data, InMemoryDataset
import os
from torchmetrics.functional import accuracy
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import os
from torch_geometric.data import DataLoader
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from collections import deque
from torch_geometric.utils import k_hop_subgraph
import time

torch.cuda.empty_cache()
torch.backends.cuda.max_split_size_mb = 16
#print("PyTorch Lightning version:", pl.__version__)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class LitClassifier(pl.LightningModule):
    def __init__(
            self, lr: float = 1e-3, num_workers: int = 4, batch_size: int = 32, graph_dataset=None,
            use_gat: bool = True, use_transformer: bool = True, use_tabular: bool = True,
    ):
        # seed_set()
        # set_deterministic()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_step_outputs_y_pred_class = []
        self.training_losses = []
        super().__init__()
        self.lr = lr
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.graph_dataset = graph_dataset


        self.transformer_cls = TransformerEncoder(
            TransformerEncoderLayer(d_model=16, nhead=2),
            num_layers=1
        )
        self.output_layer = nn.Linear(16, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.cls_proj = nn.Linear(4, 16)

        self.use_gat = use_gat
        self.use_transformer = use_transformer
        self.use_tabular = use_tabular


        # GNN layers
        if self.use_gat:

            self.conv1 = GATConv(2, 16)
            self.conv2 = GATConv(16, 16)
            self.conv3 = GATConv(16, 16)
            self.conv4 = GATConv(16, 16)
            self.conv5 = GATConv(16, 16)

        if not self.use_gat:
            self.input_proj = nn.Linear(2, 16)

        if self.use_transformer:
            self.transformer_encoder = TransformerEncoder(
                TransformerEncoderLayer(d_model=16, nhead=2), num_layers=1
            )
            self.classifier = torch.nn.Linear(16, 2)
        else:
            self.classifier = torch.nn.Linear(16, 2)

        # Tabular feature layers
        if self.use_tabular:
            self.ln1 = nn.Linear(7, 1)
            self.ln2 = nn.Linear(35, 10)
            self.ln3 = nn.Linear(10, 1)
            self.ln4 = nn.Linear(4, 2)
        else:
            self.ln4 = nn.Linear(2, 2)  # Only GAT+Transformer features

        self.log_softmax = nn.LogSoftmax(dim=1)

    def bfs_nearest_shortest_path_node(self, edge_index, node_positions):
        # Convert edge_index to an adjacency list for easier BFS
        adj_list = {i: [] for i in range(node_positions.size(0))}
        for u, v in edge_index.T:
            adj_list[u.item()].append(v.item())

        # Update positions for nodes not on the shortest path
        for node in range(node_positions.size(0)):
            if node_positions[node] != -1:
                continue

            queue = deque([node])
            visited = set([node])

            while queue:
                current_node = queue.popleft()

                for neighbor in adj_list[current_node]:
                    # If the neighbor is on the shortest path, update position
                    if node_positions[neighbor] != -1:
                        node_positions[node] = node_positions[neighbor]
                        queue.clear()
                        break

                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        return node_positions

    def forward(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)

        # open
        node_positions = self.bfs_nearest_shortest_path_node(edge_index, x[:, 1].clone())
        x[:, 1] = node_positions

        # Identify shortest path nodes and get 5-hop neighbors
        shortest_path_mask = x[:, 1] != -1
        shortest_path_nodes = torch.where(shortest_path_mask)[0]

        # print(shortest_path_nodes, shortest_path_nodes.shape)
        subgraph_nodes, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
            shortest_path_nodes, num_hops=5, edge_index=edge_index, relabel_nodes=True
        )
        subgraph_x = x[subgraph_nodes]
        subgraph_edge_attr = edge_attr[edge_mask]
        if self.use_gat:
            x = F.relu(self.conv1(subgraph_x, subgraph_edge_index, subgraph_edge_attr))
            if self.training:
                x = F.dropout(x, p=0.5, training=True)
            x = F.relu(self.conv2(x, subgraph_edge_index, subgraph_edge_attr))
            x = F.relu(self.conv3(x, subgraph_edge_index, subgraph_edge_attr))
            x = F.relu(self.conv4(x, subgraph_edge_index, subgraph_edge_attr))
            x = F.relu(self.conv5(x, subgraph_edge_index, subgraph_edge_attr))
        else:
            # Simple mean-pooled features without GAT
            x = subgraph_x

        if not self.use_gat:
            x = self.input_proj(x)  # Project from 2 → 16 to match Transformer

        # Transformer encoder (if enabled)
        if self.use_transformer:
            sorted_indices = torch.argsort(shortest_path_nodes)
            x = self.transformer_encoder(x[sorted_indices].unsqueeze(1))

        # Global mean pooling
        x = global_mean_pool(x.squeeze(1), None)
        x = self.classifier(x)

        if self.use_tabular:
            tab1 = data.tabular_data1
            tab2 = data.tabular_data2
            if tab1.size(0) / 7 == self.batch_size:
            #without k t covernumber CEO SP
            # if tab1.size(0) / 8 == self.batch_size:
            # without KSP
            # if tab1.size(0) / 5 == self.batch_size:
                tab1 = tab1.view(self.batch_size, -1)
                tab2 = tab2.view(self.batch_size, -1)
            else:
                self.batch_size = int(tab1.size(0) / 7)
                tab1 = tab1.view(self.batch_size, -1)
                tab2 = tab2.view(self.batch_size, -1)


            tab1 = self.ln1(tab1)
            tab2 = self.ln2(tab2)
            tab2 = self.ln3(tab2)
            tab = torch.cat((tab1, tab2), dim=1)
            c = torch.cat((x, tab), dim=1)
        else:
            c = x


        # transformer model
        c = self.cls_proj(c)  # Ensure shape [B, 16](4,16)
        c = self.transformer_cls(c.unsqueeze(0)).squeeze(0)  # [1, B, 16] → [B, 16]
        c = self.output_layer(c)  # [B, 2]
        c = F.log_softmax(c, dim=1)

        return c
    def test_with_input(self, batch):
        print("In test_with_input_step")
        data = batch
        data = data.to(self.device)

        y_pred = self.forward(batch)
        _, y_pred_class = torch.max(y_pred, dim=1)

        y_pred_class = y_pred_class.cpu().detach().numpy()

        return ' '.join(map(str, y_pred_class))