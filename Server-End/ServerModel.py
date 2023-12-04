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

torch.cuda.empty_cache()
torch.backends.cuda.max_split_size_mb = 16
#print("PyTorch Lightning version:", pl.__version__)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class CombinedDataset(Dataset):
    def __init__(self, tabular_pickle_file, graph_data_root):
        self.tabular_dataset = NYTabularDataset(tabular_pickle_file)
        self.graph_dataset = NYGraphDataset(graph_data_root)

    def find_line_number(self, ID1, ID2):
        with open("NY_NodePair", "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                node_pair = line.strip().split()
                if int(ID1) == int(node_pair[0]) and int(ID2) == int(node_pair[1]):
                    return i
        return -1

    def __len__(self):
        return min(len(self.tabular_dataset), len(self.graph_dataset))

    def __getitem__(self, idx):
        tabular_data1, tabular_data2, y, ID1, ID2 = self.tabular_dataset[idx]
        line_number = self.find_line_number(ID1, ID2)

        if line_number != -1:
            graph_data = self.graph_dataset[line_number]
            x = graph_data.x
            edge_index = graph_data.edge_index
            edge_attr = graph_data.edge_attr
        else:
            x = None
            edge_index = None
            edge_attr = None

            # Construct the combined data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Store additional information in the data object
        data.tabular_data1 = tabular_data1
        data.tabular_data2 = tabular_data2
        data.y = y
        data.ID1 = ID1
        data.ID2 = ID2
        # data.oTab = oTab
        # print("ID1: ", ID1, " ID2: ", ID2, " ", " x: ", x.shape, " edge_index: ",edge_index.shape, " edge_attr: ", edge_attr.shape, " line_number: ", line_number, " y: ", y, " tabular_data: ", oTab)
        return data


save_path = "./NY_testResult"

class NYTabularDataset(Dataset):
    def __init__(self, pickle_file):
        self.pickle_file = pickle_file
        self.tabular = pd.read_table(pickle_file, sep=",")

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tabular = self.tabular.iloc[idx, 0:]
        xLevel = tabular["xLevel"]
        yLevel = tabular["yLevel"]
        ID1 = tabular["ID1"]
        ID2 = tabular["ID2"]
        t = tabular["t"]
        k = tabular["k"]
        y = tabular["result"]
        if y == "simple!!!":
            y = 0
        if y == "complex!!!":
            y = 1
        tabular1 = tabular[[
            "k", "t", "coverNumber", "CEO Number", "SP",
            "5000", "10000", "15000", "20000",
        ]]
        tabular2 = tabular[[
            "1_0.001", "1_0.002", "1_0.003", "1_0.004", "1_0.005",
            "1_0.006", "1_INF", "2_0.001", "2_0.002", "2_0.003",
            "2_0.004", "2_0.005", "2_0.006", "2_INF", "3_0.001",
            "3_0.002", "3_0.003", "3_0.004", "3_0.005", "3_0.006",
            "3_INF", "4_0.001", "4_0.002", "4_0.003", "4_0.004",
            "4_0.005", "4_0.006", "4_INF", "5_0.001", "5_0.002",
            "5_0.003", "5_0.004", "5_0.005", "5_0.006", "5_INF",
        ]]
        tabular1 = tabular1.tolist()
        tabular2 = tabular2.tolist()
        tabular1 = torch.FloatTensor(tabular1)
        tabular2 = torch.FloatTensor(tabular2)
        # oTab = torch.DoubleTensor(tabular)
        # tabular = torch.FloatTensor(tabular)

        return tabular1, tabular2, y, ID1, ID2

class NYGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(NYGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['graph_data.txt', 'edge_index.txt', 'edge_attribute.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass  # Download is not implemented in this example

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        edge_index_path = os.path.join(self.raw_dir, 'edge_index.txt')
        graph_path = os.path.join(self.raw_dir, 'graph_data.txt')
        edge_attributes_path = os.path.join(self.raw_dir, 'edge_attribute.txt')
        count = 0

        with open(graph_path, 'r') as f_graph, open(edge_index_path, 'r') as f_edge_index, open(edge_attributes_path, 'r') as f_edge_attr:
            for line_graph, line_edge_index, line_edge_attr in zip(f_graph, f_edge_index, f_edge_attr):
                #print(count)
                edge_index = self.load_edge_index(line_edge_index)
                graph_data = self.load_graph_data(line_graph)
                edge_attr = self.load_edge_attributes(line_edge_attr)

                data = Data(x=graph_data, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([1]))
                data_list.append(data)
                count += 1

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def load_graph_data(line):
        elements = line.strip().split(' ')
        elements = [list(map(int, el.strip('[]').split(','))) for el in elements if el != '']
        if '' in elements:
            print(f"Found an empty string in graph_data line: {line}")
        return torch.tensor(elements, dtype=torch.float)

    @staticmethod
    def load_edge_index(line):
        elements = line.strip().split(' ')
        elements = [list(map(int, el.strip('[]').split(','))) for el in elements if el != '']
        if '' in elements:
            print(f"Found an empty string in edge_index line: {line}")
        return torch.tensor(elements, dtype=torch.long).t().contiguous()

    @staticmethod
    def load_edge_attributes(line):
        elements = line.strip().split(' ')
        elements = [list(map(int, el.strip('[]').split(','))) for el in elements if el != '']
        if '' in elements:
            print(f"Found an empty string in edge_attributes line: {line}")
        return torch.tensor(elements, dtype=torch.float)

class LitClassifier(pl.LightningModule):
    def __init__(
        self, lr: float = 1e-3, num_workers: int = 4, batch_size: int = 1, graph_dataset=None
    ):
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_step_outputs_y_pred_class = []
        super().__init__()
        self.lr = lr
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.graph_dataset = graph_dataset

        # self.conv1 = GATConv(2, 16)
        # self.conv2 = GATConv(16, 32)

        self.conv1 = GATConv(2, 16)
        self.conv2 = GATConv(16, 16)
        self.conv3 = GATConv(16, 16)
        self.conv4 = GATConv(16, 16)
        self.conv5 = GATConv(16, 16)

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=16, nhead=2), num_layers=1
        )


        self.classifier = torch.nn.Linear(16, 2)
        self.ln1 = nn.Linear(9, 1)
        self.ln2 = nn.Linear(35, 10)
        self.ln3 = nn.Linear(10, 1)
        # cat together
        self.ln4 = nn.Linear(4, 2)
        # self.ln4 = nn.Linear(4, 1)
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
        #tab = data.tabular_data

        node_positions = self.bfs_nearest_shortest_path_node(edge_index, x[:, 1].clone())
        x[:, 1] = node_positions
        shortest_path_mask = x[:, 1] != -1
        shortest_path_nodes = torch.where(shortest_path_mask)[0]

        # print(shortest_path_nodes, shortest_path_nodes.shape)
        subgraph_nodes, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
            shortest_path_nodes, num_hops=5, edge_index=edge_index, relabel_nodes=True
        )
        subgraph_x = x[subgraph_nodes]
        subgraph_edge_attr = edge_attr[edge_mask]

        # Apply GNN layers on the subgraph
        x = F.relu(self.conv1(subgraph_x, subgraph_edge_index, subgraph_edge_attr))
        if self.training:
            x = F.dropout(x, p=0.5, training=True)
        x = F.relu(self.conv2(x, subgraph_edge_index, subgraph_edge_attr))
        x = F.relu(self.conv3(x, subgraph_edge_index, subgraph_edge_attr))
        x = F.relu(self.conv4(x, subgraph_edge_index, subgraph_edge_attr))
        x = F.relu(self.conv5(x, subgraph_edge_index, subgraph_edge_attr))

        # Sort nodes by their positions and apply transformer encoder
        sorted_indices = torch.argsort(shortest_path_nodes)
        x_transformed = self.transformer_encoder(x[sorted_indices].unsqueeze(1))
        x = global_mean_pool(x_transformed.squeeze(1), None)
        x = self.classifier(x)

        tab1 = data.tabular_data1
        tab2 = data.tabular_data2
        if tab1.size(0) / 9 == self.batch_size:
            tab1 = tab1.view(self.batch_size, -1)
            tab2 = tab2.view(self.batch_size, -1)
        else:
            self.batch_size = int(tab1.size(0) / 9)
            tab1 = tab1.view(self.batch_size, -1)
            tab2 = tab2.view(self.batch_size, -1)
        tab1 = self.ln1(tab1)
        tab2 = self.ln2(tab2)
        tab2 = self.ln3(tab2)

        tab = torch.cat((tab1, tab2), dim=1)
        c = torch.cat((x, tab), dim=1)
        c = self.ln4(c)
        c = F.log_softmax(c, dim=1)
        return c

    def training_step(self, batch, batch_idx):
        data = batch
        data = data.to(self.device)
        y = data.y.to(self.device)
        criterion = torch.nn.NLLLoss()
        y_pred = self.forward(batch)
        loss = criterion(y_pred, y)
        self.training_losses.append(loss)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def on_train_epoch_end(self):
        avg_training_loss = torch.stack(self.training_losses).mean()
        print(f"Epoch {self.current_epoch} - Avg training loss: {avg_training_loss.item()}")
        self.training_losses = []

    def validation_step(self, batch, batch_idx):
        print("In validation_step")
        data = batch
        data = data.to(self.device)
        y = data.y.to(self.device)
        y_pred = self.forward(batch)
        criterion = torch.nn.NLLLoss()
        #cross loss
        val_loss = criterion(y_pred, y)
        self.validation_step_outputs.append(val_loss)
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x for x in self.validation_step_outputs]).mean()
        self.log("val_loss", avg_loss)
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_with_input(self, batch):
        print("In test_with_input_step")
        data = batch
        data = data.to(self.device)

        y_pred = self.forward(batch)
        _, y_pred_class = torch.max(y_pred, dim=1)

        y_pred_class = y_pred_class.cpu().detach().numpy()

        return ' '.join(map(str, y_pred_class))


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

    def setup(self, stage):
        # Set seed again just before splitting to ensure consistency
        generator = torch.Generator().manual_seed(42)
        #tabular_dataset = CombinedDataset('./MAN_Result_changeKSP_fileNew_addKSPLost', '.')
        #tabular_dataset = CombinedDataset('./COL_Result_changeKSP_fileNew_addKSPLost', '.')
        tabular_dataset = CombinedDataset('./NY_Result_changeKSP_fileNew_addKSPLost', '.')
        half_size = 150000
        tabular_dataset, _ = random_split(tabular_dataset, [150000, len(tabular_dataset) - 150000])
        train_size = 120000
        val_size = 15000
        test_size = 15000
        self.train_set, self.val_set, self.test_set = random_split(tabular_dataset, (train_size, val_size, test_size),
                                                                   generator=generator)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr))

    def find_line_number(self, ID1, ID2):
        with open("NY_NodePair", "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                node_pair = line.strip().split()
                if int(ID1[0].item()) == int(node_pair[0]) and int(ID2[0].item()) == int(node_pair[1]):
                    return i
        return -1



if __name__ == "__main__":
    print("read graph data")