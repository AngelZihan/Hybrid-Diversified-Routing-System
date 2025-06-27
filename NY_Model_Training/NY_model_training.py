import numpy as np
from numpy.random import shuffle
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
from torch.utils.data import Dataset, ConcatDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import glob
import random
from collections import deque
from torch_geometric.utils import k_hop_subgraph


torch.cuda.empty_cache()
torch.backends.cuda.max_split_size_mb = 16
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, specific_epochs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.specific_epochs = specific_epochs

    def _validate_condition_metric(self, logs):
        current_epoch = self.trainer.current_epoch
        if current_epoch in self.specific_epochs:
            return True
        return False


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

    def generate_additional_test_data(self, num_samples, k_value, t_value, SP):
        additional_test_data = []
        with open('additionalDataset.txt', 'w') as file:  # Open file for writing
            # Write the header line
            file.write("ID1,ID2,k,t,coverNumber,CEONumber,SP,5000,10000,15000,20000,1_0.001,1_0.002,1_0.003,1_0.004,1_0.005,1_0.006,1_INF,2_0.001,2_0.002,2_0.003,2_0.004,2_0.005,2_0.006,2_INF,3_0.001,3_0.002,3_0.003,3_0.004,3_0.005,3_0.006,3_INF,4_0.001,4_0.002,4_0.003,4_0.004,4_0.005,4_0.006,4_INF,5_0.001,5_0.002,5_0.003,5_0.004,5_0.005,5_0.006,5_INF")
            for _ in range(num_samples):
                random_index = np.random.randint(0, len(self))
                data_point = self[random_index]
                tabular_data1 = data_point.tabular_data1.clone()
                tabular_data2 = data_point.tabular_data2.clone()
                # Modify the values in tabular_data1
                tabular_data1[0] = k_value
                tabular_data1[1] = t_value
                tabular_data1[4] = SP

                # Update the data point
                new_data_point = Data(x=data_point.x, edge_index=data_point.edge_index, edge_attr=data_point.edge_attr)
                new_data_point.tabular_data1 = tabular_data1
                new_data_point.tabular_data2 = data_point.tabular_data2
                new_data_point.y = data_point.y
                new_data_point.ID1 = data_point.ID1
                new_data_point.ID2 = data_point.ID2

                additional_test_data.append(new_data_point)

                tabular_data1_list = tabular_data1.tolist()
                tabular_data2_list = tabular_data2.tolist()
                formatted_data1 = ','.join(map(str, tabular_data1_list[:9]))  # First 9 values from tabular_data1
                formatted_data2 = ','.join(map(str, tabular_data2_list))  # All values from tabular_data2
                data_line = f"{new_data_point.ID1},{new_data_point.ID2},{formatted_data1},{formatted_data2}\n"
                file.write(data_line)

        return additional_test_data

    def convert_to_dataset_format(self, item):
        fields = item.strip().split(',')
        # Convert fields to appropriate data types
        tabular_data1_fields = fields[2:4] + fields[6:13]  # Fields from positions 2 to 3 and 6 to 12
        tabular_data1 = torch.FloatTensor([float(x) for x in tabular_data1_fields])
        #tabular_data1 = torch.FloatTensor([float(x) for x in fields[2:11]])
        tabular_data2 = torch.FloatTensor([float(x) for x in fields[13:-1]])
        y = torch.tensor(int(fields[-1]))
        ID1 = torch.tensor(int(fields[0]))
        ID2 = torch.tensor(int(fields[1]))
        # Assuming you need to find the line number for graph data
        line_number = self.find_line_number(ID1, ID2)
        if line_number != -1:
            graph_data = self.graph_dataset[line_number]
            x = graph_data.x
            edge_index = graph_data.edge_index
            edge_attr = graph_data.edge_attr
        else:
            x = edge_index = edge_attr = None

        # Construct the combined data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.tabular_data1 = tabular_data1
        data.tabular_data2 = tabular_data2
        data.y = y
        data.ID1 = ID1  # Assign string directly
        data.ID2 = ID2  # Assign string directly
        #print(data)

        return data


    def __len__(self):
        #return len(self.tabular_dataset)
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
        #data.oTab = oTab
        #print("ID1: ", ID1, " ID2: ", ID2, " ", " x: ", x.shape, " edge_index: ",edge_index.shape, " edge_attr: ", edge_attr.shape, " line_number: ", line_number, " y: ", y, " tabular_data: ", oTab)
        return data



class NYTabularDataset(Dataset):
    def compute_statistics(self, t_values):
        for t_value in t_values:
            filtered_data = self.tabular[self.tabular['k'] == t_value]
            #print(filtered_data)
            total_count = len(filtered_data)
            if total_count == 0:
                print(f"No data for t = {t_value}")
                continue

            simple_count = len(filtered_data[filtered_data['result'] == 'simple!!!'])
            complex_count = len(filtered_data[filtered_data['result'] == 'complex!!!'])

            simple_percentage = (simple_count / total_count) * 100
            complex_percentage = (complex_count / total_count) * 100

            print(f"Statistics for t = {t_value}:")
            print(f"  Simple Percentage: {simple_percentage}%")
            print(f"  Complex Percentage: {complex_percentage}%")


    def normalize_data(self):
        # Normalize only numeric columns
        numeric_columns = self.tabular.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            min_val = self.tabular[column].min()
            max_val = self.tabular[column].max()
            self.tabular[column] = (self.tabular[column] - min_val) / (max_val - min_val)

    def __init__(self, pickle_file):
        self.pickle_file = pickle_file
        self.tabular = pd.read_table(pickle_file, sep=",")
        # self.tabular = pd.read_csv(pickle_file)
        # self.normalize_data()

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
                #self.apply_normalization(data_list)
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


    def apply_normalization(self, data_list):
        # Normalize features in each graph
        for data in data_list:
            for i in range(data.x.shape[1]):  # Assuming data.x is [num_nodes, num_features]
                max_val = torch.max(data.x[:, i])
                data.x[:, i] = data.x[:, i] / max_val


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




class LitClassifier(pl.LightningModule):
    def __init__(
        self, lr: float = 1e-3, num_workers: int = 4, batch_size: int = 32, graph_dataset=None,use_gat: bool = True, use_transformer: bool = True, use_tabular: bool = True,
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

        self.use_gat = use_gat
        self.use_transformer = use_transformer
        self.use_tabular = use_tabular

        # GNN layers
        if self.use_gat:
            # self.conv1 = GATConv(2, 64)
            # self.conv2 = GATConv(64, 128)
            # self.conv3 = GATConv(128, 256)

            self.conv1 = GATConv(2, 16)
            self.conv2 = GATConv(16, 16)
            self.conv3 = GATConv(16, 16)
            self.conv4 = GATConv(16, 16)
            self.conv5 = GATConv(16, 16)

        if not self.use_gat:
            self.input_proj = nn.Linear(2, 16)

        if self.use_transformer:
            # self.transformer_encoder = TransformerEncoder(
            #     TransformerEncoderLayer(d_model=256, nhead=2), num_layers=1
            # )
            # self.classifier = nn.Linear(256, 2)
            self.transformer_encoder = TransformerEncoder(
                TransformerEncoderLayer(d_model=16, nhead=2), num_layers=1
            )
            self.classifier = torch.nn.Linear(16, 2)
        else:
            # self.classifier = nn.Linear(256, 2)  # Still needs input shape matching GAT output
            self.classifier = torch.nn.Linear(16, 2)

        # Tabular feature layers
        if self.use_tabular:
            self.ln1 = nn.Linear(9, 1)
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

        #open

        node_positions = self.bfs_nearest_shortest_path_node(edge_index, x[:, 1].clone())
        x[:, 1] = node_positions

        #Identify shortest path nodes and get 5-hop neighbors
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
            x = self.transformer_encoder(x[sorted_indices].unsqueeze(1)).squeeze(1)

        # Global mean pooling
        subgraph_batch = torch.zeros(subgraph_nodes.size(0), dtype=torch.long, device=x.device)
        for idx, node in enumerate(subgraph_nodes):
            subgraph_batch[idx] = data.batch[node]
        x = global_mean_pool(x, subgraph_batch)
        x = self.classifier(x)

        if self.use_tabular:
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
            # tab = tab1
            c = torch.cat((x, tab), dim=1)
        else:
            c = x

        c = self.ln4(c)
        c = F.log_softmax(c, dim=1)
        # c = F.log_softmax(tab, dim=1)

        return c


    def training_step(self, batch, batch_idx):
        data = batch
        print(data)
        data = data.to(self.device)
        y = data.y.to(self.device)
        criterion = torch.nn.NLLLoss()
        y_pred = self.forward(batch)
        loss = criterion(y_pred, y)
        #print("y: ", y, " y_pred:", y_pred)
        #print("loss: ", loss)
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
        print(data)
        data = data.to(self.device)
        y = data.y.to(self.device)
        y_pred = self.forward(batch)
        criterion = torch.nn.NLLLoss()
        # cross loss
        val_loss = criterion(y_pred, y)
        self.validation_step_outputs.append(val_loss)
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x for x in self.validation_step_outputs]).mean()
        self.log("val_loss", avg_loss)
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        data = batch
        print(data)
        data = data.to(self.device)
        y = data.y.to(self.device)

        y_pred = self.forward(batch)
        criterion = torch.nn.NLLLoss()
        test_loss = criterion(y_pred, y)
        # print("test_loss:", test_loss)
        _, y_pred_class = torch.max(y_pred, dim=1)
        self.test_step_outputs.append(y)
        self.test_step_outputs_y_pred_class.append(y_pred_class)
        # print("y:", y)
        # print("y_pred_class:", y_pred_class)
        return {"test_loss": test_loss, "y_pred_class": y_pred_class, "y_true": y}



    def on_test_epoch_end(self):
        #print("test_step_outputs:",len(self.test_step_outputs))
        #print("test_step_outputs_y_pred_class:",len(self.test_step_outputs_y_pred_class))
        y_pred_classes = torch.cat([x for x in self.test_step_outputs_y_pred_class])
        y_true = torch.cat([m for m in self.test_step_outputs])

        y_pred_classes = y_pred_classes.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

        precision = precision_score(y_true, y_pred_classes, average='binary')
        recall = recall_score(y_true, y_pred_classes, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_classes).ravel()

        # Print precision, recall and confusion matrix
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        tensorboard_logs = {"precision": precision, "recall": recall}
        return {"precision": precision, "recall": recall, "log": tensorboard_logs}

    def train_dataloader(self):
        # sampler = None
        # if shuffle:
        #     sampler = torch.utils.data.RandomSampler(self.train_set, generator=torch.Generator().manual_seed(42))

        # return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, sampler=sampler, num_workers=0)

        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False,num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,num_workers=0)



    def setup(self, stage):
        # Set seed again just before splitting to ensure consistency
        generator = torch.Generator().manual_seed(42)
        tabular_dataset = CombinedDataset('NY_additionalDataset', '.')
        tabular_dataset, _ = random_split(tabular_dataset, [2600, len(tabular_dataset) - 2600])
        train_size = 2000
        val_size = 200
        test_size = 400
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

    generator = torch.Generator().manual_seed(SEED)
    print("read graph data")
    graph_dataset = NYGraphDataset('.')
    model = LitClassifier(graph_dataset=graph_dataset, batch_size=4)

    checkpoint_callback = CustomModelCheckpoint(
        specific_epochs=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21],  # 0-based indexing
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=30,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )


    trainer = pl.Trainer(accelerator="gpu", devices=[2], max_epochs=20,
                         default_root_dir="./NY_GNN/checkpoint",
                         callbacks=[checkpoint_callback])
    trainer.fit(model)

    # Determine the logger's version
    version = trainer.logger.version

    # After training, test model at specific checkpoints
    for epoch in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        checkpoint_path = f"./NY_GNN/checkpoint/lightning_logs/version_{version}/checkpoints/epoch={epoch - 1}-*.ckpt"
        matching_files = glob.glob(os.path.expanduser(checkpoint_path))
        if matching_files:
            checkpoint_to_load = matching_files[0]
            model = LitClassifier.load_from_checkpoint(checkpoint_to_load)
            print(f"Testing for epoch {epoch}")
            test_results = trainer.test(model)
        else:
            print(f"No checkpoint found for epoch {epoch}")

