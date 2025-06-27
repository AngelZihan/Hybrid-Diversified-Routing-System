import socket
import torch
from torch_geometric.data import Data
from NYServerModel_MLP import LitClassifier
import pytorch_lightning as pl

torch.cuda.set_device('cuda:7')

def process_data(edge_attribute, edge_index, graph_data, tabular_data1, tabular_data2):
    #print("tabular_data1", "tabular_data2")
    edge_index = load_edge_index(edge_index)
    graph_data = load_graph_data(graph_data)
    edge_attr = load_edge_attributes(edge_attribute)

    data = Data(x=graph_data, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([1]))

    data.tabular_data1 = tabular_data1
    data.tabular_data2 = tabular_data2
    #print(tabular_data)
    #data.ID1 = ID1
    #data.ID2 = ID2
    return data


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


def receive_all(conn, buffer_size):
    chunks = []
    terminator = b'END_OF_MESSAGE'  # define a unique terminator sequence
    while True:
        chunk = conn.recv(buffer_size)
        if not chunk:
            print("Connection closed by the client!")
            break
        chunks.append(chunk)
        # check if the last few received bytes match the terminator sequence
        if b''.join(chunks)[-len(terminator):] == terminator:
            break
    # return the received data, excluding the terminator sequence
    return b''.join(chunks)[:-len(terminator)].decode()



def server():
    HOST, PORT = '127.0.0.1', 65432
    BUFFER_SIZE = 1024
    TIMEOUT = 10000  # set a timeout of 5 seconds

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print("Server is waiting for connection...")

        model = LitClassifier.load_from_checkpoint(
        # NY model
            "./epoch=5-val_loss=363.05.ckpt")
        # COL model
        #     "./epoch=19-val_loss=nan.ckpt")
        model.to("cuda:7")
        trainer = pl.Trainer(accelerator="gpu", devices=[7])

        while True:
            conn, addr = s.accept()
            conn.settimeout(TIMEOUT)  # set a timeout on the connection
            with conn:
                print("Connected by", addr)
                try:
                    received_data = receive_all(conn, BUFFER_SIZE)

                    parts = received_data.split('|')
                    if len(parts) < 5:
                        print("Invalid data received")
                        continue

                    edge_attribute, edge_index, graph_data, tabular_data_str, tabular_data2_str = parts[:5]
                    tabular_data1 = torch.tensor(list(map(float, tabular_data_str.split(','))), dtype=torch.float)
                    tabular_data2 = torch.tensor(list(map(float, tabular_data2_str.split(','))), dtype=torch.float)
                    batch = process_data(edge_attribute, edge_index, graph_data, tabular_data1, tabular_data2)
                    batch = batch.to("cuda:7")

                    y_pred_class = model.test_with_input(batch)
                    predictions = ' '.join(map(str, y_pred_class))
                    conn.sendall(predictions.encode())

                except socket.timeout:
                    print("Timeout: No data received from", addr)

if __name__ == "__main__":
    server()
