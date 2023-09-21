'''
Simple GNN using PyTorch to classify q vs. g jets
'''

import os
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics

import torch
import torch_geometric
import networkx

import energyflow

#----------------------------------------
def simple_gnn_pytorch():

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using {device} device")
    print()

    # Set some parameters
    n_jets = 100000
    n_train = int(0.8*n_jets)
    hidden_dim = 64
    n_output_classes = 2
    batch_size = 128
    n_epochs = 10
    learning_rate = 1.e-3

    #----------------
    # Load data and create data loaders

    # Quark vs. gluon jet tagging data set
    # Shape of the data set: (jets, particles, (p_Ti,eta_i,phi_i,particle ID_i))
    X, y = energyflow.datasets.qg_jets.load(n_jets)
    n_particles = X.shape[1]
    n_features = X.shape[2]
    print(f'Dataset (n_jets, n_particles, features): {X.shape}')
    print(f'Labels: {y.shape}')
    print()

    #----------------
    # Construct fully-connected PyG graphs
    #  - Nodes: particle four-vectors
    #  - Edges: no edge features
    print(f'Constructing PyG particle graphs from energyflow dataset...')
    graph_list = []
    args = [(x, y[i]) for i, x in enumerate(X)] 
    for arg in tqdm.tqdm(args, desc=f'  Constructing fully-connected PyG graphs:', total=len(args)):
        graph_list.append(_construct_particle_graph_pyg(arg))

    #----------------
    # Construct DataLoader objects
    # PyG implements its own DataLoader that creates batches with a block-diagonal adjacency matrix,
    #   which allows different numbers of nodes/edges within each batch 
    # See: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#mini-batches
    print('Constructing DataLoader...')
    train_dataset = graph_list[:n_train]
    test_dataset = graph_list[n_train:n_jets]
    train_dataloader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch_geometric.loader.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print('Done.')
    print()
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    example_graph = graph_list[0]
    print(f'Number of features: {example_graph.num_features}')
    print(f'Number of node features: {example_graph.num_node_features}')
    print(f'Number of edge features: {example_graph.num_edge_features}')
    print(f'  Has self-loops: {example_graph.has_self_loops()}')
    print(f'  Is undirected: {example_graph.is_undirected()}')
    print(f'Example graph:')
    print(f'  Number of nodes: {example_graph.num_nodes}')
    print(f'  Number of edges: {example_graph.num_edges}')
    print(f'  Adjacency: {example_graph.edge_index.shape}')
    print()

    #----------------
    # Create model
    n_input_features = example_graph.num_node_features
    model = GCN(n_input_features, hidden_dim, n_output_classes)
    print(model)
    print()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #----------------
    # Train and test model
    print(f'Training...')
    start_time = time.time()
    
    model.to(device)
    model.train()

    loss_list = []
    for epoch in range(1, n_epochs+1):
        for batch in train_dataloader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)   # Forward pass
            loss = loss_fn(out, batch.y)                          # Compute loss
            loss_list.append(loss.item())                         # Store loss for plotting
            loss.backward()                                       # Compute gradients
            optimizer.step()                                      # Update model parameters
            optimizer.zero_grad()                                 # Clear gradients.

        train_acc = _accuracy(train_dataloader, model, device)
        test_acc = _accuracy(test_dataloader, model, device)
        print(f'Epoch: {epoch}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    print("Done!")
    print()

    # Plot loss
    plt.figure(1,figsize=(10,10))
    plt.plot(loss_list)
    plt.xlabel('Training batch')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.savefig('loss_gnn.png')

    # Evaluate model on test set
    print(f'Evaluate on test set...')
    pred_graphs_list = []
    label_graphs_list = []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            pred_graph = model(batch.x, batch.edge_index, batch.batch)
            pred_graphs_list.append(pred_graph.cpu().data.numpy())
            label_graphs_list.append(batch.y.cpu().data.numpy())
        pred_graphs = np.concatenate(pred_graphs_list, axis=0)
        label_graphs = np.concatenate(label_graphs_list, axis=0)
        auc = sklearn.metrics.roc_auc_score(label_graphs, pred_graphs[:,1])
        roc_curve = sklearn.metrics.roc_curve(label_graphs, pred_graphs[:,1])

    # Plot a ROC curve
    with torch.no_grad():
        fpr, tpr, thresholds = roc_curve
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.savefig('roc_curve_gnn.png')
    print("Done!")

#---------------------------------------------------------------
def _construct_particle_graph_pyg(args):
    '''
    Construct a single PyG graph for the particle-based GNNs from the energyflow dataset
    '''
    x, label = args

    # Node features -- remove the zero pads
    x = x[~np.all(x == 0, axis=1)]
    node_features = torch.tensor(x,dtype=torch.float)

    # Edge connectivity -- fully connected
    adj_matrix = np.ones((x.shape[0],x.shape[0])) - np.identity((x.shape[0]))
    row, col = np.where(adj_matrix)
    coo = np.array(list(zip(row,col)))
    edge_indices = torch.tensor(coo)
    edge_indices_long = edge_indices.t().to(torch.long).view(2, -1)

    # Construct graph as PyG data object
    graph_label = torch.tensor(label, dtype=torch.int64)
    graph = torch_geometric.data.Data(x=node_features, edge_index=edge_indices_long, edge_attr=None, y=graph_label)
    return graph

#---------------------------------------------------------------
def _accuracy(loader, model, device):
    correct = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == batch.y).sum())
    accuracy = correct / len(loader.dataset)
    return accuracy

############################################
# Define model
############################################
##################################################################
class GCN(torch.nn.Module):
    def __init__(self, n_input_features, hidden_dim, n_output_classes, dropout_rate=0.):
        '''
        :param n_input_features: number of input features
        :param hidden_dims: list of hidden dimensions
        :param n_output_classes: number of output classes
        :param dropout_rate: dropout rate
        '''
        super(GCN,self).__init__()

        # GNN layers
        self.conv1 = torch_geometric.nn.GCNConv(n_input_features, hidden_dim)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_dim, hidden_dim)

        # Dropout layer (by default, only active during training -- i.e. disabled with mode.eval() )
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Fully connected layer for graph classification
        self.fc = torch.nn.Linear(hidden_dim, n_output_classes)

    def forward(self, x, edge_index, batch):

        # GNN layers
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)

        # Global mean pooling (i.e. avg node features across each graph) to get a graph-level representation for graph classification
        # This requires the batch tensor, which keeps track of which nodes belong to which graphs in the batch.
        x = torch_geometric.nn.global_mean_pool(x, batch)

        # Fully connected layer for graph classification
        # Note: For now, we don't apply dropout here, since the dimension is small
        x = self.fc(x)
        return x

############################################
if __name__ == '__main__':
    simple_gnn_pytorch()