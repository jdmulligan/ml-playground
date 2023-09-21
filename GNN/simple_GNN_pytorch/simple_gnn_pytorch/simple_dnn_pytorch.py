'''
Simple DNN using PyTorch to classify q vs. g jets

Based in part on: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
'''

import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

import energyflow

#----------------------------------------
def simple_dnn_pytorch():

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using {device} device")
    print()

    # Set some parameters
    n_jets = 500000
    batch_size = 128
    n_epochs = 10
    learning_rate = 1e-2

    #----------------
    # Load data and create data loaders

    # Quark vs. gluon jet tagging data set
    # Shape of the data set: (jets, particles, (p_Ti,eta_i,phi_i,particle ID_i))
    X, y = energyflow.datasets.qg_jets.load(n_jets)
    n_particles = X.shape[1]
    n_features = X.shape[2]
    print(f'Dataset: {X.shape}')
    print(f'Labels: {y.shape}')
    print()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    #----------------
    # Create model
    input_dim = n_particles * n_features
    model = MyModel(input_dim).to(device)
    print(model)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #----------------
    # Train and test model
    for t in range(n_epochs):

        # Train
        size = len(train_dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {t+1}: train loss per batch: {loss.item():>7f}")

        # Test
        size = len(test_dataloader.dataset)
        num_batches = len(test_dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"         test loss per batch: {test_loss:>8f}, accuracy: {(100*correct):>0.1f}%")

    print("Done!")

    # Plot a ROC curve
    model.eval()
    with torch.no_grad():
        y_pred = model(test_dataset.data.to(device)).cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_test, y_pred[:,1])
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.savefig('roc_curve_dnn.png')

############################################
# Define dataset
############################################
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.data = torch.from_numpy(X).float()
        self.labels = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

############################################
# Define model
############################################
class MyModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.dnn_stack = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.dnn_stack(x)
        return logits

############################################
if __name__ == '__main__':
    simple_dnn_pytorch()