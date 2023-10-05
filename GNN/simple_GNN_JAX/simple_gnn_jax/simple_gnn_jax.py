'''
Simple GCN using JAX to classify q vs. g jets

Parts of this are based on:
 - https://github.com/deepmind/jraph
 - https://colab.research.google.com/github/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb
 - https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html
'''

import tqdm
import time

import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state as flax_train_state
from flax.training import checkpoints as flax_checkpoints
import optax
import jraph
import networkx

import energyflow

##########################################
def main():
    print("Using jax", jax.__version__)
    print(f'Devices: {jax.devices()}')
    print()

    # Set some parameters
    n_jets = 1000
    n_train = int(0.8*n_jets)
    hidden_dim = 8
    n_output_classes = 2
    batch_size = 1
    n_epochs = 10
    learning_rate = 1.e-3

    #----------------------------------------
    # Construct dataset
    # Note: For this part, we are free to use numpy arrays and modify variables 
    #       since we will only @jit the training step for a given batch
    #----------------------------------------

    # Quark vs. gluon jet tagging data set
    # Shape of the data set: (jets, particles, (p_Ti,eta_i,phi_i,particle ID_i))
    X, y = energyflow.datasets.qg_jets.load(n_jets)
    n_particles = X.shape[1]
    n_features = X.shape[2]
    print(f'Dataset (n_jets, n_particles, features): {X.shape}')
    print(f'Labels: {y.shape}')
    print()

    # Split into train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=n_train, random_state=42)

    # Construct fully-connected jraph graphs
    #  - Nodes: particle four-vectors
    #  - Edges: no edge features
    graphs_list_train = _construct_graphs(X_train)
    graphs_list_val = _construct_graphs(X_val)

    #----------------------------------------
    # Define our GNN model
    #
    # We should use the functional programming paradigm:
    #  - Sequence of transformations -- never modify previous objects in the sequence 
    # i.e.
    #  - Pure functions: no side effects, no randomness
    #
    # And we should use (immutable) jax.numpy arrays
    #----------------------------------------
    print('Constructing GCN classifier using JAX...')

    # The DNN should be constructed from a module object (which can contain submodules)
    model = GCN(hidden_dim=hidden_dim, n_output_classes=n_output_classes)
    print(model)

    # Create a random number generator, to be used for initializing model weights
    key = jax.random.PRNGKey(42)
    key, _, r_init = jax.random.split(key, 3)

    # Initialize the model weights
    params = model.init(r_init, graphs_list_train[0])
    print(params)
    print()

    #----------------------------------------
    # Train the model
    #
    # We @jit the training step
    #----------------------------------------
    print('Training the model...')

    optimizer = optax.adam(learning_rate=learning_rate)

    # Create TrainState object to manage training
    # The state is never modified, but rather we will construct a new updated state at each training step
    model_state = flax_train_state.TrainState.create(apply_fn=model.apply,
                                                     params=params,
                                                     tx=optimizer)
    
    trained_model_state = train_model(model_state, graphs_list_train, y_train, 
                                                   graphs_list_val, y_val,
                                                   num_epochs=n_epochs, batch_size=batch_size)

    # Save model
    flax_checkpoints.save_checkpoint(ckpt_dir='my_checkpoints/',  # Folder to save checkpoint in
                                     target=trained_model_state,  # What to save. To only save parameters, use model_state.params
                                     step=n_epochs,  # Training step or other metric to save best model on
                                     prefix='my_dnn',  # Checkpoint file name prefix
                                     overwrite=True   # Overwrite existing checkpoint files
                                    )
    #loaded_model_state = flax_checkpoints.restore_checkpoint(ckpt_dir='my_checkpoints/',   # Folder with the checkpoints
    #                                                         target=model_state,   # (optional) matching object to rebuild state in
    #                                                         prefix='my_model'  # Checkpoint file name prefix
    #                                                        )
    print()

#----------------------------------------
def train_model(state, graphs_list_train, y_train, graphs_list_val, y_val, num_epochs=100, batch_size=1):
    '''
    Main function to train model

    We construct a new state using each batch, which serves as input state for the next batch
    '''
    print(f'Training...')
    start_time = time.time()
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    for epoch in range(num_epochs):

        # Training
        train_loss = []
        train_acc = []
        for batch_graphs, batch_labels in batched_data(graphs_list_train, y_train, batch_size):
            # Here, batch_graphs is a batched GraphsTuple and batch_labels is a numpy array slice
            state, loss, acc = train_step(state, batch_graphs, batch_labels)
            train_loss.append(loss)
            train_acc.append(acc)
        train_loss_per_epoch.append(jnp.mean(jnp.array(train_loss)))

        # Evaluation
        val_loss = []
        val_acc = []
        val_logits = []
        for batch_graphs, batch_labels in batched_data(graphs_list_val, y_val, batch_size):
            loss, (acc, logits) = eval_step(state, batch_graphs, batch_labels)
            val_loss.append(loss)
            val_acc.append(acc)
            val_logits.append(logits)
        val_loss_per_epoch.append(jnp.mean(jnp.array(val_loss)))

        print(f'Epoch: {epoch}')
        print(f'  Train Acc: {jnp.mean(jnp.array(train_acc)):.4f}')
        print(f'  Val Acc: {jnp.mean(jnp.array(val_acc)):.4f}')

    print("Done!")
    print()
    end_time = time.time()
    print(f'Training time: {end_time-start_time:.2f} seconds')
    print()

    ## Plot loss
    plt.figure(1,figsize=(10,10))
    plt.plot(train_loss_per_epoch)
    plt.plot(val_loss_per_epoch)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.savefig('loss_gnn.png')
    plt.close('all')
    
    ## Plot a ROC curve
    positive_class_scores = jnp.concatenate(val_logits, axis=0)[:,1]
    roc_curve = sklearn.metrics.roc_curve(y_val, positive_class_scores)
    fpr, tpr, thresholds = roc_curve
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.savefig('roc_curve_gnn.png')
    plt.close('all')

    return state

#----------------------------------------
@jax.jit
def train_step(state, batch_graphs, batch_labels):
    '''
    Main training step – determine gradients and update parameters
    '''
    # Determine gradients for current model, parameters and batch
    (loss, (acc,_)), grads = jax.value_and_grad(calculate_loss_acc,  # Function to calculate the loss
                           argnums=1,           # Parameters are second argument of the function
                           has_aux=True         # Function has additional outputs, here accuracy
                          )(state, state.params, (batch_graphs, batch_labels))
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss, acc

#----------------------------------------
@jax.jit
def eval_step(state, batch_graphs, batch_labels):
    '''
    Main evaluation step (no need for gradients)
    '''
    loss, (acc, logits) = calculate_loss_acc(state, state.params, (batch_graphs, batch_labels))
    return loss, (acc, logits)

#----------------------------------------
def calculate_loss_acc(state, params, batch):
    '''
    Calculate the loss and accuracy for a batch of data
    '''

    # Obtain the logits and predictions of the model for the input data
    data_input, labels = batch
    logits = state.apply_fn(params, data_input)
    labels_predicted = (logits > 0).astype(jnp.float32)

    # Calculate the loss and accuracy
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (labels_predicted == labels).mean()
    return loss, (acc, logits) 

#----------------------------------------
def batched_data(graphs_list, labels, batch_size):
    '''
    Construct a batch of data from a list of GraphsTuples

    TODO: There are two possible ways to batch efficiently such that we can take advantage of @jit without recompilation as the batch size changes:
            - Implicit batching: Construct a single GraphsTuple for a batch, and zero-pad that
            - Explicit batching: Zero-pad all individual graphs to ensure a fixed input size

    Note: we do not use PyTorch DataLoader, since it only works on CPU, which will require inefficient transferring during training

    :param graphs_list (list of jraph.GraphsTuple): A list containing individual graph data.
    :param labels (array-like): The labels associated with each graph.
    :param batch_size (int): Number of graphs to batch together.
    '''
    n = len(labels)
    for i in range(0, n, batch_size):
        yield jraph.batch(graphs_list[i:i+batch_size]), labels[i:i+batch_size]

#----------------------------------------
def _construct_graphs(X):
    '''
    Construct jraph GraphsTuples from energyflow dataset

    :param X: 3D array of shape (n_graphs, n_nodes, n_features) i.e. (jets, particles, (p_Ti,eta_i,phi_i,particle ID_i))
              where the nodes are zero-padded to fixed length

    :return graphs_list: List of jraph GraphsTuples, each consisting of a single graph
                         We preserve the initial zero-padding so that we can @jit the training step.
                         Since all graphs are zero-padded to the same dimension, we can also easily batch them
                         (at the expense of inefficient memory usage).
    '''
    print(f'Constructing jraph GraphsTuples from energyflow dataset...')

    graphs_list = []
    max_nodes = X.shape[1]
    max_edges = max_nodes*(max_nodes-1)
    for jet in tqdm.tqdm(X, desc=f'  Constructing fully-connected jraphs graphs:', total=X.shape[0]):

        # Construct fully-connected edges based on the padded nodes – Remove diagonal to avoid self-connections
        j, k = jnp.meshgrid(jnp.arange(max_nodes), jnp.arange(max_nodes))
        mask = j != k
        senders, receivers = j[mask], k[mask]

        # Create a GraphsTuple for a single graph
        graphs_tuple = jraph.GraphsTuple(
            n_node=jnp.array([max_nodes]),
            n_edge=jnp.array([max_edges]),
            nodes=jet,
            edges=None,  
            globals=None,
            senders=senders,
            receivers=receivers
        )

        graphs_list.append(graphs_tuple)

    # Draw an example
    example_graph = graphs_list[0]
    filename = "example_graph.png"
    print("Example:")
    print("  nodes:", jnp.sum(example_graph.n_node))
    print("  edges:", jnp.sum(example_graph.n_edge))
    _draw_and_save_graph(example_graph, filename=filename)
    print(f'Saved example graph to {filename}')

    return graphs_list

#----------------------------------------
def _draw_and_save_graph(graph, filename):
    '''
    Draw and save a specific graph from a GraphsTuple.
    
    Args:
    - graph (jraph.GraphsTuple): The GraphsTuple containing a single graph
    - filename (str): Path to save the drawn graph.
    '''

    # Transfer nodes and edges from device to host
    nodes_data = jax.device_get(graph.nodes)
    senders_data = jax.device_get(graph.senders)
    receivers_data = jax.device_get(graph.receivers)

    # Create a networkx graph and add nodes and edges
    G = networkx.Graph()
    for i, node_feature in enumerate(nodes_data):
        G.add_node(i, feature=node_feature)
    for sender, receiver in zip(senders_data, receivers_data):
        G.add_edge(int(sender), int(receiver))

    # Draw the graph
    pos = networkx.spring_layout(G)
    networkx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color='gray')
    plt.savefig(filename)
    plt.close()

##########################################
class GCN(flax.linen.Module):
    hidden_dim : int
    n_output_classes : int

    #----------------------------------------
    def setup(self):
        self.gcn = jraph.GraphConvolution(update_node_fn=self.node_update_fn,
                                          aggregate_nodes_fn = jraph.segment_sum,
                                          add_self_edges = True,
                                          symmetric_normalization = True)
        self.mlp = MLP(hidden_dim=self.hidden_dim, output_dim=self.hidden_dim)
        self.output_layer = MLP(hidden_dim=self.hidden_dim, output_dim=self.n_output_classes)

    #----------------------------------------
    # Define node update function
    def node_update_fn(self, node_features: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(node_features)

    #----------------------------------------
    def __call__(self, graph):
        '''
        This function defines the forward pass of the network
        '''
        # GCN layer
        graph = self.gcn(graph)

        # Create graph representation by averaging node features for each graph in the batch
        mean_node_features = jnp.sum(graph.nodes, axis=1) / jnp.expand_dims(graph.n_node, axis=-1)

        # Return logits
        logits = self.output_layer(mean_node_features)
        return logits

##########################################
class MLP(flax.linen.Module):
    '''
    We define a simple MLP w/flax to be used as a component of our GNN.
    '''
    hidden_dim : int
    output_dim : int

    #----------------------------------------
    @flax.linen.compact
    def __call__(self, x):
        '''
        This function defines the forward pass of the network
        '''
        x = flax.linen.Dense(features=self.hidden_dim)(x)
        x = jax.nn.relu(x)
        x = flax.linen.Dense(features=self.output_dim)(x)
        return x

#----------------------------------------
#----------------------------------------
#----------------------------------------
if __name__ == '__main__':
    main()