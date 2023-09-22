'''
Basic NN classifier using JAX

Based on:
 - https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html
'''

import os
import tqdm

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state as flax_train_state
from flax.training import checkpoints as flax_checkpoints
import optax

import numpy as np
import torch

##########################################
def main():
    print("Using jax", jax.__version__)
    print(f'Devices: {jax.devices()}')
    print()

    # Training parameters
    learning_rate = 1.e-3
    n_epochs = 10

    #----------------------------------------
    # Construct dataset
    # Note: For this part, we are free to use static variables and numpy arrays
    #       since we will only @jit the training step for a given batch
    #----------------------------------------

    # Use pytorch Dataset and DataLoader class to construct XOR dataset and iterate through batches
    # Note: We want to collate the samples using ndarrays rather than pytorch tensors
    print('Constructing dataset using pytorch...')
    std = 0.3
    batch_size = 128
    dataset_train = MyDataset(N=2500, std=std, seed=42)
    dataset_val = MyDataset(N=500, std=std, seed=43)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, 
                                                    collate_fn=numpy_collate)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, 
                                                    collate_fn=numpy_collate)
    
    print(f"Size of training dataset: {len(dataset_train)} (x,y) pairs")
    print(f"Size of validation dataset: {len(dataset_val)} (x,y) pairs")
    sample_batch_inputs, sample_batch_labels = next(iter(data_loader_train))
    print(f'Sample batch:')
    print(f'  Inputs: {sample_batch_inputs.shape}')
    print(f'  Labels: {sample_batch_labels.shape}')
    print()

    #----------------------------------------
    # We can use flax to construct our DNN
    #
    # We should use the functional programming paradigm:
    #  - Sequence of transformations -- never modify previous objects in the sequence 
    # i.e.
    #  - Pure functions: no side effects, no randomness
    #
    # And we should use (immutable) jax.numpy arrays
    #----------------------------------------
    print('Constructing classifier using JAX...')

    # The DNN should be constructed from a module object (which can contain submodules)
    model = MyClassifier(num_hidden=8, num_outputs=1)
    print(model)

    # Create a random number generator, to be used for initializing model weights
    key = jax.random.PRNGKey(42)
    key, r_input, r_init = jax.random.split(key, 3)

    # Generate input
    input = jax.random.normal(r_input, (8, 2))

    # Initialize the model weights
    params = model.init(r_init, input)
    #print(params)
    print()

    #----------------------------------------
    # Train the model
    #
    # We @jit the training step
    #----------------------------------------
    print('Training the model...')

    optimizer = optax.sgd(learning_rate=learning_rate)

    # Create TrainState object to manage training
    # The state is never modified, but rather we will construct a new updated state at each training step
    model_state = flax_train_state.TrainState.create(apply_fn=model.apply,
                                                     params=params,
                                                     tx=optimizer)
    
    trained_model_state = train_model(model_state, data_loader_train, num_epochs=n_epochs)

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
    # Evaluate model
    #----------------------------------------
    accuracy = eval_model(trained_model_state, data_loader_val)
    print(f"Accuracy of the model: {100.0*accuracy:4.2f}%")
    print()

    # Run a few test points by directly computing the output of the model
    print('Running a few test points...')
    logits = model.apply(trained_model_state.params, 
                         jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    result = flax.linen.sigmoid(logits)
    print(f"Result: {result}")
    print()

#----------------------------------------
def train_model(state, data_loader, num_epochs=100):
    '''
    Main function to train model

    We construct a new state using each batch, which serves as input state for the next batch
    '''
    for epoch in tqdm.tqdm(range(num_epochs)):
        for batch in data_loader:
            state, loss, acc = train_step(state, batch)
            # We could use the loss and accuracy for logging here, e.g. in TensorBoard
    return state

#----------------------------------------
@jax.jit
def train_step(state, batch):
    '''
    Main training step
    '''
    # Determine gradients for current model, parameters and batch
    (loss, acc), grads = jax.value_and_grad(calculate_loss_acc,  # Function to calculate the loss
                                            argnums=1,           # Parameters are second argument of the function
                                            has_aux=True         # Function has additional outputs, here accuracy
                                           )(state, state.params, batch)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss, acc

#----------------------------------------
def calculate_loss_acc(state, params, batch):
    '''
    Calculate the loss and accuracy for a batch of data
    '''

    # Obtain the logits and predictions of the model for the input data
    data_input, labels = batch
    logits = state.apply_fn(params, data_input).squeeze(axis=-1)
    labels_predicted = (logits > 0).astype(jnp.float32)

    # Calculate the loss and accuracy
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (labels_predicted == labels).mean()
    return loss, acc

#----------------------------------------
def eval_model(state, data_loader):
    all_accuracies, batch_sizes = [], []
    for batch in data_loader:
        batch_acc = eval_step(state, batch)
        all_accuracies.append(batch_acc)
        batch_sizes.append(batch[0].shape[0])
    # Weighted average since some batches might be smaller
    accuracy = sum([a*b for a,b in zip(all_accuracies, batch_sizes)]) / sum(batch_sizes)
    return accuracy

#----------------------------------------
@jax.jit
def eval_step(state, batch):
    '''
    Main evaluation step (no need for gradients)
    '''
    # Determine the accuracy
    _, acc = calculate_loss_acc(state, state.params, batch)
    return acc

#----------------------------------------
def numpy_collate(batch):
    '''
    Collate a batch of samples into a single ndarray

    Taken from: https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
    '''
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)    

##########################################
class MyClassifier(flax.linen.Module):
    '''
    We define a simple DNN classifier with one hidden layer and one output layer.

    Define dataclass attributes: hidden dimension, number of layers, etc.
    '''
    num_hidden : int   # Number of hidden neurons
    num_outputs : int  # Number of output neurons

    #----------------------------------------
    def setup(self):
        '''
        Create the submodules we need to build the network
        Flax uses "lazy" initialization -- setup() called once before you call the model or access attributes
        '''
        self.linear1 = flax.linen.Dense(features=self.num_hidden)
        self.linear2 = flax.linen.Dense(features=self.num_outputs)

    #----------------------------------------
    def __call__(self, x):
        '''
        This function defines the forward pass of the network

        Note: we could alternately decorate with @nn.compact to tell Flax to look
              for defined submodules, rather than defining them in setup()
        '''
        x = self.linear1(x)
        x = flax.linen.tanh(x)
        x = self.linear2(x)
        return x

##########################################
class MyDataset(torch.utils.data.Dataset):
    '''
    Use pytorch to create dataset

    The dataset consists of:
      - (x,y) pairs, where x,y={0,1} plus gaussian noise
      - label: xor of x and y (0 if x=y, 1 if x!=y)
    '''
    #----------------------------------------
    def __init__(self, N=0, std=0.1, seed=42):
        '''
        Inputs:
            N: Number of data points
            std: Standard deviation of the noise
            seed: Seed for PRNG state to generate the data points
        '''
        super().__init__()

        # Generate data
        self.random_number_generator_numpy = np.random.RandomState(seed=seed)
        self.data = self.random_number_generator_numpy.randint(low=0, high=2, size=(N, 2)).astype(np.float32)
        self.label = (self.data.sum(axis=1) == 1).astype(np.int32)
        self.data += self.random_number_generator_numpy.normal(loc=0.0, scale=std, size=self.data.shape)

    #----------------------------------------
    def __len__(self):
        '''
        Return the number of data points in the dataset
        '''
        return self.label.shape[0]

    #----------------------------------------
    def __getitem__(self, i):
        '''
        Return the i-th data point of the dataset: (data, label)
        '''
        data_point = self.data[i]
        data_label = self.label[i]
        return data_point, data_label
    
#----------------------------------------
#----------------------------------------
#----------------------------------------
if __name__ == '__main__':
    main()