import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Utility function for model checkpointing.
def save_checkpoint(epoch, model, opt, dir):
    """Utility function for model checkpointing.

    Args:
        epoch (int): The current epoch number.
        model (torch.nn.Module): The model to save the state of.
        opt (torch.optim.Optimizer): The optimizer to save the state of.
        dir (str): The directory where the checkpoint will be saved.

    Saves the model state and optimizer state to a file named with the epoch number.
    """
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
        }, os.path.join(dir, f'checkpoint-{epoch}.pt'))

# Utility function to load a model checkpoint.
def load_checkpoint(fname, model, opt=None):
    """
    Load a model checkpoint from a specified file.

    Parameters:
    fname (str): The file name of the checkpoint.
    model (torch.nn.Module): The model instance to load the state into.
    opt (optional): The optimizer instance to load the state into. Default is None.

    Returns:
    torch.nn.Module: The model with the loaded state.
    """
    checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint['model_state_dict'])
    if opt:
        opt.load_state_dict(checkpoint['opt_state_dict'])
    return model

# A simple nn.Module for a variable depth and width MLP.
class PolicyNet(nn.Module):
    """
    A simple, but generic, policy network with a variable number of hidden layers.

    Attributes:
        hidden (nn.Sequential): A sequential container for all hidden layers.
        out (nn.Linear): The output layer that maps the hidden layer output to action space.
    """

    def __init__(self, env, n_hidden=1, width=128):
        """
        Initializes the PolicyNet instance with specified parameters.

        Args:
            env: The gym environment used to determine the input and output dimensions.
            n_hidden (int): The number of hidden layers in the network.
            width (int): The number of neurons in each hidden layer.
        """
        super().__init__()
        hidden_layers = [nn.Linear(env.observation_space.shape[0], width), nn.ReLU()]
        hidden_layers += [nn.Linear(width, width), nn.ReLU()] * (n_hidden - 1)
        self.hidden = nn.Sequential(*hidden_layers)
        self.out = nn.Linear(width, env.action_space.n)

    def forward(self, s):
        """
        Forward pass through the network.

        Args:
            s: The input state from the environment.

        Returns:
            torch.Tensor: The softmax probabilities over the action space.
        """
        s = self.hidden(s)
        s = F.softmax(self.out(s), dim=-1)
        return s
