import torch
import pickle
import syft as sy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Assuming load_pickle is defined in utils.py
from .utils import load_pickle

class SklearnWrapper(torch.nn.Module):
    def __init__(self, sk_model):
        super(SklearnWrapper, self).__init__()
        self.sk_model = sk_model

    def forward(self, x):
        # Ensure x is a tensor before passing it to the model
        x = x.detach().numpy()  # Convert tensor to numpy array
        predictions = self.sk_model.predict(x)
        return torch.tensor(predictions, dtype=torch.float32)

def save_tensor():
    # Load your scikit-learn model
    sk_model = load_pickle('KNN')

    # Create a virtual worker (in this case, let's use bob as an example)
    hook = sy.TorchHook(torch)  # Hook Torch
    bob = sy.VirtualMachine(name="bob")  # Create a virtual machine
    bob_client = bob.get_root_client()  # Get a client to interact with the virtual machine

    # Wrap your scikit-learn model in the PyTorch wrapper
    wrapper_model = SklearnWrapper(sk_model)

    # Save PyTorch model
    torch.save(wrapper_model.state_dict(), 'tmp/models/KNN_DBT.pth')

    # Send the model to the virtual worker
    wrapper_model_ptr = wrapper_model.send(bob_client)

    # Example of how to use the model
    data_to_predict = torch.tensor(np.random.randn(10, 10), dtype=torch.float32).send(bob_client)
    predictions = wrapper_model_ptr(data_to_predict).get()  # Get predictions from the model

    print("Predictions:", predictions.numpy())
