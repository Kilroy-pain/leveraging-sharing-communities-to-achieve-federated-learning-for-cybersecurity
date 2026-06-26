import numpy as np
import torch
from torch import nn
from torch.optim import SGD

class StreamingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StreamingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def incremental_train(model, optimizer, criterion, data, labels):
    model.train()
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def merge_models(global_model, local_model, alpha=0.5):
    for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
        global_param.data = alpha * global_param.data + (1 - alpha) * local_param.data

def evaluate_model(model, data, labels):
    model.eval()
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    outputs = model(data)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == labels).sum().item() / labels.size(0)
    return accuracy

if __name__ == '__main__':
    # Dummy data for demonstration
    input_dim = 10
    hidden_dim = 20
    output_dim = 2
    num_samples = 100

    # Generate random data
    np.random.seed(42)
    data = np.random.rand(num_samples, input_dim)
    labels = np.random.randint(0, output_dim, num_samples)

    # Initialize models
    global_model = StreamingModel(input_dim, hidden_dim, output_dim)
    local_model = StreamingModel(input_dim, hidden_dim, output_dim)

    # Optimizers and loss function
    global_optimizer = SGD(global_model.parameters(), lr=0.01)
    local_optimizer = SGD(local_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Simulate streaming data and incremental training
    for i in range(num_samples):
        sample_data = data[i].reshape(1, -1)
        sample_label = np.array([labels[i]])
        incremental_train(local_model, local_optimizer, criterion, sample_data, sample_label)

    # Merge local model into global model
    merge_models(global_model, local_model, alpha=0.5)

    # Evaluate the merged global model
    accuracy = evaluate_model(global_model, data, labels)
    print(f"Accuracy of the merged global model: {accuracy:.2f}")