import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from functions import config

# Define the base model
baseModel = resnet50(pretrained=True)
baseModel = nn.Sequential(*list(baseModel.children())[:-1])  # Remove the last fully connected layer

# Define the custom head
headModel = nn.Sequential(
    nn.Flatten(),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, len(config.CLASSES)),
    nn.Softmax(dim=1)
)

# Combine the base model and the head
model = nn.Sequential(baseModel, headModel)

# Set all parameters in the base model to be non-trainable
for param in baseModel.parameters():
    param.requires_grad = False

# Define the optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=config.MIN_LR, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Move the model to the GPU if available
device = torch.device("cuda")
model.to(device)


# Define the federated training function
def federated_train(model, train_loader, optimizer, criterion, device):
    model.train()
    client_models = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % config.LOG_INTERVAL == 0:
            print(f"Train Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

        # Save a copy of the client model's state_dict
        client_models.append(model.state_dict())

    # Aggregating gradients or model updates using FedAvg
    server_state = model.state_dict()
    for key in server_state:
        server_state[key] = torch.stack([client_state[key] for client_state in client_models]).mean(0)
    model.load_state_dict(server_state)


# Set the model in the FedML training loop
for epoch in range(config.NUM_EPOCHS):
    federated_train(model, train_loader, optimizer, criterion, device)
