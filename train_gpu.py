import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load dataset
X = np.load("X.npy")
y = np.load("y.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# move data to GPU
X = X.permute(0,3,1,2)   # channel first for pytorch
X = X.to(device)
y = y.to(device)

# CNN model
class FingerCNN(nn.Module):
    def __init__(self):
        super(FingerCNN,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*6*6,128),
            nn.ReLU(),
            nn.Linear(128,6)
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x


model = FingerCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 10

for epoch in range(epochs):

    outputs = model(X)
    loss = criterion(outputs,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

torch.save(model.state_dict(),"finger_model_gpu.pth")

print("Training complete")