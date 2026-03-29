import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FingerCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32*14*14,128),
            nn.ReLU(),
            nn.Linear(128,6)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


model = FingerCNN().to(device)
model.load_state_dict(torch.load("finger_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    img = cv2.resize(frame,(64,64))
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        prediction = torch.argmax(output,1).item()

    cv2.putText(frame,f"Fingers: {prediction}",(30,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Finger Counter",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()