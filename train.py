import torch
import torch.nn as nn
import torch.optim as optim

# 1. Import your custom code from your other files!
# (Make sure the filenames match what you actually named them)
from model import PointNetClassifier 
from datasetloader import train_dataloader


# 2. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Instantiate the model and move it to the device
model = PointNetClassifier(num_classes=4).to(device)

# 4. Setup Loss Function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Run the Training Loop
epochs = 10 

for epoch in range(epochs):
    model.train() 
    running_loss = 0.0
    
    for inputs, labels in train_dataloader: 
        inputs, labels = inputs.to(device), labels.to(device)
        
        logits = model(inputs)                 
        loss = loss_fn(logits, labels)         
        optimizer.zero_grad()                  
        loss.backward()                        
        optimizer.step()                       
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")