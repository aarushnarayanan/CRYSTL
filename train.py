import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
from utils.helpers import BottleDataset, collate_fn
import matplotlib.pyplot as plt
import cv2
import os
from torchvision.models import MobileNet_V2_Weights
import sys
import torch

# Configuration
NUM_EPOCHS = 10
BATCH_SIZE = 4
CLASS_NAMES = ['bottle', 'can', 'jar', 'other']  # UPDATE THESE TO MATCH YOUR DATASET
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Checking if GPU available to run on
print("Is GPU available?", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("Current device index:", torch.cuda.current_device(), flush=True)
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()), flush=True)
print("Using device:", DEVICE, flush=True)

sys.stdout.flush()  # Force any buffered output to print now

# Download dataset
#rf = Roboflow(api_key="YOUR_API_KEY_HERE")  # ← REPLACE WITH YOUR KEY
#project = rf.workspace("chemical").project("march13")
#dataset = project.version(1).download("coco")

# Model setup
def create_model(num_classes):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1  
    backbone = torchvision.models.mobilenet_v2(weights=weights).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    return FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)


# Training function
def train():
    # Load data
    train_data = BottleDataset("train")
    valid_data = BottleDataset("valid")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = create_model(len(CLASS_NAMES) + 1)  # +1 for background class
    model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        # Training phase
        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in valid_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                val_loss += sum(model(images, targets).values()).item()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {total_loss/len(train_loader):.2f} | Val Loss: {val_loss/len(valid_loader):.2f}")
    
    torch.save(model.state_dict(), "bottle_model.pth")
    print("Model saved!")

# Detection function
def detect_image(image_path):
    model = create_model(len(CLASS_NAMES) + 1)
    model.load_state_dict(torch.load("bottle_model.pth", map_location=DEVICE))
    model.eval()
    
    img = cv2.imread(image_path)
    img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        prediction = model(img_tensor)[0]
    
    # Draw predictions
    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f"{CLASS_NAMES[label-1]} {score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    train()
    # Uncomment to test:
    # detect_image("test.jpg")