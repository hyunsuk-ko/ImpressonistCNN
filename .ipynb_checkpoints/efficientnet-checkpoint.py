from torchvision.models import EfficientNet_B0_Weights

# Load the pre-trained EfficientNet-B0 model
model = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

# Freeze all layers except the last two blocks
for name, param in model.named_parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if 'blocks.15' in name or 'blocks.16' in name or 'fc' in name:
        param.requires_grad = True

# Modify the fully connected head
DO = 0.
model.fc = nn.Sequential(
    nn.Linear(1280, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(DO),

    nn.Linear(512, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
    nn.Dropout(DO),

    nn.Linear(128, len(artists))
)

# Print the named parameters to confirm that the correct ones are frozen and unfrozen
for name, param in model.named_parameters():
    print(name, param.requires_grad)

# Load the model to device
model = model.to(device)
