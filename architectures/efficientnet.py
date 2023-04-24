from torchvision.models import EfficientNet_B0_Weights

# Load the pre-trained EfficientNet-B0 model
model = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

# Freeze all layers except the last two blocks
for name, param in model.named_parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    for i in [7,8]:
        if name.startswith(f'features.{i}'):
            param.requires_grad = True

# Modify the fully connected head
DO = 0.
model.classifier = nn.Sequential(
    nn.Linear(1280, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(DO),

    nn.Linear(256, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(inplace=True),
    nn.Dropout(DO),

    nn.Linear(64, len(artists))
)

# Print the named parameters to confirm that the correct ones are frozen and unfrozen
for name, param in model.named_parameters():
    print(name, param.requires_grad)

# Load the model to device
model = model.to(device)
