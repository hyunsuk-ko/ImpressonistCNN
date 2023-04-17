# define the model to have the pre-trained resnet 50 parameters
model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
# freeze all of the parameters in the model
for param in model.parameters():
  param.requires_grad = False
# unfreeze the parameters in the last residual block of the architecture
for name, param in model.named_parameters():
  for i in [4]:
    if name.startswith(f'layer{i}'):
      param.requires_grad = True
# EDIT DROPOUT RATE HERE (dropout actually doesn't help here, so let's use L2 regularization instead)
DO = 0.0
# construct the fully connected head which will receive the flattened convolutional output
model.fc = nn.Sequential(
               nn.Linear(2048, 512),
               nn.BatchNorm1d(512),
               nn.ReLU(inplace=True),
               nn.Dropout(DO),
               
               nn.Linear(512, 128),
               nn.BatchNorm1d(128),
               nn.ReLU(inplace=True),
               nn.Dropout(DO),

              #  nn.Linear(128, 32),
              #  nn.BatchNorm1d(32),
              #  nn.ReLU(inplace=True),
              #  #nn.Dropout(DO),

               nn.Linear(128, len(artists)))

# print all the named parameters in the model to confirm that the correct ones are frozen and unfrozen
for name, param in model.named_parameters():
  print(name, param.requires_grad)

# load the model to device
model = model.to(device)
