import torch

# Replace with your actual file path
weights = torch.load('pretrained_weights.pth', map_location='cpu')

# Print the keys of the loaded state dict or object
print(type(weights))
if isinstance(weights, dict):
    print("Keys:", weights.keys())
