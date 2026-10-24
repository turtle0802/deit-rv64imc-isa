import torch
import timm

MODEL_NAME = "deit_tiny_patch16_224"

print("Loading DeiT-Tiny...")

model = timm.create_model(
    MODEL_NAME,
    pretrained=True
)

model.eval()

num_params = sum(p.numel() for p in model.parameters())

x = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    features = model.forward_features(x)
    output = model(x)

print()
print("=== Model Information ===")
print(f"Parameters : {num_params:,}")

print()
print("=== Tensor Shapes ===")
print(f"Input      : {tuple(x.shape)}")
print(f"Features   : {tuple(features.shape)}")
print(f"Output     : {tuple(output.shape)}")

print()
print("DeiT-Tiny loaded successfully.")