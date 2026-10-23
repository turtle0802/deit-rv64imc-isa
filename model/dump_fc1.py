import torch
import timm

MODEL_NAME = "deit_tiny_patch16_224"

model = timm.create_model(
    MODEL_NAME,
    pretrained=True
)

model.eval()

saved = {}


def fc1_hook(module, inputs, output):
    saved["input"] = inputs[0].detach().cpu()
    saved["output"] = output.detach().cpu()


handle = model.blocks[0].mlp.fc1.register_forward_hook(fc1_hook)

# 先用固定 random seed，確保每次都能重現
torch.manual_seed(0)

x = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    model(x)

handle.remove()

fc1_input = saved["input"]
fc1_output = saved["output"]

weight = model.blocks[0].mlp.fc1.weight.detach().cpu()
bias = model.blocks[0].mlp.fc1.bias.detach().cpu()

print("=== Block 0 MLP FC1 ===")
print()

print("Input shape :", tuple(fc1_input.shape))
print("Weight shape:", tuple(weight.shape))
print("Bias shape  :", tuple(bias.shape))
print("Output shape:", tuple(fc1_output.shape))

print()
print("=== Sample Values ===")

print("Input [0, 0, 0:8]:")
print(fc1_input[0, 0, 0:8])

print()
print("Weight [0, 0:8]:")
print(weight[0, 0:8])

print()
print("Bias [0]:")
print(bias[0])

print()
print("Output [0, 0, 0:8]:")
print(fc1_output[0, 0, 0:8])