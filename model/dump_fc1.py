import torch
import timm
import numpy as np

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

generator = torch.Generator()
generator.manual_seed(0)

x_image = torch.randn(
    1, 3, 224, 224,
    generator=generator
)

print("Image sample:", x_image[0, 0, 0, 0:8])
print("Image sum   :", x_image.sum().item())

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

np.savez(
    "results/fc1_fp32.npz",

    image=x_image.detach().cpu().numpy(),

    input=fc1_input.numpy(),
    weight=weight.numpy(),
    bias=bias.numpy(),
    output=fc1_output.numpy()
)

print()
print("Saved results/fc1_fp32.npz")