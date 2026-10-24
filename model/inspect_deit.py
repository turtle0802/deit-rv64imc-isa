import timm

MODEL_NAME = "deit_tiny_patch16_224"

model = timm.create_model(
    MODEL_NAME,
    pretrained=True
)

model.eval()

print("=== DeiT-Tiny Structure ===")
print()

print("Number of Transformer Blocks:", len(model.blocks))
print("Embedding Dimension:", model.embed_dim)
print()

block = model.blocks[0]

print("=== Block 0 ===")
print(block)

print()
print("=== Important Weight Shapes ===")

print(
    "QKV weight       :",
    tuple(block.attn.qkv.weight.shape)
)

print(
    "QKV bias         :",
    tuple(block.attn.qkv.bias.shape)
)

print(
    "Attention proj   :",
    tuple(block.attn.proj.weight.shape)
)

print(
    "MLP FC1 weight   :",
    tuple(block.mlp.fc1.weight.shape)
)

print(
    "MLP FC2 weight   :",
    tuple(block.mlp.fc2.weight.shape)
)