from pathlib import Path

import numpy as np
import torch
import timm


SOURCE = Path("results/fc1_fp32.npz")
QKV_FILE = Path("results/qkv_fp32.npz")
FC2_FILE = Path("results/fc2_fp32.npz")

OUTPUT = Path("results/block0_fp32.npz")


def remove_batch(x):
    if x.ndim == 3 and x.shape[0] == 1:
        return x[0]

    return x


def main():
    frozen = np.load(SOURCE)

    image = frozen["image"].astype(np.float32)

    model = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=True,
    )

    model.eval()

    block = model.blocks[0]

    captured = {}

    # ------------------------------------------------------------
    # Block input X0
    # ------------------------------------------------------------
    def block_pre_hook(module, inputs):
        captured["x0"] = (
            inputs[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

    # ------------------------------------------------------------
    # LN1 output
    # ------------------------------------------------------------
    def norm1_hook(module, inputs, output):
        captured["norm1"] = (
            output
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

    # ------------------------------------------------------------
    # Attention output
    #
    # This is after:
    # QKV -> QK^T -> Softmax -> AV -> Projection
    # ------------------------------------------------------------
    def attn_hook(module, inputs, output):
        captured["attn"] = (
            output
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

    # ------------------------------------------------------------
    # Input of LN2 = first residual output
    # ------------------------------------------------------------
    def norm2_pre_hook(module, inputs):
        captured["residual1"] = (
            inputs[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

    # ------------------------------------------------------------
    # LN2 output
    # ------------------------------------------------------------
    def norm2_hook(module, inputs, output):
        captured["norm2"] = (
            output
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

    # ------------------------------------------------------------
    # MLP output
    # ------------------------------------------------------------
    def mlp_hook(module, inputs, output):
        captured["mlp"] = (
            output
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

    # ------------------------------------------------------------
    # Final Block 0 output
    # ------------------------------------------------------------
    def block_hook(module, inputs, output):
        captured["output"] = (
            output
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

    hooks = [
        block.register_forward_pre_hook(
            block_pre_hook
        ),

        block.norm1.register_forward_hook(
            norm1_hook
        ),

        block.attn.register_forward_hook(
            attn_hook
        ),

        block.norm2.register_forward_pre_hook(
            norm2_pre_hook
        ),

        block.norm2.register_forward_hook(
            norm2_hook
        ),

        block.mlp.register_forward_hook(
            mlp_hook
        ),

        block.register_forward_hook(
            block_hook
        ),
    ]

    with torch.no_grad():
        _ = model(
            torch.from_numpy(image)
        )

    for h in hooks:
        h.remove()

    for key in list(captured.keys()):
        captured[key] = remove_batch(
            captured[key]
        )

    # ------------------------------------------------------------
    # Parameters for LayerNorm
    # ------------------------------------------------------------
    norm1_gamma = (
        block.norm1.weight
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    norm1_beta = (
        block.norm1.bias
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    norm2_gamma = (
        block.norm2.weight
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    norm2_beta = (
        block.norm2.bias
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    eps1 = float(block.norm1.eps)
    eps2 = float(block.norm2.eps)

    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------
    for name in [
        "x0",
        "norm1",
        "attn",
        "residual1",
        "norm2",
        "mlp",
        "output",
    ]:
        assert captured[name].shape == (
            197,
            192,
        ), (
            name,
            captured[name].shape,
        )

    # Residual checks
    residual1_calc = (
        captured["x0"]
        + captured["attn"]
    )

    residual2_calc = (
        captured["residual1"]
        + captured["mlp"]
    )

    residual1_error = float(
        np.max(
            np.abs(
                residual1_calc
                - captured["residual1"]
            )
        )
    )

    residual2_error = float(
        np.max(
            np.abs(
                residual2_calc
                - captured["output"]
            )
        )
    )

    # ------------------------------------------------------------
    # Cross-check against files already generated.
    # ------------------------------------------------------------
    qkv = np.load(QKV_FILE)

    qkv_input = remove_batch(
        qkv["input"]
    )

    fc1_input = remove_batch(
        frozen["input"]
    )

    fc2 = np.load(FC2_FILE)

    fc2_output = remove_batch(
        fc2["output"]
    )

    norm1_vs_qkv = float(
        np.max(
            np.abs(
                captured["norm1"]
                - qkv_input
            )
        )
    )

    norm2_vs_fc1 = float(
        np.max(
            np.abs(
                captured["norm2"]
                - fc1_input
            )
        )
    )

    mlp_vs_fc2 = float(
        np.max(
            np.abs(
                captured["mlp"]
                - fc2_output
            )
        )
    )

    np.savez(
        OUTPUT,

        x0=captured["x0"],

        norm1=captured["norm1"],

        attn=captured["attn"],

        residual1=captured["residual1"],

        norm2=captured["norm2"],

        mlp=captured["mlp"],

        output=captured["output"],

        norm1_gamma=norm1_gamma,
        norm1_beta=norm1_beta,

        norm2_gamma=norm2_gamma,
        norm2_beta=norm2_beta,

        norm1_eps=np.array(
            eps1,
            dtype=np.float32,
        ),

        norm2_eps=np.array(
            eps2,
            dtype=np.float32,
        ),
    )

    print("Saved:", OUTPUT)

    print()
    print("Block 0 shapes")
    print("--------------------------------")

    for name in [
        "x0",
        "norm1",
        "attn",
        "residual1",
        "norm2",
        "mlp",
        "output",
    ]:
        print(
            f"{name:10s}:",
            captured[name].shape,
        )

    print()
    print("Residual consistency")
    print("--------------------------------")
    print(
        "Residual 1 max error:",
        residual1_error,
    )
    print(
        "Residual 2 max error:",
        residual2_error,
    )

    print()
    print("Previous-file consistency")
    print("--------------------------------")
    print(
        "LN1 vs QKV input max error:",
        norm1_vs_qkv,
    )

    print(
        "LN2 vs FC1 input max error:",
        norm2_vs_fc1,
    )

    print(
        "MLP vs FC2 output max error:",
        mlp_vs_fc2,
    )

    print()
    print("LayerNorm parameters")
    print("--------------------------------")
    print("LN1 eps:", eps1)
    print("LN2 eps:", eps2)

    print()
    print(
        "LN1 gamma range:",
        float(np.min(norm1_gamma)),
        "to",
        float(np.max(norm1_gamma)),
    )

    print(
        "LN2 gamma range:",
        float(np.min(norm2_gamma)),
        "to",
        float(np.max(norm2_gamma)),
    )


if __name__ == "__main__":
    main()
