from pathlib import Path

import numpy as np
import torch
import timm


SOURCE = Path("results/fc1_fp32.npz")
OUTPUT = Path("results/qkv_fp32.npz")


def main():
    frozen = np.load(SOURCE)

    image = frozen["image"].astype(np.float32)

    print("Frozen image shape:", image.shape)

    model = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=True,
    )

    model.eval()

    # Verify same pretrained model as FC1 experiment.
    saved_fc1_weight = frozen["weight"]

    current_fc1_weight = (
        model.blocks[0]
        .mlp.fc1.weight
        .detach()
        .cpu()
        .numpy()
    )

    err = np.max(
        np.abs(
            saved_fc1_weight
            - current_fc1_weight
        )
    )

    print(
        "FC1 model consistency max error:",
        err,
    )

    if err != 0:
        raise RuntimeError(
            "Loaded model differs from frozen FC1 model."
        )

    qkv = model.blocks[0].attn.qkv

    captured = {}

    def hook(module, inputs, output):
        captured["input"] = (
            inputs[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        captured["output"] = (
            output
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

    h = qkv.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(
            torch.from_numpy(image)
        )

    h.remove()

    x = captured["input"]
    y = captured["output"]

    if x.ndim == 3:
        x = x[0]

    if y.ndim == 3:
        y = y[0]

    weight = (
        qkv.weight
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    bias = (
        qkv.bias
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    assert x.shape == (197, 192)
    assert weight.shape == (576, 192)
    assert bias.shape == (576,)
    assert y.shape == (197, 576)

    # timm layout:
    # [token, 3(QKV), 3(heads), 64(dim)]
    split = y.reshape(
        197,
        3,
        3,
        64,
    )

    q = split[:, 0].transpose(1, 0, 2)
    k = split[:, 1].transpose(1, 0, 2)
    v = split[:, 2].transpose(1, 0, 2)

    assert q.shape == (3, 197, 64)
    assert k.shape == (3, 197, 64)
    assert v.shape == (3, 197, 64)

    np.savez(
        OUTPUT,
        input=x,
        weight=weight,
        bias=bias,
        output=y,
        q=q,
        k=k,
        v=v,
    )

    macs = 197 * 192 * 576

    print()
    print("Saved:", OUTPUT)

    print()
    print("QKV shapes")
    print("--------------------------------")
    print("Input :", x.shape)
    print("Weight:", weight.shape)
    print("Bias  :", bias.shape)
    print("Output:", y.shape)
    print("Q     :", q.shape)
    print("K     :", k.shape)
    print("V     :", v.shape)

    print()
    print("MACs:", f"{macs:,}")
    print(
        "Expected Xqmac8:",
        f"{macs // 8:,}",
    )

    print()
    print("First Q values:")
    print(q[0, 0, :8])


if __name__ == "__main__":
    main()
