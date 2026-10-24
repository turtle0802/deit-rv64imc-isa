from pathlib import Path
import numpy as np
import torch
import timm

SOURCE = Path("results/fc1_fp32.npz")
OUTPUT = Path("results/block0_mlp_fp32.npz")


def strip_batch(x):
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

    def save_hook(name):
        def hook(module, inputs, output):
            captured[name + "_input"] = (
                inputs[0].detach().cpu().numpy().copy()
            )
            captured[name + "_output"] = (
                output.detach().cpu().numpy().copy()
            )
        return hook

    hooks = [
        block.norm2.register_forward_hook(
            save_hook("norm2")
        ),
        block.mlp.fc1.register_forward_hook(
            save_hook("fc1")
        ),
        block.mlp.act.register_forward_hook(
            save_hook("gelu")
        ),
        block.mlp.fc2.register_forward_hook(
            save_hook("fc2")
        ),
        block.mlp.register_forward_hook(
            save_hook("mlp")
        ),
    ]

    with torch.no_grad():
        _ = model(torch.from_numpy(image))

    for h in hooks:
        h.remove()

    for k in captured:
        captured[k] = strip_batch(captured[k])

    # Exact topology checks.
    e1 = np.max(np.abs(
        captured["norm2_output"]
        - captured["fc1_input"]
    ))

    e2 = np.max(np.abs(
        captured["fc1_output"]
        - captured["gelu_input"]
    ))

    e3 = np.max(np.abs(
        captured["gelu_output"]
        - captured["fc2_input"]
    ))

    e4 = np.max(np.abs(
        captured["fc2_output"]
        - captured["mlp_output"]
    ))

    np.savez(
        OUTPUT,

        norm2_input=captured["norm2_input"],
        norm2_output=captured["norm2_output"],

        fc1_input=captured["fc1_input"],
        fc1_weight=block.mlp.fc1.weight.detach().cpu().numpy(),
        fc1_bias=block.mlp.fc1.bias.detach().cpu().numpy(),
        fc1_output=captured["fc1_output"],

        gelu_input=captured["gelu_input"],
        gelu_output=captured["gelu_output"],

        fc2_input=captured["fc2_input"],
        fc2_weight=block.mlp.fc2.weight.detach().cpu().numpy(),
        fc2_bias=block.mlp.fc2.bias.detach().cpu().numpy(),
        fc2_output=captured["fc2_output"],

        mlp_output=captured["mlp_output"],
    )

    print("Saved:", OUTPUT)

    print()
    print("True Block0 MLP shapes")
    print("--------------------------------")
    for k in [
        "norm2_input",
        "norm2_output",
        "fc1_input",
        "fc1_output",
        "gelu_output",
        "fc2_input",
        "fc2_output",
        "mlp_output",
    ]:
        print(
            f"{k:14s}:",
            (
                captured[k].shape
                if k in captured
                else {
                    "norm2_input": captured["norm2_input"],
                    "norm2_output": captured["norm2_output"],
                }[k].shape
            )
        )

    print()
    print("Topology consistency")
    print("--------------------------------")
    print("norm2 -> fc1 :", float(e1))
    print("fc1 -> GELU  :", float(e2))
    print("GELU -> fc2  :", float(e3))
    print("fc2 -> MLP   :", float(e4))

    print()
    print("Comparison with old FC1 workload")
    print("--------------------------------")

    old_input = frozen["input"]
    if old_input.ndim == 3:
        old_input = old_input[0]

    print(
        "Old FC1 input vs true LN2 max error:",
        float(
            np.max(
                np.abs(
                    old_input
                    - captured["fc1_input"]
                )
            )
        ),
    )


if __name__ == "__main__":
    main()
