from pathlib import Path

import numpy as np
import torch
import timm


FC1_FILE = Path("results/fc1_fp32.npz")
FC2_FILE = Path("results/fc2_fp32.npz")


def main():
    # ------------------------------------------------------------
    # Load the same frozen FC1 input that was already used
    # for the existing FC1 experiment.
    # ------------------------------------------------------------
    fc1_npz = np.load(FC1_FILE)

    print("FC1 NPZ keys:", fc1_npz.files)

    if "input" not in fc1_npz.files:
        raise KeyError(
            "Expected key 'input' in results/fc1_fp32.npz"
        )

    fc1_input_np = fc1_npz["input"].astype(np.float32)

    # Accept either [197, 192] or [1, 197, 192].
    if fc1_input_np.ndim == 2:
        fc1_input_np = fc1_input_np[None, :, :]

    if fc1_input_np.shape != (1, 197, 192):
        raise ValueError(
            f"Unexpected FC1 input shape: {fc1_input_np.shape}"
        )

    # ------------------------------------------------------------
    # Recreate the same DeiT-Tiny model.
    # ------------------------------------------------------------
    torch.manual_seed(0)

    model = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=True,
    )

    model.eval()

    block0 = model.blocks[0]
    fc1 = block0.mlp.fc1
    fc2 = block0.mlp.fc2

    # ------------------------------------------------------------
    # Check that the currently loaded pretrained model is the
    # same one used when fc1_fp32.npz was generated.
    # ------------------------------------------------------------
    if "weight" in fc1_npz.files:
        saved_fc1_weight = fc1_npz["weight"]

        current_fc1_weight = (
            fc1.weight
            .detach()
            .cpu()
            .numpy()
        )

        max_weight_error = np.max(
            np.abs(
                saved_fc1_weight
                - current_fc1_weight
            )
        )

        print(
            "FC1 weight consistency max error:",
            max_weight_error,
        )

        if max_weight_error != 0:
            raise RuntimeError(
                "Loaded DeiT model does not match "
                "the frozen FC1 reference."
            )

    if "bias" in fc1_npz.files:
        saved_fc1_bias = fc1_npz["bias"]

        current_fc1_bias = (
            fc1.bias
            .detach()
            .cpu()
            .numpy()
        )

        max_bias_error = np.max(
            np.abs(
                saved_fc1_bias
                - current_fc1_bias
            )
        )

        print(
            "FC1 bias consistency max error:",
            max_bias_error,
        )

        if max_bias_error != 0:
            raise RuntimeError(
                "Loaded DeiT model does not match "
                "the frozen FC1 bias."
            )

    # ------------------------------------------------------------
    # Capture the actual input and output of FC2.
    #
    # MLP path:
    #
    # FC1 -> GELU -> Dropout/Norm -> FC2
    #
    # In eval mode dropout is disabled.
    # ------------------------------------------------------------
    captured = {}

    def fc2_hook(module, inputs, output):
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

    hook_handle = fc2.register_forward_hook(
        fc2_hook
    )

    x = torch.from_numpy(fc1_input_np)

    with torch.no_grad():
        mlp_output = block0.mlp(x)

    hook_handle.remove()

    if "input" not in captured:
        raise RuntimeError(
            "FC2 hook did not capture input."
        )

    # Remove batch dimension.
    fc2_input = captured["input"][0]
    fc2_output = captured["output"][0]

    fc2_weight = (
        fc2.weight
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    fc2_bias = (
        fc2.bias
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    # ------------------------------------------------------------
    # Shape checks.
    # ------------------------------------------------------------
    assert fc2_input.shape == (197, 768)
    assert fc2_weight.shape == (192, 768)
    assert fc2_bias.shape == (192,)
    assert fc2_output.shape == (197, 192)

    # ------------------------------------------------------------
    # Save frozen FP32 reference.
    # ------------------------------------------------------------
    FC2_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        FC2_FILE,
        input=fc2_input,
        weight=fc2_weight,
        bias=fc2_bias,
        output=fc2_output,
    )

    print()
    print("Saved:", FC2_FILE)

    print()
    print("FC2 shapes")
    print("------------------------------")
    print("Input :", fc2_input.shape)
    print("Weight:", fc2_weight.shape)
    print("Bias  :", fc2_bias.shape)
    print("Output:", fc2_output.shape)

    print()
    print("FC2 GEMM")
    print("------------------------------")
    print(
        "[197 x 768] x [768 x 192]"
        " -> [197 x 192]"
    )

    macs = 197 * 768 * 192

    print("MACs:", f"{macs:,}")

    print()
    print("FC2 input first 8:")
    print(fc2_input[0, :8])

    print()
    print("FC2 output first 8:")
    print(fc2_output[0, :8])

    # Sanity check that hook output is also the MLP output.
    mlp_output_np = (
        mlp_output
        .detach()
        .cpu()
        .numpy()[0]
    )

    max_output_error = np.max(
        np.abs(
            mlp_output_np
            - fc2_output
        )
    )

    print()
    print(
        "FC2 output vs MLP output max error:",
        max_output_error,
    )


if __name__ == "__main__":
    main()
