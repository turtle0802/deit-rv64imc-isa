from pathlib import Path
from collections import Counter
import re

DOT8_OPS = 1_000_000
MACS = DOT8_OPS * 8

CONFIGS = [
    (
        "Scalar RV64IMC",
        "results/dot8_scalar_pc_hist.txt",
        "results/dot8_scalar_disasm.txt",
    ),
    (
        "Xqmac8",
        "results/dot8_xqmac8_pc_hist.txt",
        "results/dot8_xqmac8_disasm.txt",
    ),
]


def parse_hist(path):
    counts = {}

    for line in Path(path).read_text().splitlines():
        parts = line.split()

        if len(parts) != 2:
            continue

        try:
            pc = int(parts[0], 16)
            count = int(parts[1])
        except ValueError:
            continue

        counts[pc] = count

    return counts


def parse_disasm(path):
    lines = Path(path).read_text().splitlines()

    symbols = []
    pc_to_mnemonic = {}

    symbol_re = re.compile(
        r"^\s*([0-9a-fA-F]+)\s+<([^>]+)>:$"
    )

    insn_re = re.compile(
        r"^\s*([0-9a-fA-F]+):\s+"
        r"[0-9a-fA-F]+\s+"
        r"([.\w]+)"
    )

    for line in lines:
        m = symbol_re.match(line)

        if m:
            symbols.append(
                (int(m.group(1), 16), m.group(2))
            )

        m = insn_re.match(line)

        if m:
            pc = int(m.group(1), 16)
            mnemonic = m.group(2)

            pc_to_mnemonic[pc] = mnemonic

    main_start = None
    main_end = None

    for i, (addr, name) in enumerate(symbols):
        if name == "main":
            main_start = addr

            if i + 1 < len(symbols):
                main_end = symbols[i + 1][0]
            else:
                main_end = 1 << 64

            break

    if main_start is None:
        raise RuntimeError(
            f"Cannot find main() in {path}"
        )

    return (
        main_start,
        main_end,
        pc_to_mnemonic,
    )


def classify(mnemonic):
    if mnemonic == ".insn":
        return "Custom"

    if mnemonic in {
        "lb", "lbu",
        "lh", "lhu",
        "lw", "lwu",
        "ld",
    }:
        return "Load"

    if mnemonic in {
        "sb", "sh", "sw", "sd",
    }:
        return "Store"

    if mnemonic in {
        "mul", "mulw",
        "mulh", "mulhu", "mulhsu",
    }:
        return "Multiply"

    if (
        mnemonic.startswith("b")
        or mnemonic in {
            "j", "jal", "jalr", "ret",
        }
    ):
        return "Branch/Jump"

    return "Integer ALU/Other"


def profile(
    name,
    hist_path,
    disasm_path,
):
    hist = parse_hist(hist_path)

    (
        main_start,
        main_end,
        pc_to_mnemonic,
    ) = parse_disasm(disasm_path)

    mnemonic_counts = Counter()
    category_counts = Counter()

    main_instructions = 0

    for pc, count in hist.items():
        if not (
            main_start <= pc < main_end
        ):
            continue

        main_instructions += count

        mnemonic = pc_to_mnemonic.get(
            pc,
            "unknown",
        )

        mnemonic_counts[mnemonic] += count

        category_counts[
            classify(mnemonic)
        ] += count

    print("=" * 52)
    print(name)
    print("=" * 52)

    print(
        f"dot8 operations       : "
        f"{DOT8_OPS:,}"
    )

    print(
        f"MAC operations        : "
        f"{MACS:,}"
    )

    print(
        f"main instructions     : "
        f"{main_instructions:,}"
    )

    print(
        f"Instructions / dot8   : "
        f"{main_instructions / DOT8_OPS:.6f}"
    )

    print(
        f"Instructions / MAC    : "
        f"{main_instructions / MACS:.6f}"
    )

    print(
        f"MAC / instruction     : "
        f"{MACS / main_instructions:.6f}"
    )

    loads = category_counts["Load"]

    print(
        f"Loads / MAC           : "
        f"{loads / MACS:.6f}"
    )

    print(
        f"Custom instructions   : "
        f"{category_counts['Custom']:,}"
    )

    print()

    print("Instruction mix")
    print("-" * 52)

    for category in [
        "Load",
        "Store",
        "Multiply",
        "Custom",
        "Branch/Jump",
        "Integer ALU/Other",
    ]:
        value = category_counts[category]

        percentage = (
            100.0
            * value
            / main_instructions
        )

        print(
            f"{category:18s}: "
            f"{value:12,d} "
            f"({percentage:6.2f}%)"
        )

    print()

    print("Dynamic mnemonic counts")
    print("-" * 52)

    for mnemonic, count in (
        mnemonic_counts.most_common()
    ):
        print(
            f"{mnemonic:12s} "
            f"{count:12,d}"
        )

    print()

    return main_instructions


results = {}

for (
    name,
    hist,
    disasm,
) in CONFIGS:
    results[name] = profile(
        name,
        hist,
        disasm,
    )


scalar = results["Scalar RV64IMC"]
xqmac = results["Xqmac8"]

print("=" * 52)
print("Scalar vs Xqmac8")
print("=" * 52)

print(
    f"Instruction reduction : "
    f"{scalar / xqmac:.4f}x"
)

print(
    f"Instruction decrease  : "
    f"{(1.0 - xqmac / scalar) * 100:.2f}%"
)
