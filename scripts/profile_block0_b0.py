import re
from collections import Counter


HIST_FILE = "results/block0_b0_pc_hist.txt"
DISASM_FILE = "results/block0_b0_disasm.txt"

TOTAL_MACS = 102_049_152


def classify(mnemonic):
    loads = {
        "lb", "lbu",
        "lh", "lhu",
        "lw", "lwu",
        "ld",
    }

    stores = {
        "sb", "sh", "sw", "sd",
    }

    multiplies = {
        "mul", "mulw",
        "mulh", "mulhu", "mulhsu",
    }

    branches = {
        "beq", "bne",
        "blt", "bge",
        "bltu", "bgeu",

        # objdump pseudo instructions
        "beqz", "bnez",
        "bltz", "bgez",
        "bgtz", "blez",

        "j", "jr",
        "jal", "jalr",
        "ret",
        "call", "tail",
    }

    if mnemonic in loads:
        return "Load"

    if mnemonic in stores:
        return "Store"

    if mnemonic in multiplies:
        return "Multiply"

    if mnemonic in branches:
        return "Branch/Jump"

    return "Integer ALU/Other"


# ============================================================
# Parse objdump
# ============================================================

instruction_map = {}
current_symbol = None

symbol_pattern = re.compile(
    r"^([0-9a-fA-F]+) <([^>]+)>:$"
)

with open(DISASM_FILE, "r") as f:
    for raw_line in f:
        line = raw_line.strip()

        symbol_match = symbol_pattern.match(line)

        if symbol_match:
            current_symbol = symbol_match.group(2)
            continue

        parts = line.split()

        if len(parts) < 3:
            continue

        if not parts[0].endswith(":"):
            continue

        pc_string = parts[0][:-1]

        try:
            pc = int(pc_string, 16)
        except ValueError:
            continue

        encoding = parts[1]

        if not all(
            c in "0123456789abcdefABCDEF"
            for c in encoding
        ):
            continue

        mnemonic = parts[2]

        instruction_map[pc] = {
            "mnemonic": mnemonic,
            "symbol": current_symbol,
        }


# ============================================================
# Parse Spike PC histogram
# ============================================================

histogram = {}

with open(HIST_FILE, "r") as f:
    for line in f:
        parts = line.split()

        if len(parts) != 2:
            continue

        try:
            pc = int(parts[0], 16)
            count = int(parts[1])
        except ValueError:
            continue

        histogram[pc] = count


# ============================================================
# Aggregate
# ============================================================

program_total = 0
main_total = 0
boot_or_unknown_total = 0

category_count = Counter()
mnemonic_count = Counter()

unknown_pcs = []

for pc, count in histogram.items():

    if pc not in instruction_map:
        boot_or_unknown_total += count
        unknown_pcs.append(
            (pc, count)
        )
        continue

    info = instruction_map[pc]

    mnemonic = info["mnemonic"]
    symbol = info["symbol"]

    program_total += count

    if symbol != "main":
        continue

    main_total += count

    mnemonic_count[mnemonic] += count

    category = classify(mnemonic)

    category_count[category] += count


# ============================================================
# Print
# ============================================================

print()
print("========================================")
print("Block0 B0 Dynamic Instruction Profile")
print("========================================")

print(f"MACs                 : {TOTAL_MACS:,}")
print()

print(
    f"Program instructions : "
    f"{program_total:,}"
)

print(
    f"main instructions    : "
    f"{main_total:,}"
)

print(
    f"Boot/unknown         : "
    f"{boot_or_unknown_total:,}"
)

print()
print("main() instruction mix")
print("----------------------------------------")

categories = [
    "Load",
    "Store",
    "Multiply",
    "Branch/Jump",
    "Integer ALU/Other",
]

for category in categories:

    count = category_count[category]

    percent = (
        count / main_total * 100.0
        if main_total
        else 0.0
    )

    print(
        f"{category:18s}: "
        f"{count:12,d} "
        f"({percent:6.2f}%)"
    )


print()
print("Derived metrics")
print("----------------------------------------")

print(
    "Instructions / MAC   : "
    f"{main_total / TOTAL_MACS:.6f}"
)

print(
    "MAC / instruction    : "
    f"{TOTAL_MACS / main_total:.6f}"
)

print(
    "Loads / MAC          : "
    f"{category_count['Load'] / TOTAL_MACS:.6f}"
)

print(
    "Stores / MAC         : "
    f"{category_count['Store'] / TOTAL_MACS:.6f}"
)

print(
    "Multiply / MAC       : "
    f"{category_count['Multiply'] / TOTAL_MACS:.6f}"
)


print()
print("Dynamic mnemonic counts")
print("----------------------------------------")

for mnemonic, count in mnemonic_count.most_common():

    percent = count / main_total * 100.0

    print(
        f"{mnemonic:12s} "
        f"{count:12,d} "
        f"({percent:6.2f}%)"
    )


print()
print("Spike boot / unknown PCs")
print("----------------------------------------")

for pc, count in sorted(unknown_pcs):
    print(
        f"0x{pc:08x}: {count:,}"
    )
