import json
import csv
import math
from pathlib import Path

# ==========================================
# 1. Configuration
# ==========================================

FILES = {
    "Base": "/gemini/code/output/base.jsonl",
    "GRPO_1Epoch": "/gemini/code/output/grpo_1epoch.jsonl",
    "GRPO_2Epoch": "/gemini/code/output/grpo_2epoch.jsonl",
    "Dynamic_1Epoch": "/gemini/code/output/dynamic_grpo_1epoch.jsonl",
    "Dynamic_2Epoch": "/gemini/code/output/dynamic_grpo_2epoch.jsonl",
    "Dynamic_beta0_1Epoch": "/gemini/code/output/grpo_beta0_1epoch.jsonl",
    "Dynamic_beta0_2Epoch": "/gemini/code/output/grpo_beta0_2epoch.jsonl",
}

OUTPUT_DIR = Path("/gemini/code/output/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Pairwise comparisons
# ==========================================

# Accuracy_Diff_pp = Model_B accuracy - Model_A accuracy, in percentage points.
# Positive values mean Model_B is more accurate.
PAIRS = [
    # Original comparisons
    ("Base", "GRPO_1Epoch"),
    ("Base", "Dynamic_1Epoch"),
    ("Base", "Dynamic_2Epoch"),
    ("GRPO_1Epoch", "GRPO_2Epoch"),
    ("GRPO_1Epoch", "Dynamic_1Epoch"),
    ("GRPO_2Epoch", "Dynamic_2Epoch"),
    ("Dynamic_1Epoch", "Dynamic_2Epoch"),

    # Beta=0 comparisons
    ("Base", "Dynamic_beta0_1Epoch"),
    ("Base", "Dynamic_beta0_2Epoch"),
    ("GRPO_1Epoch", "Dynamic_beta0_1Epoch"),
    ("GRPO_2Epoch", "Dynamic_beta0_2Epoch"),
    ("Dynamic_1Epoch", "Dynamic_beta0_1Epoch"),
    ("Dynamic_2Epoch", "Dynamic_beta0_2Epoch"),
    ("Dynamic_beta0_1Epoch", "Dynamic_beta0_2Epoch"),
]

# Dedicated Beta=0 analysis.
# The first two are the key ablation comparisons:
# Dynamic(beta>0) vs Dynamic(beta=0), at the same epoch.
BETA0_ANALYSIS_PAIRS = [
    (
        "Dynamic_1Epoch",
        "Dynamic_beta0_1Epoch",
        "核心消融：1 Epoch 下，移除 counts/frequency penalty 的影响",
    ),
    (
        "Dynamic_2Epoch",
        "Dynamic_beta0_2Epoch",
        "核心消融：2 Epoch 下，移除 counts/frequency penalty 的影响",
    ),
    (
        "GRPO_1Epoch",
        "Dynamic_beta0_1Epoch",
        "1 Epoch 下，difficulty-based sampling（beta=0）相对普通 GRPO 的效果",
    ),
    (
        "GRPO_2Epoch",
        "Dynamic_beta0_2Epoch",
        "2 Epoch 下，difficulty-based sampling（beta=0）相对普通 GRPO 的效果",
    ),
    (
        "Dynamic_beta0_1Epoch",
        "Dynamic_beta0_2Epoch",
        "Beta=0 方案内部：训练轮数从 1 增加到 2 的效果",
    ),
    (
        "Base",
        "Dynamic_beta0_1Epoch",
        "Beta=0、1 Epoch 相对未训练 Base 的整体变化",
    ),
    (
        "Base",
        "Dynamic_beta0_2Epoch",
        "Beta=0、2 Epoch 相对未训练 Base 的整体变化",
    ),
]


# ==========================================
# 3. Load JSONL
# ==========================================

def load_jsonl(path):
    records = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            item = json.loads(line)

            if "idx" not in item:
                raise ValueError(f"{path}, line {line_num}: missing idx")

            idx = item["idx"]

            if idx in records:
                raise ValueError(f"{path}, line {line_num}: duplicate idx={idx}")

            if "is_correct" not in item:
                raise ValueError(
                    f"{path}, line {line_num}: missing is_correct"
                )

            if "extraction_failure" not in item:
                item["extraction_failure"] = (
                    item.get("prediction_answer") is None
                )

            item["is_correct"] = bool(item["is_correct"])
            item["extraction_failure"] = bool(item["extraction_failure"])

            records[idx] = item

    print(f"Loaded {path}: {len(records)} samples")
    return records


# ==========================================
# 4. Wilson 95% Confidence Interval
# ==========================================

def wilson_interval(correct, total, z=1.96):
    if total == 0:
        return None, None

    p = correct / total
    denominator = 1 + z**2 / total

    center = (
        p + z**2 / (2 * total)
    ) / denominator

    margin = (
        z * math.sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        ) / denominator
    )

    return center - margin, center + margin


# ==========================================
# 5. Model Statistics
# ==========================================

def analyze_model(name, records):
    total = len(records)

    correct = sum(
        r["is_correct"] for r in records.values()
    )

    failures = sum(
        r["extraction_failure"] for r in records.values()
    )

    valid = total - failures
    accuracy = correct / total if total else 0

    conditional_accuracy = (
        correct / valid if valid else None
    )

    failure_rate = failures / total if total else 0
    ci_low, ci_high = wilson_interval(correct, total)

    return {
        "Model": name,
        "Total": total,
        "Correct": correct,
        "Accuracy": accuracy,
        "CI95_Lower": ci_low,
        "CI95_Upper": ci_high,
        "Extraction_Failures": failures,
        "Failure_Rate": failure_rate,
        "Conditional_Accuracy": conditional_accuracy,
    }


# ==========================================
# 6. McNemar Exact Test
# ==========================================

def mcnemar_exact(a_only, b_only):
    """
    Two-sided exact McNemar test.

    a_only: A correct, B wrong
    b_only: A wrong, B correct
    """
    n = a_only + b_only

    if n == 0:
        return 1.0

    k = min(a_only, b_only)

    probability = sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2 ** n)

    return min(1.0, 2 * probability)


def holm_adjust(p_values):
    """
    Holm-Bonferroni adjusted p-values.
    Returns adjusted values in the original input order.
    """
    if not p_values:
        return []

    m = len(p_values)
    ordered = sorted(
        enumerate(p_values),
        key=lambda x: x[1]
    )

    adjusted = [None] * m
    running_max = 0.0

    for rank, (original_idx, p_value) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[original_idx] = running_max

    return adjusted


# ==========================================
# 7. Paired Model Comparison
# ==========================================

def compare_models(name_a, name_b, data_a, data_b):
    ids_a = set(data_a)
    ids_b = set(data_b)

    if ids_a != ids_b:
        missing_in_b = ids_a - ids_b
        missing_in_a = ids_b - ids_a
        raise ValueError(
            f"Dataset mismatch: {name_a} vs {name_b}; "
            f"missing in B={len(missing_in_b)}, "
            f"missing in A={len(missing_in_a)}"
        )

    both_correct = 0
    both_wrong = 0
    a_only = 0
    b_only = 0

    # Extraction-failure transitions
    a_fail_b_success = 0
    b_fail_a_success = 0

    disagreements = []

    for idx in sorted(ids_a):
        a = data_a[idx]
        b = data_b[idx]

        if a.get("gold_answer") != b.get("gold_answer"):
            raise ValueError(
                f"Gold answer mismatch at idx={idx}: "
                f"{name_a} vs {name_b}"
            )

        ca = a["is_correct"]
        cb = b["is_correct"]

        if ca and cb:
            both_correct += 1
        elif not ca and not cb:
            both_wrong += 1
        elif ca and not cb:
            a_only += 1
        else:
            b_only += 1

        fa = a["extraction_failure"]
        fb = b["extraction_failure"]

        if fa and not fb:
            a_fail_b_success += 1

        if fb and not fa:
            b_fail_a_success += 1

        if ca != cb:
            disagreements.append({
                "comparison": f"{name_a}_vs_{name_b}",
                "idx": idx,
                "question": a.get("question"),
                "gold_answer": a.get("gold_answer"),
                f"{name_a}_prediction": a.get("prediction_answer"),
                f"{name_a}_correct": ca,
                f"{name_b}_prediction": b.get("prediction_answer"),
                f"{name_b}_correct": cb,
            })

    p_value = mcnemar_exact(a_only, b_only)

    total = len(data_a)
    correct_a = sum(r["is_correct"] for r in data_a.values())
    correct_b = sum(r["is_correct"] for r in data_b.values())

    failures_a = sum(
        r["extraction_failure"] for r in data_a.values()
    )
    failures_b = sum(
        r["extraction_failure"] for r in data_b.values()
    )

    acc_a = correct_a / total if total else 0
    acc_b = correct_b / total if total else 0

    result = {
        "Model_A": name_a,
        "Model_B": name_b,
        "Total": total,
        "Accuracy_A": acc_a,
        "Accuracy_B": acc_b,
        "Accuracy_Diff_pp": (acc_b - acc_a) * 100,
        "Both_Correct": both_correct,
        "Both_Wrong": both_wrong,
        "A_Only_Correct": a_only,
        "B_Only_Correct": b_only,
        "Net_Improvement": b_only - a_only,
        "McNemar_P": p_value,
        "Extraction_Failures_A": failures_a,
        "Extraction_Failures_B": failures_b,
        "Failure_Rate_A": failures_a / total if total else 0,
        "Failure_Rate_B": failures_b / total if total else 0,
        "Failure_Diff_B_minus_A": failures_b - failures_a,
        "A_Fail_B_Success": a_fail_b_success,
        "B_Fail_A_Success": b_fail_a_success,
    }

    return result, disagreements


# ==========================================
# 8. CSV / JSONL Writers
# ==========================================

def write_csv(path, rows):
    if not rows:
        print(f"Skip empty CSV: {path}")
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ==========================================
# 9. Main
# ==========================================

def main():
    all_data = {}

    for name, path in FILES.items():
        all_data[name] = load_jsonl(path)

    # Validate that every model uses exactly the same evaluation examples.
    reference_name = next(iter(all_data))
    reference_ids = set(all_data[reference_name])

    for name, data in all_data.items():
        if set(data) != reference_ids:
            raise ValueError(
                f"Dataset idx mismatch: {reference_name} vs {name}"
            )

    # --------------------------------------
    # 9.1 Basic statistics
    # --------------------------------------

    summary = []

    print("\n========== MODEL SUMMARY ==========\n")

    for name, records in all_data.items():
        stats = analyze_model(name, records)
        summary.append(stats)

        print(f"\nModel: {name}")
        print(f"Accuracy: {stats['Accuracy']:.4%}")
        print(
            f"95% CI: "
            f"[{stats['CI95_Lower']:.4%}, "
            f"{stats['CI95_Upper']:.4%}]"
        )
        print(
            f"Correct: "
            f"{stats['Correct']}/{stats['Total']}"
        )
        print(
            f"Extraction Failures: "
            f"{stats['Extraction_Failures']}"
        )
        print(
            f"Failure Rate: "
            f"{stats['Failure_Rate']:.4%}"
        )

        if stats["Conditional_Accuracy"] is not None:
            print(
                f"Conditional Accuracy: "
                f"{stats['Conditional_Accuracy']:.4%}"
            )

    write_csv(
        OUTPUT_DIR / "model_summary.csv",
        summary
    )

    # --------------------------------------
    # 9.2 Pairwise comparisons
    # --------------------------------------

    pairwise_results = []
    all_disagreements = []
    results_by_pair = {}

    print("\n========== PAIRED COMPARISON ==========\n")

    for name_a, name_b in PAIRS:
        result, disagreements = compare_models(
            name_a,
            name_b,
            all_data[name_a],
            all_data[name_b],
        )

        pairwise_results.append(result)
        all_disagreements.extend(disagreements)
        results_by_pair[(name_a, name_b)] = result

        print(f"\n{name_a} vs {name_b}")
        print(
            f"Accuracy Diff (B - A): "
            f"{result['Accuracy_Diff_pp']:+.3f} pp"
        )
        print(
            f"A Only Correct: "
            f"{result['A_Only_Correct']}"
        )
        print(
            f"B Only Correct: "
            f"{result['B_Only_Correct']}"
        )
        print(
            f"Net Improvement (B - A): "
            f"{result['Net_Improvement']:+d}"
        )
        print(
            f"McNemar P-value: "
            f"{result['McNemar_P']:.6f}"
        )
        print(
            f"Extraction Failures (A -> B): "
            f"{result['Extraction_Failures_A']} -> "
            f"{result['Extraction_Failures_B']}"
        )
        print(
            f"Failure -> Success: "
            f"{result['A_Fail_B_Success']}"
        )

    write_csv(
        OUTPUT_DIR / "pairwise_comparison.csv",
        pairwise_results
    )

    write_jsonl(
        OUTPUT_DIR / "disagreements.jsonl",
        all_disagreements
    )

    # --------------------------------------
    # 9.3 Dedicated Beta=0 ablation analysis
    # --------------------------------------

    print("\n========== BETA=0 ABLATION ANALYSIS ==========\n")

    beta0_raw_rows = []
    beta0_disagreements = []

    for name_a, name_b, purpose in BETA0_ANALYSIS_PAIRS:
        result = results_by_pair[(name_a, name_b)]

        row = {
            "Comparison": f"{name_a}_vs_{name_b}",
            "Purpose": purpose,
            "Model_A": name_a,
            "Model_B": name_b,
            "Accuracy_A": result["Accuracy_A"],
            "Accuracy_B": result["Accuracy_B"],
            "Accuracy_Diff_pp_B_minus_A": result["Accuracy_Diff_pp"],
            "A_Only_Correct": result["A_Only_Correct"],
            "B_Only_Correct": result["B_Only_Correct"],
            "Net_Improvement_B_minus_A": result["Net_Improvement"],
            "McNemar_P": result["McNemar_P"],
            "Extraction_Failures_A": result["Extraction_Failures_A"],
            "Extraction_Failures_B": result["Extraction_Failures_B"],
            "Failure_Diff_B_minus_A": result["Failure_Diff_B_minus_A"],
            "A_Fail_B_Success": result["A_Fail_B_Success"],
            "B_Fail_A_Success": result["B_Fail_A_Success"],
        }
        beta0_raw_rows.append(row)

    # Holm correction across the seven pre-specified Beta=0 analysis contrasts.
    adjusted_p_values = holm_adjust([
        row["McNemar_P"] for row in beta0_raw_rows
    ])

    for row, adjusted_p in zip(beta0_raw_rows, adjusted_p_values):
        row["McNemar_P_Holm"] = adjusted_p

    # Keep only disagreements belonging to the dedicated Beta=0 contrasts.
    beta0_comparison_names = {
        f"{name_a}_vs_{name_b}"
        for name_a, name_b, _ in BETA0_ANALYSIS_PAIRS
    }

    beta0_disagreements = [
        item for item in all_disagreements
        if item["comparison"] in beta0_comparison_names
    ]

    write_csv(
        OUTPUT_DIR / "beta0_ablation_summary.csv",
        beta0_raw_rows
    )

    write_jsonl(
        OUTPUT_DIR / "beta0_ablation_disagreements.jsonl",
        beta0_disagreements
    )

    for row in beta0_raw_rows:
        print(f"\n{row['Comparison']}")
        print(f"Purpose: {row['Purpose']}")
        print(
            f"Accuracy Diff (B - A): "
            f"{row['Accuracy_Diff_pp_B_minus_A']:+.3f} pp"
        )
        print(
            f"Net Improvement (B - A): "
            f"{row['Net_Improvement_B_minus_A']:+d}"
        )
        print(
            f"McNemar P: {row['McNemar_P']:.6f}; "
            f"Holm-adjusted P: {row['McNemar_P_Holm']:.6f}"
        )
        print(
            f"Extraction Failures (A -> B): "
            f"{row['Extraction_Failures_A']} -> "
            f"{row['Extraction_Failures_B']}"
        )

    # --------------------------------------
    # 9.4 Interpretation reminder
    # --------------------------------------

    print("\n========== INTERPRETATION NOTE ==========")
    print(
        "Beta=0 does NOT disable difficulty-based sampling. "
        "For w_i = (difficulty_i + 1e-6) / "
        "(1 + counts_i / tau)^beta, setting beta=0 gives "
        "w_i = difficulty_i + 1e-6. "
        "It removes only the counts/frequency penalty."
    )
    print(
        "Therefore, Dynamic(beta>0) vs Dynamic(beta=0) at the "
        "same epoch isolates the incremental effect of the "
        "repeat/frequency penalty, assuming other training settings "
        "and random seeds are controlled."
    )
    print(
        "McNemar P is an exact paired test. Holm-adjusted P in "
        "beta0_ablation_summary.csv corrects across the seven "
        "Beta=0 analysis contrasts."
    )

    print("\n========== ANALYSIS FINISHED ==========")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - model_summary.csv")
    print("  - pairwise_comparison.csv")
    print("  - disagreements.jsonl")
    print("  - beta0_ablation_summary.csv")
    print("  - beta0_ablation_disagreements.jsonl")


if __name__ == "__main__":
    main()
