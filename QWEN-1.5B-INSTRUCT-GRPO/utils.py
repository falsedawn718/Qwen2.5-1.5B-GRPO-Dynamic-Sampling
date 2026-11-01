from decimal import Decimal, InvalidOperation
import re
def extract_answer(text):
    NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
    patterns = [
        rf"####\s*({NUMBER})",
        rf"\\boxed\{{\s*({NUMBER})\s*\}}",
        rf"(?:final answer is|answer is)\s*:?\s*\$?({NUMBER})",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                return Decimal(matches[-1].replace(",", ""))
            except InvalidOperation:
                return None

    return None

def correctness_reward(completions, ground_truth, **kwargs):
    rewards = []

    for completion, gold in zip(completions, ground_truth):
        if isinstance(completion, dict):
            text = completion["content"]
        elif isinstance(completion, list):
            text = completion[0]["content"]
        else:
            text = completion

        model_answer = extract_answer(text)
        correct_answer = Decimal(gold)

        reward = float(
            model_answer is not None
            and model_answer == correct_answer
        )
        rewards.append(reward)

    return rewards
def process_dataset(dataset):
    def process(example, idx):
        answer = extract_answer(example["answer"])

        if answer is None:
            raise ValueError(f"Invalid answer: {idx}")

        return {
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful math assistant. "
                        "Solve the problem step by step. "
                        "End your response with a single line "
                        "in the format: #### <number>"
                    )
                },
                {
                    "role": "user",
                    "content": example["question"]
                }
            ],
            "ground_truth": str(answer),
            "sample_id": idx
        }

    return dataset.map(
        process,
        with_indices=True,
        remove_columns=dataset.column_names
    )