import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from trl import GRPOTrainer
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from utils import extract_answer, correctness_reward, process_dataset
import json
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
dataset = load_dataset("openai/gsm8k", "main")
test_dataset=dataset["test"]

model = AutoModelForCausalLM.from_pretrained(
    "/gemini/pretrain/grpo_beta0/checkpoint-3698",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)
model.eval()

tokenizer.padding_side = "left"


correct = 0
extraction_failures = 0
total = len(test_dataset)
batch_size = 128

# 每次实验使用不同的文件名，避免覆盖之前的结果
results_path = "/gemini/output/grpo_test_predictions.jsonl"

with open(results_path, "w", encoding="utf-8") as fout:
    for start in range(0, total, batch_size):
        batch = test_dataset[start:start + batch_size]

        messages = [
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful math assistant. "
                        "Solve the problem step by step. "
                        "End your response with a single line "
                        "in the format: #### <number>"
                    )
                },
                {"role": "user", "content": question}
            ]
            for question in batch["question"]
        ]

        prompts = [
            tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True
            )
            for msg in messages
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        # 仅解码新生成的回答，不包含输入 Prompt
        generated = outputs[:, inputs["input_ids"].shape[1]:]

        predictions = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )

        for offset, (question, gold, pred) in enumerate(
            zip(batch["question"], batch["answer"], predictions)
        ):
            pred_answer = extract_answer(pred)
            gold_answer = extract_answer(gold)

            # 提取失败单独统计
            extraction_failure = pred_answer is None
            if extraction_failure:
                extraction_failures += 1

            is_correct = (
                pred_answer is not None
                and gold_answer is not None
                and pred_answer == gold_answer
            )

            if is_correct:
                correct += 1

            # 保存每一道题的完整评测记录
            record = {
                "idx": start + offset,
                "question": question,
                "gold_raw": gold,
                "gold_answer": str(gold_answer) if gold_answer is not None else None,
                "prediction_raw": pred,
                "prediction_answer": str(pred_answer) if pred_answer is not None else None,
                "is_correct": is_correct,
                "extraction_failure": extraction_failure
            }

            fout.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

        # 每个 batch 保存一次，降低中途退出时的数据丢失风险
        fout.flush()

        print(f"Evaluated: {min(start + batch_size, total)}/{total}")

print(f"Test Accuracy: {correct / total:.4f}")
print(f"Correct: {correct}/{total}")
print(f"Extraction failures: {extraction_failures}")
print(f"Results saved to: {os.path.abspath(results_path)}")
