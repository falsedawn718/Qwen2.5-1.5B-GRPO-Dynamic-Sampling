import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from trl import GRPOTrainer
from trl import GRPOConfig
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import argparse
import re
from decimal import Decimal, InvalidOperation

from utils import extract_answer, correctness_reward, process_dataset

# from peft import LoraConfig

# peft_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=[
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#     ],
#     lora_dropout=0.0,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
dataset = load_dataset("openai/gsm8k", "main")

split = dataset["train"].train_test_split(
    test_size=0.01,
    seed=42
)





train_dataset = process_dataset(split["train"])
val_dataset =  process_dataset(split["test"])
# config = GRPOConfig(
#     output_dir="./gemini/output/grpo",

#     learning_rate=1e-6,
#     max_steps=16,

#     # Batch
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     num_generations=4,

#     max_completion_length=256,

#     # Memory
#     bf16=True,
#     gradient_checkpointing=True,
#     gradient_checkpointing_kwargs={
#         "use_reentrant": False
#     },

#     # Generation
#     generation_batch_size=4,
#     use_vllm=False,

#     # Evaluation
#     eval_strategy="no",

#     # Logging
#     logging_steps=1,
#     report_to="none",
# )
config = GRPOConfig(
    output_dir="/gemini/output/grpo",
    learning_rate=1e-6,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    max_steps=-1,
    num_train_epochs=2,
    num_generations=8,
    max_completion_length=1024,
    bf16=True,
    eval_strategy="steps",
    logging_steps=1,
    eval_steps=50,
    per_device_eval_batch_size=8,
    use_vllm=True
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=correctness_reward,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer
    # peft_config=peft_config
)


result = trainer.train()

print(result.global_step)
print(result.training_loss)
print(result.metrics)
metrics = trainer.evaluate()
print(metrics)
trainer.save_model("/gemini/output/grpo/QWEN-1.5B-INSTRUCT-GRPO/model")
tokenizer.save_pretrained("/gemini/output/grpo/QWEN-1.5B-INSTRUCT-GRPO/model")
# print(extract_answer("The final answer is: $123.45")==extract_answer("#### 123.4500"))
# print(process_dataset(dataset["train"].select(range(2))))
# print(correctness_reward([{"role":"assisstant","content": "The final answer is: $123.45"}], ["123.4500"]))
