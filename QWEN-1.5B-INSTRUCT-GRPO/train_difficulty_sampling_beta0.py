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
from trl.trainer.utils import RepeatSampler
from collections import defaultdict
from utils import extract_answer, correctness_reward, process_dataset
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
dataset = load_dataset("openai/gsm8k", "main")

split = dataset["train"].train_test_split(
    test_size=0.01,
    seed=42
)


train_dataset = process_dataset(split["train"])
val_dataset =  process_dataset(split["test"])
train_dataset=train_dataset.add_column(
    "sample_ids",
    list(range(len(train_dataset))))



class DifficultyTracker:
    def __init__(self, num_samples, num_generations=8, momentum=0.5):
        self.accuracy = torch.full(
            (num_samples,), 0.5, dtype=torch.float64
        )
        self.counts = torch.zeros(
            num_samples, dtype=torch.long
        )
        self.num_generations = num_generations
        self.momentum = momentum

    def update(self, sample_ids, rewards):
        grouped = defaultdict(list)

        for idx, reward in zip(sample_ids, rewards):
            idx = int(idx)
            if idx >= 0:
                grouped[idx].append(float(reward))

        for idx, values in grouped.items():
            batch_accuracy=sum(values)/len(values)
            self.accuracy[idx] = (
                (1 - self.momentum) * self.accuracy[idx]
                + self.momentum * batch_accuracy
            )
            self.counts[idx]+=1

    def get_weights(self, tau=4, beta=0):
        p = self.accuracy
        G = self.num_generations

        difficulty = 1 - p.pow(G) - (1 - p).pow(G)

        penalty = (
            1 + self.counts.double() / tau
        ).pow(beta)

        return (difficulty + 1e-6) / penalty 


class WeightedRepeatSampler(RepeatSampler):
    def __init__(self, data_source, mini_repeat_count, batch_size = 1, repeat_count = 1, shuffle = True, seed = 42,tracker=None):
        super().__init__(data_source, mini_repeat_count, batch_size, repeat_count, shuffle, seed)
        self.num_samples=len(data_source)
        if tracker is None:
            raise ValueError("DifficultyTracker is required")
        self.tracker=tracker
    def __iter__(self):
        num_chunks = self.num_samples // self.batch_size
        for _ in range(num_chunks):
            weights=self.tracker.get_weights()
            indexes=torch.multinomial(
                weights,
                num_samples=self.batch_size,
                replacement=False,
                generator=self.generator#torch.Generator() 
            )

            # 保持 RepeatSampler 的分组重复结构
            for _ in range(self.repeat_count):
                for idx in indexes:
                    for _ in range(self.mini_repeat_count):
                        yield idx
                        

tracker = DifficultyTracker(
    num_samples=len(train_dataset),
    num_generations=8,
    momentum=0.5
)
def tracked_reward(
    completions,
    ground_truth,
    sample_ids=None,
    **kwargs):
    rewards=correctness_reward(
        completions=completions,
        ground_truth=ground_truth,
        **kwargs
    )
    if sample_ids is not None:
        tracker.update(sample_ids, rewards)
    return rewards
    
class WeightedGRPOTrainer(GRPOTrainer):
    def __init__(self,*args, tracker=None, **kwargs):
        self.difficulty_tracker=tracker
        super().__init__(*args,**kwargs)
    def _get_train_sampler(self, dataset = None):
        if dataset is None:
            dataset = self.train_dataset
        return WeightedRepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=(
                self.args.generation_batch_size
                // self.num_generations
            ),
            repeat_count=(
                self.num_iterations
                * self.args.steps_per_generation
            ),
            tracker=self.difficulty_tracker
        )

config = GRPOConfig(
    output_dir="/gemini/output/grpo_beta0",
    learning_rate=1e-6,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    max_steps=-1,
    num_train_epochs=2,
    num_generations=8,
    max_completion_length=1024,
    bf16=True,
    eval_strategy="steps",
    save_strategy="epoch",
    save_total_limit=3,
    logging_steps=1,
    eval_steps=50,
    per_device_eval_batch_size=8,
    use_vllm=True
)
trainer = WeightedGRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=tracked_reward,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    tracker=tracker
)


result = trainer.train()

print(result.global_step)
print(result.training_loss)
print(result.metrics)
metrics = trainer.evaluate()
print(metrics)
trainer.save_model("/gemini/output/grpo/QWEN-1.5B-INSTRUCT-GRPO/model")
tokenizer.save_pretrained("/gemini/output/grpo/QWEN-1.5B-INSTRUCT-GRPO/model")