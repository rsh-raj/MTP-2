import argparse
import torch
from datasets import load_dataset
from huggingface_hub import login
from unsloth import FastLanguageModel
import re
from trl import SFTTrainer
from transformers import TrainingArguments

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="Model name from HuggingFace or Unsloth")
parser.add_argument("--output_file", type=str, required=True, help="Output file name to store predictions")
args = parser.parse_args()

# Login to HF
login(token="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")

MAX_CONTEXT_SIZE = 1024
dtype = None
load_in_4bit = True

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model_name,
    max_seq_length=MAX_CONTEXT_SIZE,
    dtype=dtype,
    load_in_4bit=False,
    load_in_8bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
You are a commit classifier based on commit message and code diff.
    Please classify the given commit into one of the ten categories: docs, perf, style, refactor, feat, fix, test, ci, build, and chore. The definitions of each category are as follows:\n
    **feat**: Code changes aim to introduce new features to the codebase, encompassing both internal and user-oriented features.\n
    **fix**: Code changes aim to fix bugs and faults within the codebase.\n
    **perf**: Code changes aim to improve performance, such as enhancing execution speed or reducing memory consumption.\n
    **style**: Code changes aim to improve readability without affecting the meaning of the code. This type encompasses aspects like variable naming, indentation, and addressing linting or code analysis warnings.\n
    **refactor**: Code changes aim to restructure the program without changing its behavior, aiming to improve maintainability. To avoid confusion and overlap, we propose the constraint that this category does not include changes classified as ``perf'' or ``style''. Examples include enhancing modularity, refining exception handling, improving scalability, conducting code cleanup, and removing deprecated code.\n
    **docs**: Code changes that modify documentation or text, such as correcting typos, modifying comments, or updating documentation.\n
    **test**: Code changes that modify test files, including the addition or updating of tests.\n
    **ci**: Code changes to CI (Continuous Integration) configuration files and scripts, such as configuring or updating CI/CD scripts, e.g., ``.travis.yml'' and ``.github/workflows''.\n
    **build**: Code changes affecting the build system (e.g., Maven, Gradle, Cargo). Change examples include updating dependencies, configuring build configurations, and adding scripts.\n
    **chore**: Code changes for other miscellaneous tasks that do not neatly fit into any of the above categories.\n

### Input:
- given commit message:\n
{}
- given commit diff\n
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def format_prompt(samples):
    commit_msgs = samples["masked_commit_message"]
    git_diffs = samples["summarised_git_diff"]
    outputs = samples["annotated_type"]
    texts = []
    for commit_msg, git_diff, output in zip(commit_msgs, git_diffs, outputs):
        git_diff = git_diff or ""
        available_length = MAX_CONTEXT_SIZE - len(tokenizer.encode(prompt.format(commit_msg, "", output)))
        git_diff = git_diff[:available_length]
        text = prompt.format(commit_msg, git_diff, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Dataset loading and formatting
train_dataset = load_dataset("rsh-raj/ccs-dataset-diff-summarised", split="train")
eval_dataset = load_dataset("rsh-raj/eval-summarised", split="test")

dataset = train_dataset.map(format_prompt, batched=True)
eval_dataset = eval_dataset.map(format_prompt, batched=True)

# Trainer setup
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_CONTEXT_SIZE,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=10,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        num_train_epochs=5,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=".",
    ),
)

trainer.train()

# Inference mode
FastLanguageModel.for_inference(model)

result = []
test_dataset = load_dataset("rsh-raj/ccs-summarised-git-diff-test", split="test")

with open(args.output_file, "w") as f:
    for sample in test_dataset:
        commit_msg = sample["masked_commit_message"]
        git_diff = sample["summarised_git_diff"] or ""
        available_length = MAX_CONTEXT_SIZE - len(tokenizer.encode(prompt.format(commit_msg, "", "")))
        git_diff = git_diff[:available_length]

        input_text = prompt.format(commit_msg, git_diff, "")
        inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=32, use_cache=True)
        decoded_output = tokenizer.batch_decode(outputs)[0]

        match = re.search(r'Response:\s*(\w+)', decoded_output)
        if match:
            extracted_word = match.group(1)
            result.append(extracted_word)
            f.write(f'Extracted word: {extracted_word}\n')
        else:
            f.write(f'No match found, response: {decoded_output}\n')
            result.append("")

        del inputs, outputs, git_diff

# Accuracy calculation
ans = [sample["annotated_type"] for sample in test_dataset]
accuracy = sum(1 for x, y in zip(ans, result) if x == y) / len(test_dataset)
print(result)
print(f"Accuracy is {accuracy}")
