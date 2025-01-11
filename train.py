CURR_DIR = "/kaggle/input/train-data/"
import torch
import datasets
import pandas
from sklearn.model_selection import train_test_split
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
from transformers import TrainerCallback
from contextlib import nullcontext
import pandas as pd
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
from pytorch_lightning import Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, AdamW
from argparse import ArgumentParser
from collections import Counter
from pytorch_lightning.callbacks import ModelCheckpoint

DATASET_PATH = "/kaggle/input/finaldata/final_data.csv"
LLAMA_MODEL = "0x404/ccs-code-llama-7b"
ENABLE_PROFILER = False
OUTPUT_DIR = "/kagagle/output"
df = pd.read_csv(DATASET_PATH)
print(df.head())
PROMPT_HEAD = (
    f"<s>[INST] <<SYS>>\n"
    f"You are a commit classifier based on commit message and code diff."
    f"Please classify the given commit into one of the ten categories: docs, perf, style, refactor, feat, fix, test, ci, build, and chore. The definitions of each category are as follows:\n"
    f"**feat**: Code changes aim to introduce new features to the codebase, encompassing both internal and user-oriented features.\n"
    f"**fix**: Code changes aim to fix bugs and faults within the codebase.\n"
    f"**perf**: Code changes aim to improve performance, such as enhancing execution speed or reducing memory consumption.\n"
    f"**style**: Code changes aim to improve readability without affecting the meaning of the code. This type encompasses aspects like variable naming, indentation, and addressing linting or code analysis warnings.\n"
    f"**refactor**: Code changes aim to restructure the program without changing its behavior, aiming to improve maintainability. To avoid confusion and overlap, we propose the constraint that this category does not include changes classified as ``perf'' or ``style''. Examples include enhancing modularity, refining exception handling, improving scalability, conducting code cleanup, and removing deprecated code.\n"
    f"**docs**: Code changes that modify documentation or text, such as correcting typos, modifying comments, or updating documentation.\n"
    f"**test**: Code changes that modify test files, including the addition or updating of tests.\n"
    f"**ci**: Code changes to CI (Continuous Integration) configuration files and scripts, such as configuring or updating CI/CD scripts, e.g., ``.travis.yml'' and ``.github/workflows''.\n"
    f"**build**: Code changes affecting the build system (e.g., Maven, Gradle, Cargo). Change examples include updating dependencies, configuring build configurations, and adding scripts.\n"
    f"**chore**: Code changes for other miscellaneous tasks that do not neatly fit into any of the above categories.\n"
    f"<</SYS>>\n\n"
)
PROMPT_COMMIT_MESSAGE = f"- given commit message:\n{{message}}\n"
PROMPT_COMMIT_DIFF = f"- given commit diff: \n{{diff}}\n"
TOKENIZER = LlamaTokenizer.from_pretrained(LLAMA_MODEL)
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.padding_side = "right"


def preprocess_dataset(dataset: datasets.Dataset):
    def apply_prompt_template(sample):
        return {
            "prompt_commit_message": PROMPT_COMMIT_MESSAGE.format(
                message=sample["masked_commit_message"]
            ),
            "prompt_commit_diff": PROMPT_COMMIT_DIFF.format(diff=sample["git_diff"]),
            "response": f"[/INST] {sample['annotated_type']} </s>",
        }

    def tokenize_add_label(sample):
        prompt_head = TOKENIZER.encode(
            PROMPT_HEAD,
            add_special_tokens=False,
        )
        message = TOKENIZER.encode(
            sample["prompt_commit_message"],
            max_length=64,
            truncation=True,
            add_special_tokens=False,
        )
        response = TOKENIZER.encode(
            sample["response"], max_length=20, truncation=True, add_special_tokens=False
        )
        diff = TOKENIZER.encode(
            sample["prompt_commit_diff"],
            max_length=1023 - len(prompt_head) - len(message) - len(response),
            truncation=True,
            add_special_tokens=False,
        )

        max_length = 1024 - len(prompt_head) - len(diff) - len(response) - len(message)

        if max_length > 0:
            pad = TOKENIZER.encode(
                TOKENIZER.eos_token,
                add_special_tokens=False,
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
        else:
            pad = []

        sample = {
            "input_ids": prompt_head + message + diff + response + pad,
            "attention_mask": [1] * 1024,
            "labels": [-100] * len(prompt_head + message + diff)
            + response
            + [-100] * len(pad),
        }

        return sample

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset


def make_dataset():
    df = pandas.read_csv(DATASET_PATH)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["type"], random_state=42
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=2 / 3, stratify=temp_df["type"], random_state=42
    )

    train_dataset = datasets.Dataset.from_pandas(train_df)

    train_dataset = preprocess_dataset(train_dataset)

    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return train_dataset
BATCH_SIZE = 16
N_EPOCHS = 10
LEARNING_RATE = 2e-5
DATASET_PATH = "/kaggle/working/final_data.csv"



def preprocess_dataset() -> [pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(DATASET_PATH)
    if "encoded_labels" not in dataset.columns:
        label_encoder = LabelEncoder()
        dataset["encoded_labels"] = label_encoder.fit_transform(
            dataset["type"]
        )
        dataset.to_csv(DATASET_PATH, index=False)
    train_df, temp_df = train_test_split(
        dataset, test_size=0.3, random_state=42
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=2 / 3, random_state=42
    )

    train_sha_set = set(train_df.sha)
    test_sha_set = set(test_df.sha)
    # assert len(train_sha_set.intersection(test_sha_set)) == 0
    # assert len(train_sha_set) == 1400
    # assert len(test_sha_set) == 400
    return train_df, valid_df, test_df


class CommitDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]

        commit_message = data_row.masked_commit_message
        git_diff = data_row.git_diff
        labels = data_row.encoded_labels

        encoding = self.tokenizer.encode_plus(
            commit_message + " <SEP> " + git_diff,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(labels, dtype=torch.long),
        )


class CommitClassifier(pl.LightningModule):
    def __init__(self, n_classes: int = 10, steps_per_epoch=None, n_epochs=None):
        super().__init__()
        self.bert = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.test_pred, self.test_label = [], []
        self.val_pred, self.val_label = [], []

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        return self.classifier(pooled_output)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1)
        self.val_pred.extend(predictions.cpu().numpy())
        self.val_label.extend(labels.cpu().numpy())

    def on_validation_epoch_end(self):
        acc = accuracy_score(self.val_label, self.val_pred)
        self.val_label = []
        self.val_pred = []
        self.log("val_acc", acc, sync_dist=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=LEARNING_RATE)

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask)

        loss = self.criterion(outputs, labels)
        self.log("test_loss", loss, on_step=True, on_epoch=True)

        predictions = torch.argmax(outputs, dim=1)
        correct_count = torch.sum(predictions == labels)
        accuracy = correct_count.float() / labels.shape[0]
        self.log("test_accuracy", accuracy, on_step=True, on_epoch=True, sync_dist=True)

        self.test_pred.extend(predictions.cpu().numpy())
        self.test_label.extend(labels.cpu().numpy())

    def on_test_epoch_end(self):
        # assert len(self.test_pred) == 400
        # assert len(self.test_label) == 400
        print(f"acc: {accuracy_score(self.test_label, self.test_pred)}")
        print(f"macro f1: {f1_score(self.test_label, self.test_pred, average='macro')}")


def raw_test_model(model_cls, checkpoint: str, loader: DataLoader):
    model = model_cls.load_from_checkpoint(checkpoint)
    test_labels, test_preds = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to("cuda:0")
            attention_mask = batch["attention_mask"].to("cuda:0")
            labels = batch["labels"].to("cuda:0")
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(predictions.cpu().numpy())
    print(f"acc: {accuracy_score(test_labels, test_preds)}")
    print(f"macro f1: {f1_score(test_labels, test_preds, average='macro')}")
    print(f"macro precision: {precision_score(test_labels, test_preds, average='macro')}")
    print(f"macro recall: {recall_score(test_labels, test_preds, average='macro')}")
    error_type_counter = Counter(x for x, y in zip(test_labels, test_preds) if x != y)
    right_type_counter = Counter(x for x, y in zip(test_labels, test_preds) if x == y)
    print(len(test_labels), len(test_preds))
    print(f"error samples: {error_type_counter}")
    print(f"right samples: {right_type_counter}")


def train_model(model_cls, train_loader: DataLoader, valid_loader: DataLoader):
    model = model_cls()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        filename="best_model-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        dirpath="checkpoints/",
    )

    trainer = Trainer(
        devices=1,
        max_epochs=N_EPOCHS,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    train, valid, test = preprocess_dataset()
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    train_dataset = CommitDataset(train, tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    valid_dataset = CommitDataset(valid, tokenizer)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4)
    test_dataset = CommitDataset(test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # parser = ArgumentParser()
    # parser.add_argument("--train", action="store_true", default=False)
    # parser.add_argument("--test", type=str, default=None)
    # args = parser.parse_args()
    # args = {
    #     train_:True,
    #     test_:True,
    # }

    # if args.train_:
    train_model(CommitClassifier, train_loader, valid_loader)
    # if args.test_ is not None: