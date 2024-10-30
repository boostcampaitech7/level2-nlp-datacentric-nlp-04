import os
import yaml
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import Dataset
from huggingface_hub import HfApi, Repository, create_repo

import wandb
import re
from dotenv import load_dotenv

import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split


class BERTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        input_texts = data["text"]
        targets = data["target"]
        self.inputs = []
        self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx]["input_ids"].squeeze(0),
            "attention_mask": self.inputs[idx]["attention_mask"].squeeze(0),
            "labels": self.labels[idx].squeeze(0),
        }

    def __len__(self):
        return len(self.labels)


# seed 고정
def seed_fix(SEED=456):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# config parser로 가져오기
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml"
    )  # 입력 없을 시, 기본값으로 config.yaml을 가져옴
    return parser.parse_args()


def data_setting(test_size, max_length, SEED, train_path, tokenizer):
    data = pd.read_csv(train_path)
    dataset_train, dataset_valid = train_test_split(
        data, test_size=test_size, random_state=SEED
    )

    data_train = BERTDataset(dataset_train, tokenizer, max_length)
    data_valid = BERTDataset(dataset_valid, tokenizer, max_length)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )  # padding이 되어있지 않아도 자동으로 맞춰주는 역할

    return data_train, data_valid, data_collator


def compute_metrics(eval_pred):
    f1 = evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average="macro")


# 학습
def train(
    SEED,
    train_batch_size,
    eval_batch_size,
    learning_rate,
    model,
    output_dir,
    data_train,
    data_valid,
    data_collator,
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # logging_steps=100,
        # eval_steps=100,
        # save_steps=100,
        save_total_limit=2,
        learning_rate=float(learning_rate),
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return model


# 평가
def evaluating(model, tokenizer, test_path, output_dir):
    model.eval()
    preds = []

    dataset_test = pd.read_csv(test_path)

    for idx, sample in tqdm(
        dataset_test.iterrows(), total=len(dataset_test), desc="Evaluating"
    ):
        inputs = tokenizer(sample["text"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)

    dataset_test["target"] = preds
    dataset_test.to_csv(os.path.join(output_dir, "output.csv"), index=False)


# config 확인 (print)
def config_print(config, depth=0):
    for k, v in config.items():
        prefix = ["\t" * depth, k, ":"]

        if type(v) == dict:
            print(*prefix)
            config_print(v, depth + 1)
        else:
            prefix.append(v)
            print(*prefix)


def wandb_name(train_path, train_lr, train_batch_size, test_size, wandb_user_name):
    match = re.search(r"([^/]+)\.csv$", train_path)
    data_name = match.group(1) if match else "unknown"
    lr = train_lr
    bs = train_batch_size
    ts = test_size
    user_name = wandb_user_name
    return f"{user_name}_{data_name}_{lr}_{bs}_{ts}"


def upload_to_huggingface(model, tokenizer, hf_token, hf_organization, hf_repo_id):
    try:
        model.push_to_hub(
            repo_id=hf_repo_id, organization=hf_organization, use_auth_token=hf_token
        )
        tokenizer.push_to_hub(
            repo_id=hf_repo_id, organization=hf_organization, use_auth_token=hf_token
        )
        print(f"your model pushed successfully in {hf_repo_id}, hugging face")
    except Exception as e:
        print(f"An error occurred while uploading to Hugging Face: {e}")


def load_env_file(filepath=".env"):
    try:
        # .env 파일 로드 시도
        if load_dotenv(filepath):
            print(f".env 파일을 성공적으로 로드했습니다: {filepath}")
        else:
            raise FileNotFoundError  # 파일이 없으면 예외 발생
    except FileNotFoundError:
        print(f"경고: 지정된 .env 파일을 찾을 수 없습니다: {filepath}")
    except Exception as e:
        print(f"오류 발생: .env 파일 로드 중 예외가 발생했습니다: {e}")


if __name__ == "__main__":
    parser = get_parser()
    with open(os.path.join("../config", parser.config)) as f:
        CFG = yaml.safe_load(f)

    # 허깅페이스 API키 관리
    load_env_file("../setup/.env")

    # config의 파라미터를 불러와 변수에 저장함.
    # parser을 사용하여 yaml 가져오기 & parser 입력이 없으면, default yaml을 가져오기
    SEED = CFG["SEED"]

    # default는 False, Debug 동작설정
    DEBUG_MODE = CFG.get("DEBUG", False)

    train_path = CFG["data"]["train_path"]
    test_path = CFG["data"]["test_path"]
    output_dir = CFG["data"]["output_dir"]
    test_size = CFG["data"]["test_size"]
    max_length = CFG["data"]["max_length"]

    config_train = CFG["train"]
    train_batch_size = CFG["train"]["train_batch_size"]
    eval_batch_size = CFG["train"]["eval_batch_size"]
    learning_rate = CFG["train"]["lr"]

    wandb_project = CFG["wandb"]["project"]
    wandb_user_name = CFG["wandb"]["entity"]

    # Hugging Face 업로드 설정 확인 없어도 오류안뜨도록 .get형태로 불러옴
    hf_config = CFG.get("huggingface", {})
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    hf_organization = "paper-company"
    hf_repo_id = hf_config.get("repo_id")

    if DEBUG_MODE:
        print("Debug mode is ON. Displaying config parameters:")
        config_print(CFG)

    wandb.init(
        project=wandb_project,
        name=wandb_name(
            train_path, learning_rate, train_batch_size, test_size, wandb_user_name
        ),
    )

    seed_fix(SEED)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if DEBUG_MODE:
        print(f"DEVICE : {DEVICE}")

    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=7
    ).to(DEVICE)

    data_train, data_valid, data_collator = data_setting(
        test_size, max_length, SEED, train_path, tokenizer
    )

    trained_model = train(
        SEED,
        train_batch_size,
        eval_batch_size,
        learning_rate,
        model,
        output_dir,
        data_train,
        data_valid,
        data_collator,
    )

    evaluating(trained_model, tokenizer, test_path, output_dir)

    if not (hf_token or hf_repo_id):
        print("Hugging Face 설정이 누락되었습니다. 모델 업로드가 실행되지 않습니다.")
    else:
        # 모델 업로드
        upload_to_huggingface(
            trained_model,
            tokenizer,
            hf_token,
            hf_organization,
            f"{hf_repo_id}_{wandb_user_name}",
        )

    wandb.finish()
