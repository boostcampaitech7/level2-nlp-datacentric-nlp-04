import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import wandb
import yaml

from .utils import (
    HF_TEAM_NAME,
    check_dataset,
    config_print,
    get_parser,
    load_env_file,
    make_json_report,
    seed_fix,
    set_debug_mode,
    upload_report,
    wandb_name,
)


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


def data_setting(test_size, max_length, SEED, train_path, tokenizer, is_stratify=True):
    data = pd.read_csv(train_path)
    data.loc[:, "text"] = data["text"].astype("str")
    if is_stratify:
        # target 레이블을 기준으로 stratified split 적용
        dataset_train, dataset_valid = train_test_split(
            data, test_size=test_size, random_state=SEED, stratify=data["target"]
        )
    else:
        dataset_train, dataset_valid = train_test_split(data, test_size=test_size, random_state=SEED)

    data_train = BERTDataset(dataset_train, tokenizer, max_length)
    data_valid = BERTDataset(dataset_valid, tokenizer, max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # padding이 되어있지 않아도 자동으로 맞춰주는 역할

    return data_train, data_valid, data_collator


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro"),
    }


def compute_metrics_detailed(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # 전체 메트릭
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")

    # 클래스별 메트릭
    f1_per_class = f1_score(labels, predictions, average=None)
    class_accuracies = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        class_accuracies[f"accuracy_class_{label}"] = accuracy_score(labels[mask], predictions[mask])

    # 전체 메트릭
    results = {
        "accuracy": accuracy,
        "f1": f1_macro,
    }

    # 클래스별 메트릭 추가
    for i, label in enumerate(unique_labels):
        results[f"f1_class_{label}"] = f1_per_class[i]
    results.update(class_accuracies)

    return results


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
    exp_name,
):
    # 주의: 베이스라인에서 설정한 파라미터 건들지 말 것.
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
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
        run_name=exp_name,
        report_to="wandb",
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

    # 테이블 형식으로 detail evaluation 출력
    trainer.compute_metrics = compute_metrics_detailed
    final_metrics = trainer.evaluate()
    metrics_table = [[metric, f"{value:.4f}"] for metric, value in final_metrics.items() if isinstance(value, float)]
    print("\n" + tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

    return model


# 평가
def evaluating(device, model, tokenizer, eval_batch_size, test_path, output_dir):
    model.eval()
    preds = []

    dataset_test = pd.read_csv(test_path)

    # 배치 단위로 처리
    for i in tqdm(range(0, len(dataset_test), eval_batch_size), desc="Evaluating"):
        # 배치 데이터 추출
        batch_samples = dataset_test.iloc[i : i + eval_batch_size]
        texts = batch_samples["text"].tolist()

        # 배치 단위로 토크나이징
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

        # 예측
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)

    dataset_test["target"] = preds
    dataset_test.to_csv(os.path.join(output_dir, "output.csv"), index=False)
    return dataset_test


if __name__ == "__main__":
    parser = get_parser()
    with open(os.path.join("../config", parser.config), encoding="utf-8") as f:
        CFG = yaml.safe_load(f)

    # config의 파라미터를 불러와 변수에 저장함.
    # parser을 사용하여 yaml 가져오기 & parser 입력이 없으면, default yaml을 가져오기
    SEED = CFG["SEED"]

    # default는 False, Debug 동작설정
    set_debug_mode(CFG.get("DEBUG", False))

    # 추후 추가한 기능이기 때문에 config에 없음을 고려하여 default값을 부여합니다.
    is_stratify = CFG.get("datashuffle_stratify", False)

    train_file_name = CFG["data"]["train_name"]
    test_file_name = CFG["data"]["test_name"]
    output_dir = CFG["data"]["output_dir"]
    test_size = CFG["data"]["test_size"]
    max_length = CFG["data"]["max_length"]

    config_train = CFG["train"]
    train_batch_size = CFG["train"]["train_batch_size"]
    eval_batch_size = CFG["train"]["eval_batch_size"]
    learning_rate = CFG["train"]["lr"]

    user_name = CFG["exp"]["username"]
    upload_gdrive = CFG["gdrive"]["upload"]

    # wandb 설정
    wandb_project = CFG["wandb"]["project"]
    wandb_entity = CFG["wandb"]["entity"]

    exp_name = wandb_name(train_file_name, learning_rate, train_batch_size, test_size, user_name)
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=exp_name,
    )

    # HuggingFace API키 및 설정
    load_env_file("../setup/.env")
    hf_config = CFG.get("huggingface", {})
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    hf_organization = HF_TEAM_NAME

    config_print(CFG)

    # 로컬에 있는지 체크, 다운로드
    check_dataset(hf_organization, hf_token, train_file_name)

    # link data
    train_path = os.path.join("..", "data", f"{train_file_name}.csv")
    test_path = os.path.join("..", "data", f"{test_file_name}.csv")

    seed_fix(SEED)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 모델명 절대 수정하지 말 것.
    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

    data_train, data_valid, data_collator = data_setting(
        test_size, max_length, SEED, train_path, tokenizer, is_stratify
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
        exp_name,
    )

    dataset_test = evaluating(DEVICE, trained_model, tokenizer, eval_batch_size, test_path, output_dir)

    # upload output & report to gdrive
    if upload_gdrive:
        json_report = make_json_report(dataset_test)
        upload_report(
            dataset_name=train_file_name,
            user_name=user_name,
            exp_name=exp_name,
            result_df=dataset_test,
            result_json=json_report,
        )

    wandb.finish()
