import os

from cleanlab.filter import find_label_issues
from main import BERTDataset, compute_metrics, data_setting
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import wandb
import yaml

from ..utils import (
    HF_TEAM_NAME,
    check_dataset,
    config_print,
    get_parser,
    load_env_file,
    seed_fix,
    set_debug_mode,
    wandb_name,
)


"""
clean_lab/class_relabel.py와 다른 점은, is_noise==1이면 data_train으로, is_noise==0이면 eval_train으로 split합니다.
k-fold를 사용하지 않습니다.
1. 훈련된 모델: 훈련이 완료된 모델이 지정된 output 경로에 저장됩니다.
2. retrained_data.csv: 이 파일은 새롭게 라벨링된 훈련 데이터셋으로, 이후 모델 훈련 시 사용할 수 있습니다.
3. cleaned_data.csv: 각 라벨의 확률이 포함된 데이터셋으로, 모델의 예측 결과를 확인할 수 있습니다.
"""


def clean_labels(data, pred_probs):
    ordered_label_issues = find_label_issues(
        labels=data["target"],
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    data["new_label"] = data["target"]

    print("Number of label issues found:", len(ordered_label_issues))

    for issue_idx in ordered_label_issues:
        new_label = data.iloc[issue_idx]["prob"].argmax()
        data.iloc[issue_idx, data.columns.get_loc("new_label")] = new_label

    return data


def save_modified_data(output_dir, data):
    modified_data = data[["ID", "text", "new_label"]].copy()

    modified_data.rename(columns={"new_label": "target"}, inplace=True)

    modified_data.to_csv(os.path.join(output_dir, "retrained_data.csv"), index=False)
    print("수정된 데이터가 retrained_data.csv로 저장되었습니다.")


def train_for_clean_labels_modified(
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

    data = pd.read_csv(train_path)
    train_data = data[data["is_noise"] == 1]
    test_data = data[data["is_noise"] == 0]
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    data_train = BERTDataset(train_data, tokenizer, max_length)
    data_eval = BERTDataset(test_data, tokenizer, max_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.eval()

    predictions = trainer.predict(data_eval)
    olist = torch.from_numpy(predictions.predictions)
    olist = torch.softmax(olist, dim=1).detach().numpy()

    test_data["prob"] = list(olist)
    pred_probs = np.stack(test_data["prob"])

    cleaned_data = clean_labels(test_data, pred_probs)

    cleaned_data.to_csv(os.path.join(output_dir, "cleaned_data.csv"), index=False)
    save_modified_data(output_dir, cleaned_data)

    return model


if __name__ == "__main__":
    # ArgumentParser 설정
    parser = get_parser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="klue/bert-base",
        help="사용할 모델 이름",
    )

    args = parser.parse_args()
    parser = get_parser()
    with open(os.path.join("../config", parser.config)) as f:
        CFG = yaml.safe_load(f)

    SEED = CFG["SEED"]

    set_debug_mode(CFG.get("DEBUG", False))

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

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

    data_train, data_valid, data_collator = data_setting(test_size, max_length, SEED, train_path, tokenizer)

    trained_model = train_for_clean_labels_modified(
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

    wandb.finish()
