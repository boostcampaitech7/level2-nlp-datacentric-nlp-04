import os

from cleanlab.filter import find_label_issues
from main import BERTDataset, compute_metrics, data_setting, evaluating
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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
1. 훈련된 모델: 훈련이 완료된 모델이 지정된 output 경로에 저장됩니다.
2. retrained_data.csv: 이 파일은 새롭게 라벨링된 훈련 데이터셋으로, 이후 모델 훈련 시 사용할 수 있습니다.
3. cleaned_data.csv: 각 라벨의 확률이 포함된 데이터셋으로, 모델의 예측 결과를 확인할 수 있습니다.
"""


def clean_labels(data, pred_probs):
    # Cleanlab의 find_label_issues 함수를 사용하여 레이블 이슈 찾기
    ordered_label_issues = find_label_issues(
        labels=data["target"],
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    # 새로운 레이블 열 생성 (초기값은 원래 레이블)
    data["new_label"] = data["target"]

    print("Number of label issues found:", len(ordered_label_issues))

    # 식별된 레이블 이슈에 대해 새로운 레이블 할당
    for issue_idx in ordered_label_issues:
        new_label = data.iloc[issue_idx]["prob"].argmax()
        data.iloc[issue_idx, data.columns.get_loc("new_label")] = new_label

    return data


def save_modified_data(output_dir, data):
    # 새로운 데이터프레임 생성: ID, text, new_label 열만 포함
    modified_data = data[["ID", "text", "new_label"]].copy()

    # 새로운 라벨로 'target' 열 이름 변경
    modified_data.rename(columns={"new_label": "target"}, inplace=True)

    # 수정된 데이터프레임을 CSV 파일로 저장
    modified_data.to_csv(os.path.join(output_dir, "retrained_data.csv"), index=False)
    print("수정된 데이터가 retrained_data.csv로 저장되었습니다.")


def train_for_clean_labels(
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
    data.loc[:, "text"] = data["text"].astype("str")
    prob_list = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    for i, (train_index, test_index) in enumerate(skf.split(data, data["target"])):
        data_train = BERTDataset(data.iloc[train_index], tokenizer, max_length)
        data_eval = BERTDataset(data.iloc[test_index], tokenizer, max_length)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data_train,
            eval_dataset=data_valid,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        model.eval()

        # 각 폴드의 검증 데이터에 대한 예측 확률 계산
        predictions = trainer.predict(data_eval)
        olist = torch.from_numpy(predictions.predictions)
        olist = torch.softmax(olist, dim=1).detach().numpy()

        prob_list.extend(zip(test_index, olist))

    # 예측 확률을 원래 데이터 순서대로 정렬
    prob_list.sort(key=lambda x: x[0])

    # 원본 데이터에 확률 추가
    data["prob"] = [value for idx, value in prob_list]

    # 예측 확률을 numpy 배열로 변환
    pred_probs = np.stack(data["prob"])

    # Cleanlab을 사용하여 레이블 정제
    cleaned_data = clean_labels(data, pred_probs)

    # 수정된 레이블을 포함한 새로운 데이터셋 저장
    cleaned_data.to_csv(os.path.join(output_dir, "cleaned_data.csv"), index=False)

    # 수정된 데이터를 저장하는 함수 호출
    save_modified_data(output_dir, cleaned_data)

    return model  # 훈련된 모델 반환


if __name__ == "__main__":
    # ArgumentParser 설정
    parser = get_parser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="vaiv/kobigbird-roberta-large",
        help="사용할 모델 이름",
    )

    args = parser.parse_args()

    parser = get_parser()
    with open(os.path.join("../config", parser.config)) as f:
        CFG = yaml.safe_load(f)

    # config의 파라미터를 불러와 변수에 저장함.
    # parser을 사용하여 yaml 가져오기 & parser 입력이 없으면, default yaml을 가져오기
    SEED = CFG["SEED"]

    # default는 False, Debug 동작설정
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

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

    data_train, data_valid, data_collator = data_setting(test_size, max_length, SEED, train_path, tokenizer)

    trained_model = train_for_clean_labels(
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

    wandb.finish()
