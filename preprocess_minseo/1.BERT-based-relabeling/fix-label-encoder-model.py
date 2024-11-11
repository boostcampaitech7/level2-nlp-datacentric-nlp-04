import logging
import os

from compare_label_changes import compare_target_changes
from datasets import Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import yaml


# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


# YAML 파일 로드 함수
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# 설정 로드
config = load_config()

# GPU 사용 여부 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"사용 중인 디바이스: {device}")

# 파일 경로 설정
train_data_path = config["paths"]["train_path"]
test_data_path = config["paths"]["test_path"]
checkpoint_dir = config["paths"]["checkpoint_dir"]
output_file_path = config["paths"]["output_path"]

# 모델과 학습 관련 설정
model_name = config["model"]["name"]
train_checkpoint = config["model"]["train_checkpoint"]

# 데이터 전처리 설정
padding = config["tokenization"]["padding"]
truncation = config["tokenization"]["truncation"]

# 데이터 로드
logger.info("데이터 로딩 중...")
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# 학습 데이터셋 준비 (train, validation으로 분할)
logger.info("훈련 데이터를 학습과 검증 데이터로 분할 중...")
train_data, val_data = train_test_split(train_df, test_size=0.3, random_state=42)
train_data = Dataset.from_pandas(train_data)
val_data = Dataset.from_pandas(val_data)

# test 데이터셋 준비
test_data = Dataset.from_pandas(test_df)

# 데이터셋을 DatasetDict 형식으로 변환
dataset = DatasetDict({"train": train_data, "validation": val_data, "test": test_data})

# 모델과 토크나이저 설정
logger.info(f"모델 및 토크나이저 로딩 중: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

if train_checkpoint and os.path.exists(checkpoint_dir):
    logger.info("체크포인트에서 모델 로드 중...")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
else:
    logger.info("새 모델을 초기화하여 학습을 준비 중...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(train_df["target"].unique())
    ).to(device)


# 데이터 전처리 함수
def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples["text"], padding=padding, truncation=truncation)
    if "target" in examples:
        tokenized_inputs["labels"] = examples["target"]
    return tokenized_inputs


# 데이터셋에 전처리 적용
logger.info("데이터셋에 전처리 적용 중...")
encoded_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 개별 학습 파라미터 설정
output_dir = config["training_args"]["output_dir"]
evaluation_strategy = config["training_args"]["evaluation_strategy"]
save_strategy = config["training_args"]["save_strategy"]
save_steps = config["training_args"]["save_steps"]
save_total_limit = config["training_args"]["save_total_limit"]
learning_rate = config["training_args"]["learning_rate"]
per_device_train_batch_size = config["training_args"]["per_device_train_batch_size"]
per_device_eval_batch_size = config["training_args"]["per_device_eval_batch_size"]
num_train_epochs = config["training_args"]["num_train_epochs"]

# 학습 설정
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy=evaluation_strategy,
    save_strategy=save_strategy,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    learning_rate=learning_rate,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    num_train_epochs=num_train_epochs,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 학습 여부에 따라 학습 실행
if train_checkpoint:
    logger.info("모델 학습 시작...")
    trainer.train()
    logger.info("모델 학습 완료 및 체크포인트 저장됨")

# test 데이터에 대해 예측
logger.info("테스트 데이터에 대한 예측 시작")
predictions = trainer.predict(encoded_dataset["test"])
pred_labels = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()

# test 데이터프레임에 예측 결과 추가
test_df["target"] = pred_labels

# 예측 결과를 포함한 test.csv 파일 저장
test_df.to_csv(output_file_path, index=False)
logger.info(f"예측 결과가 {output_file_path}에 저장되었습니다.")

# 라벨 변경된 행의 개수 확인 및 로그 기록
changed_rows_count = compare_target_changes(output_file_path, test_data_path)
logger.info(f"총 변경된 라벨 개수: {changed_rows_count}")
