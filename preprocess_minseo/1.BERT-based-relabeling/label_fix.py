# 필요한 라이브러리 임포트
import os

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


# GPU 사용 여부 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드
train_df = pd.read_csv("../../data/5_base_noise_detected_train.csv")
# test_df = pd.read_csv("../../data/5_base_noise_detected_test.csv")

# 디노이징 추론
test_df = pd.read_csv("../denoised_results_wo_error.csv")

# 학습 데이터셋 준비 (train, validation으로 분할)
train_data, val_data = train_test_split(train_df, test_size=0.3, random_state=42)
train_data = Dataset.from_pandas(train_data)
val_data = Dataset.from_pandas(val_data)

# test 데이터셋 준비
test_data = Dataset.from_pandas(test_df)

# 데이터셋을 DatasetDict 형식으로 변환
dataset = DatasetDict({"train": train_data, "validation": val_data, "test": test_data})

# 모델과 토크나이저 설정
model_name = "kakaobank/kf-deberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 모델 체크포인트 로드 설정
train_checkpoint = False  # 학습할 경우 True, 기존 체크포인트로 테스트만 할 경우 False
checkpoint_dir = "./checkpoint"  # 체크포인트 저장 경로

if train_checkpoint and os.path.exists(checkpoint_dir):
    # 체크포인트에서 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
else:
    # 새로 모델을 초기화하여 학습
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(train_df["target"].unique())
    ).to(device)


# 데이터 전처리 함수
def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True)
    if "target" in examples:  # train 데이터의 경우
        tokenized_inputs["labels"] = examples["target"]  # 'labels'에 타깃 라벨을 추가
    return tokenized_inputs


# 데이터셋에 전처리 적용
encoded_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 모델 학습 설정
training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_steps=500,  # 학습 시 매 500 스텝마다 체크포인트 저장
    save_total_limit=2,  # 최근 2개의 체크포인트만 저장
    learning_rate=5e-6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],  # 평가 데이터 설정
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 학습 여부에 따라 학습 실행
if train_checkpoint:
    trainer.train()
    print("모델 학습 완료 및 체크포인트 저장")

# test 데이터에 대해 예측
predictions = trainer.predict(encoded_dataset["test"])
pred_labels = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()

# test 데이터프레임에 예측 결과 추가
test_df["target"] = pred_labels

# 예측 결과를 포함한 test.csv 파일 저장
test_df.to_csv("../../data/denoised_results_relabel.csv", index=False)

print("test_with_predictions.csv 파일이 생성되었습니다.")
