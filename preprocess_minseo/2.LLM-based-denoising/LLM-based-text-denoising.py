import logging
import os
import random
import re
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# YAML 설정 파일 로드 함수
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# 설정 불러오기
config = load_config()

# 모델 및 토크나이저 설정
model_name = config["model"]["name"]
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
logger.info(f"모델과 토크나이저 로드 완료 - 디바이스: {device}")

# 데이터 로드 및 라벨별 데이터프레임 분할
few_shot_df = pd.read_csv(config["paths"]["few_shot_data"])  # few-shot 예제 데이터
need_denoise_df = pd.read_csv(config["paths"]["denoise_data"])  # 노이즈 제거 대상 데이터
need_denoise_df["denoised_text"] = ""

# few-shot 데이터를 라벨별로 분류
label_dfs = {label: few_shot_df[few_shot_df["target"] == label] for label in few_shot_df["target"].unique()}


# 노이즈 추가 함수
def add_extended_ascii_noise(text, noise_level=0.5, ascii_ratio=0.93):
    noisy_text = ""
    for char in text:
        if random.random() < noise_level:
            if random.random() < ascii_ratio:
                noisy_text += chr(random.randint(33, 126))  # ASCII 범위 내 특수 문자, 대소문자, 숫자
            else:
                noisy_text += chr(random.randint(0x4E00, 0x9FFF))  # 한자 유니코드 범위
        else:
            noisy_text += char
    return noisy_text


# 예시 생성 함수
def generate_few_shot_examples(label_dfs, target_label):
    few_shot_examples = label_dfs[target_label].sample(3)
    sample_list = "예시: \n"
    for _, example_row in few_shot_examples.iterrows():
        original_example = example_row["text"]
        noisy_example = add_extended_ascii_noise(
            original_example, noise_level=config["noise"]["level"], ascii_ratio=config["noise"]["ascii_ratio"]
        )
        sample_list += f"입력된 제목: {noisy_example}\n복원된 제목: {original_example}\n\n"
    return sample_list


# 노이즈 제거 함수
def denoise_text_with_few_shot(row, label_dfs):
    target_label = row["target"]
    noisy_text = row["text"]

    sample_list = generate_few_shot_examples(label_dfs, target_label)

    prompt = f"""다음은 아스키코드가 치환되는 방식으로 많은 노이즈가 있는 한국어 뉴스 기사 제목입니다.
    노이즈가 들어가지 않은 한글부분을 참고하여 원래의 정확한 한국어 제목으로 복원해주세요.
    앞뒤에 설명 붙이지 말고, 반드시 복원된 제목만 대답하세요.
    아래의 예시는 동일한 주제의 복원된 뉴스기사 제목입니다.
    예시를 참고하여 예시와 반드시 동일한 주제로 복원하세요.
    입력된 제목보다 짧지 않도록 생성하세요.
    {sample_list}
    입력된 제목: {noisy_text}
    복원된 제목:"""

    inputs = tokenizer.encode(
        prompt, return_tensors="pt", max_length=config["model"]["max_length"], truncation=True
    ).to(device)
    outputs = model.generate(
        inputs,
        max_new_tokens=config["model"]["max_new_tokens"],
        num_return_sequences=config["model"]["num_return_sequences"],
        no_repeat_ngram_size=config["model"]["no_repeat_ngram_size"],
        top_k=config["model"]["top_k"],
        top_p=config["model"]["top_p"],
        temperature=config["model"]["temperature"],
        do_sample=True,
    )
    restored_headline = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "복원된 제목:" in restored_headline:
        restored_headline = restored_headline.split("복원된 제목:")[-1].strip()
    restored_headline = re.sub(r"^답변:\s*", "", restored_headline)

    try:
        restored_headline = restored_headline.splitlines()[0]
    except IndexError:
        restored_headline = ""

    return restored_headline.strip()


# 진행 상황 저장 및 불러오기
save_path = config["paths"]["save_path"]
if os.path.exists(save_path):
    need_denoise_df = pd.read_csv(save_path)
    last_processed_index = need_denoise_df["denoised_text"].last_valid_index() or -1
else:
    last_processed_index = -1

total_rows = len(need_denoise_df)
logger.info(f"{last_processed_index + 1}번째 행부터 시작합니다 (총 {total_rows} 행)")

# tqdm을 이용한 진행 상태 표시 및 denoise 수행
for index, row in tqdm(need_denoise_df.iterrows(), total=total_rows, desc="노이즈 제거 진행"):
    if index <= last_processed_index:
        continue

    start_time = time.time()
    try:
        denoised_text = denoise_text_with_few_shot(row, label_dfs)
    except Exception as e:
        logger.error(f"{index}번째 행 처리 중 오류 발생: {e}")
        denoised_text = ""

    need_denoise_df.at[index, "denoised_text"] = denoised_text
    elapsed_time = time.time() - start_time
    logger.info(f"{index + 1}/{total_rows} 행 처리 완료 - 소요 시간: {elapsed_time:.2f}초")

    # 중간 진행 저장
    if (index + 1) % config["save_frequency"] == 0:
        need_denoise_df.to_csv(save_path, index=False)
        logger.info(f"{index + 1}번째 행에서 중간 저장 완료")

# 최종 저장
need_denoise_df.to_csv(save_path, index=False)
logger.info("모든 행 처리 완료. 최종 결과가 저장되었습니다.")
