import argparse
import json
import os
import random
import time

from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np
import torch

from .gdrive_manager import GoogleDriveManager


DEBUG_MODE = False


def set_debug_mode(debug_mode):
    global DEBUG_MODE
    DEBUG_MODE = debug_mode

    debug_print("DEBUG_MODE가 설정되었습니다.")
    if not DEBUG_MODE:
        import warnings

        warnings.filterwarnings(action="ignore")


def debug_print(text):
    if DEBUG_MODE:
        print(text)


# config parser로 가져오기
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")  # 입력 없을 시, 기본값으로 config.yaml을 가져옴
    return parser.parse_args()


# config 확인 (print)
def config_print(config, depth=0):
    if depth == 0:
        print("*" * 40)
    for k, v in config.items():
        prefix = ["\t" * depth, k, ":"]

        if isinstance(v, dict):
            print(*prefix)
            config_print(v, depth + 1)
        else:
            prefix.append(v)
            print(*prefix)
    if depth == 0:
        print("*" * 40)


def wandb_name(train_file_name, train_lr, train_batch_size, test_size, user_name):
    data_name = train_file_name
    lr = train_lr
    bs = train_batch_size
    ts = test_size
    user_name = user_name
    return f"{user_name}_{data_name}_{lr}_{bs}_{ts}"


def load_env_file(filepath=".env"):
    try:
        # .env 파일 로드 시도
        if load_dotenv(filepath):
            debug_print(f".env 파일을 성공적으로 로드했습니다: {filepath}")
        else:
            raise FileNotFoundError  # 파일이 없으면 예외 발생
    except FileNotFoundError:
        debug_print(f"경고: 지정된 .env 파일을 찾을 수 없습니다: {filepath}")
    except Exception as e:
        debug_print(f"오류 발생: .env 파일 로드 중 예외가 발생했습니다: {e}")


def check_dataset(hf_organization, hf_token, train_file_name):
    """
    로컬에 데이터셋 폴더가 없으면 Hugging Face에서 데이터를 다운로드하여 로컬에 CSV로 저장하는 함수.
    데이터셋을 로컬에 저장만 하고 반환값은 없습니다.

    Parameters:
    - hf_organization (str): Hugging Face Organization 이름
    - hf_token (str): Hugging Face 토큰
    - train_file_name (str): 로컬에 저장할 train file 이름
    - dataset_repo_id (str): Hugging Face에 저장된 데이터셋 리포지토리 ID (기본값: datacentric-orginal)
    """
    # Define the folder path and file paths
    folder_path = os.path.join("..", "data")
    train_path = os.path.join(folder_path, "train.csv")

    # Check if local data folder exists
    if not os.path.exists(train_path):
        debug_print(f"로컬에 '{train_path}' 데이터가 존재하지 않습니다.허깅페이스에서 다운로드를 시도합니다.")

        # Load dataset from Hugging Face if local folder is missing
        full_repo_id = f"{hf_organization}/datacentric-{train_file_name}"
        dataset = load_dataset(full_repo_id, split="train", token=hf_token)

        # 데이터셋을 CSV로 저장
        dataset.to_pandas().to_csv(train_path, index=False)
        debug_print(f"데이터셋이 '{train_path}'에 다운로드되었습니다.")
    else:
        debug_print("로컬파일을 로드합니다.")


# seed 고정
def seed_fix(SEED=456):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def get_timestamp():
    return round(time.time())


def make_json_report(df):
    json_report = {}

    # 가정: 전체 데이터셋의 클래스별 샘플 수는 균등 할 것
    num_classes = 7
    public_percentages = [0.1428] * num_classes
    public_samples = 15000

    # Public 데이터셋의 클래스별 샘플 수
    public_distribution = {
        "class_distribution": {str(i): int(p * public_samples) for i, p in enumerate(public_percentages)},
        "num_classes": num_classes,
        "class_balance": {str(i): round(p, 4) for i, p in enumerate(public_percentages)},
    }

    # 현재 데이터의 타겟 레이블 분포 분석
    target_counts = df["target"].value_counts().to_dict()
    sorted_target_counts = dict(sorted(target_counts.items()))
    json_report["target_distribution"] = {
        "class_distribution": sorted_target_counts,
        "num_classes": len(sorted_target_counts),
        "class_balance": {str(label): round(count / len(df), 4) for label, count in sorted_target_counts.items()},
    }

    json_report["public_distribution"] = public_distribution

    # NumPy 타입을 처리하기 위한 커스텀 JSONEncoder
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return round(float(obj), 4)
            if isinstance(obj, np.ndarray):
                return [round(float(x), 4) for x in obj.tolist()]
            return super().default(obj)

    json_str = json.dumps(json_report, cls=NumpyEncoder, indent=4, ensure_ascii=False)
    return json_str


def upload_report(dataset_name, user_name, exp_name, result_df, result_json):
    timestamp = get_timestamp()
    drive_manager = GoogleDriveManager()
    # 데이터셋으로 폴더명을 찾고, 없다면 실험자 명으로 찾음
    folder_id = drive_manager.find_folder_id_by_name(f"{user_name}-{dataset_name}")
    if not folder_id:
        folder_id = drive_manager.find_folder_id_by_name(user_name)
    _ = drive_manager.upload_dataframe(result_df, f"{exp_name}_{timestamp}_output.csv", folder_id)
    _ = drive_manager.upload_json_data(result_json, f"{exp_name}_{timestamp}_report.json", folder_id)

    gdrive_url = os.path.join("https://drive.google.com/drive/folders", folder_id)
    print(f"구글 드라이브에 업로드 되었습니다: {gdrive_url}")
