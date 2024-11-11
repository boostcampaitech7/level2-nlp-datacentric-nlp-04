import os

from datasets import load_dataset
from huggingface_hub import HfApi
from main import get_parser, load_env_file
import yaml


# 절대 바꾸지 않을 값이므로 따로 config 처리X
HF_TEAM_NAME = "paper-company"
HF_PROJECT_NAME = "datacentric"


def upload_train_file_to_hub(file_name, token, private=True):
    """
    폴더 내 데이터 파일들을 Hugging Face Hub에 데이터셋으로 업로드하는 함수.

    Parameters:
    - file_name (str): 업로드할 로컬 데이터 파일 이름
    - token (str): Hugging Face 액세스 토큰. 쓰기 권한 필요
    - private (bool): True면 비공개, False면 공개 설정

    Returns:
    - None
    """
    api = HfApi()
    repo_id = f"{HF_TEAM_NAME}/{HF_PROJECT_NAME}-{file_name}"

    # 리포지토리 존재 여부 확인
    try:
        api.repo_info(repo_id, repo_type="dataset", token=token)
        print(f"'{repo_id}' 리포지토리가 이미 존재합니다. 기존 리포지토리에 데이터셋을 업로드합니다.")
    except Exception:
        # 리포지토리가 없으면 생성
        print(f"'{repo_id}' 리포지토리가 존재하지 않습니다. 새로 생성한 후 업로드합니다.")
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, token=token)

    # 파일 경로 설정
    file_path = os.path.join("..", "data", f"{file_name}.csv")
    if not os.path.exists(file_path):
        print(f"파일 '{file_path}'이 존재하지 않습니다.")
        return

    # 데이터셋 로드 및 업로드
    dataset = load_dataset("csv", data_files={"train": file_path})
    dataset.push_to_hub(repo_id, token=token)
    print(f"데이터셋이 '{repo_id}'에 업로드되었습니다.")


if "__main__" == __name__:
    parser = get_parser()
    with open(os.path.join("../../config", parser.config)) as f:
        CFG = yaml.safe_load(f)

    # 허깅페이스 API키 관리
    load_env_file("../../setup/.env")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    # 업로드할 파일명 지정
    file_name = CFG["data"]["train_name"]

    upload_train_file_to_hub(file_name, hf_token, private=True)
