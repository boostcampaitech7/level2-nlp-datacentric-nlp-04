import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def init_model(model_name="LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"):
    # 모델 및 토크나이저 로드
    logging.info("모델 및 토크나이저 로딩 중...")
    model_name = model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPU 사용 가능 시 모델을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logging.info(f"모델을 {device}로 이동했습니다.")
    return model, tokenizer, device
