"""
denoise.py 를 실행후 processing.py로 후처리 해주세요
"""

import argparse
import csv
import logging
import re

import pandas as pd
import torch
from tqdm import tqdm

from .model import init_model
from .prompt import get_prompt_denoise


def restore_headline(noisy_headline, prompt_version, model, tokenizer, device, args):
    prompt = get_prompt_denoise(prompt_version, noisy_headline)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
        )

    restored_headline = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # "복원된 제목:" 이후의 텍스트를 추출
    if "복원된 제목:" in restored_headline:
        restored_headline = restored_headline.split("복원된 제목:")[-1].strip()

    # "답변:" 부분 제거
    restored_headline = re.sub(r"^답변:\s*", "", restored_headline)

    # 필요에 따라 추가적인 처리...
    return restored_headline.strip()


if __name__ == "__main__":
    # ArgumentParser 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", type=int, default=1, help="프롬프트 버전")
    parser.add_argument(
        "--input",
        type=str,
        default="../data/21_relabeling_2800.csv",
        help="입력 CSV 파일 경로",
    )
    parser.add_argument(
        "--semi-output",
        type=str,
        default="../data/semi-final3.csv",
        help="중간 출력 CSV 파일 경로",
    )
    parser.add_argument(
        "--final-output",
        type=str,
        default="../data/final3.csv",
        help="최종 출력 CSV 파일 경로",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=15,
        help="",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=2,
        help="",
    )
    parser.add_argument("--top-k", type=int, default=50, help="")
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="")

    args = parser.parse_args()
    prompt_version = args.prompt_version
    input_file = args.input
    semi_final_output_file = args.semi_output
    final_output_file = args.final_output

    # pandas 디스플레이 설정
    pd.set_option("display.max_rows", None)

    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # CSV 파일 읽기, 데이터 타입 지정
    logging.info(f"CSV 파일 '{input_file}' 읽는 중...")
    df = pd.read_csv(
        input_file,
        quoting=csv.QUOTE_ALL,
        dtype={"ID": str, "text": str, "target": int, "is_noise": int},
    )
    logging.info(f"총 {len(df)} 개의 행을 읽었습니다.")

    # is_noise가 1인 항목 필터링
    noisy_df = df[df["is_noise"] == 1]
    logging.info(f"노이즈가 있는 {len(noisy_df)} 개의 행을 찾았습니다.")

    # 초반 100개 항목만 선택
    noisy_df_subset = noisy_df.head(1602)

    # 모델 초기화
    model, tokenizer, device = init_model(args.model_name)

    # 디노이징 수행
    logging.info("디노이징 작업 시작...")
    tqdm.pandas()
    noisy_df_subset["denoised_text"] = noisy_df_subset["text"].progress_apply(
        lambda x: restore_headline(x, prompt_version, model, tokenizer, device, args)
    )
    logging.info("디노이징 작업 완료.")

    # 결과 DataFrame 생성 (is_noise 열은 포함하지 않음)
    result_df = noisy_df_subset[["ID", "denoised_text", "target"]]
    result_df = result_df.rename(columns={"denoised_text": "text"})

    logging.info(f"결과를 '{semi_final_output_file}'에 저장 중...")
    result_df.to_csv(semi_final_output_file, index=False, encoding="utf-8")

    # is_noise가 0인 항목 필터링
    non_noisy_df = df[df["is_noise"] == 0]
    logging.info(f"is_noise가 0인 {len(non_noisy_df)} 개의 행을 찾았습니다.")

    # 두 DataFrame 합치기
    final_df = pd.concat([result_df, non_noisy_df[["ID", "text", "target"]]], ignore_index=True)

    # 데이터프레임 섞기
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    # 결과를 CSV 파일로 저장 (is_noise 열은 포함하지 않음)
    logging.info(f"결과를 '{final_output_file}'에 저장 중...")
    final_df.to_csv(final_output_file, index=False, encoding="utf-8")

    logging.info(f"디노이징 완료. 결과가 '{final_output_file}' 파일에 저장되었습니다.")
