"""
T5 모델을 사용하여 디노이징(노이즈 제거)하는 작업을 수행
1. CSV 파일에서 텍스트 데이터 읽기
2. T5 모델을 사용하여 텍스트 디노이징
3. 처리된 결과를 새로운 CSV 파일로 저장
"""

import argparse

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# 텍스트 배치를 디노이징하는 함수 정의
def denoise(batch_texts, tokenizer, model):
    # 입력 텍스트를 토큰화하고 패딩 및 잘라내기 적용
    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
    # 모델을 사용하여 출력 생성
    outputs = model.generate(**inputs)
    # 생성된 출력을 디코딩하여 텍스트로 변환
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# 데이터를 처리하는 메인 함수 정의
def process_data(
    input_file,
    output_file,
    model_name="eenzeenee/t5-base-korean-summarization",
    batch_size=16,
):
    # 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 입력 데이터 읽기
    data = pd.read_csv(input_file)

    results = []

    # 배치 단위로 데이터 처리
    for i in tqdm(range(0, len(data), batch_size)):
        # 현재 배치의 텍스트 추출
        batch = data["text"][i : i + batch_size].tolist()
        # 배치 디노이징 수행
        batch_results = denoise(batch, tokenizer, model)
        # 결과를 전체 결과 리스트에 추가
        results.extend(batch_results)

    # 'text' 열을 디노이징된 결과로 업데이트
    data["text"] = results

    # 처리된 데이터를 CSV 파일로 저장
    data.to_csv(output_file, index=False)
    print(f"처리된 데이터가 {output_file}에 저장되었습니다.")


if __name__ == "__main__":
    # 인자 처리
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="eenzeenee/t5-base-korean-summarization",
        help="사용할 모델 이름",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../data/3_d_2800_hanzi_dictionary.csv",
        help="입력 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/tokenized_denoised_data2.csv",
        help="출력 파일 경로",
    )
    args = parser.parse_args()

    # 데이터 처리 함수 호출
    process_data(args.model_name, args.input, args.output)
