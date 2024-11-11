import argparse

import pandas as pd


def processing(
    input_path="../data/processing_input.csv",
    output_path="../data/processing_output.csv",
):
    # CSV 파일 읽기
    df = pd.read_csv(input_path)

    # text 열에서 \n 이후의 내용을 제거하는 함수 정의
    def clean_text(text):
        if isinstance(text, str):  # text가 문자열인 경우에만 처리
            return text.split("\n")[0]  # \n 이전의 텍스트만 반환
        return text  # 문자열이 아닌 경우 원래 값을 반환

    # clean_text 함수를 text 열에 적용
    df["text"] = df["text"].apply(clean_text)

    # 결과를 새로운 CSV 파일로 저장
    df.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    # ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="../data/processing_input.csv",
        help="입력 CSV 파일 경로",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/processing_output.csv",
        help="출력 CSV 파일 경로",
    )
    args = parser.parse_args()

    processing(args.input_path, args.output_path)
