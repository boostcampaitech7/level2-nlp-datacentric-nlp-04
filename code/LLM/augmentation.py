import argparse
import csv
import re

from .model import init_model
from .prompt import get_prompt_article_to_title, get_prompt_title_to_article


def generate_article(prompt, prompt_end_word, model, tokenizer, device, args):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.article_max_tokens,
        num_return_sequences=args.article_num_sequences,
        no_repeat_ngram_size=args.article_no_repeat_ngram,
        temperature=args.article_temperature,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split(prompt_end_word)[1].strip()


def generate_new_title(prompt, model, tokenizer, device, args):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.title_max_tokens,
        num_return_sequences=args.title_num_sequences,
        no_repeat_ngram_size=args.title_no_repeat_ngram,
        temperature=args.title_temperature,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("새로운 제목:")[-1].strip()


def save_to_csv(data, filename):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["ID", "text", "target"])
        for i, (text, target) in enumerate(data):
            cleaned_text = re.sub(r'^"|"$', "", text)
            cleaned_text = re.sub(r"\s*이\s*제목은.*$", "", cleaned_text)
            cleaned_text = " ".join(cleaned_text.split())
            writer.writerow([f"ynat-v1_train_{i:05d}", cleaned_text, target])


def process_title(title, target, model, tokenizer, device, args):
    prompt_title_to_article, prompt_title_to_article_end_word = get_prompt_title_to_article(title)
    article = generate_article(
        prompt_title_to_article,
        prompt_title_to_article_end_word,
        model,
        tokenizer,
        device,
        args,
    )

    prompt_article_to_title = get_prompt_article_to_title(article)
    new_title = generate_new_title(prompt_article_to_title, model, tokenizer, device, args)
    return article, new_title, target


def read_csv(filename):
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 스킵
        return [(row[1], row[2]) for row in reader]  # (text, target) 튜플 리스트 반환


def process_csv(input_filename, output_filename, model, tokenizer, device, args):
    data = read_csv(input_filename)
    data.sort(key=lambda x: len(x[0]), reverse=True)

    if args.max_rows:
        data = data[: args.max_rows]

    processed_data = []
    for i, (title, target) in enumerate(data):
        article, new_title, _ = process_title(title, target, model, tokenizer, device, args)

        print(f"\n원래 제목: {title}")
        print(f"생성된 기사 내용:\n{article}")
        print(f"새로운 제목: {new_title}")
        print("-" * 50)

        processed_data.append((new_title, target))

    save_to_csv(processed_data, output_filename)
    print(f"처리 완료. 결과가 '{output_filename}' 파일에 저장되었습니다.")


if __name__ == "__main__":
    # ArgumentParser
    parser = argparse.ArgumentParser()
    # 기존 인자들
    parser.add_argument(
        "--model-name",
        type=str,
        default="LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        help="사용할 모델 이름",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../data/aug_input.csv",
        help="입력 CSV 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/aug_output.csv",
        help="출력 CSV 파일 경로",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=600,
        help="처리할 최대 행 수. 0이면 모든 행 처리",
    )

    # generate_article 함수 인자
    parser.add_argument("--article-max-tokens", type=int, default=200, help="기사 생성시 최대 토큰 수")
    parser.add_argument("--article-num-sequences", type=int, default=1, help="기사 생성시 시퀀스 수")
    parser.add_argument(
        "--article-no-repeat-ngram",
        type=int,
        default=2,
        help="기사 생성시 반복하지 않을 n-gram 크기",
    )
    parser.add_argument("--article-temperature", type=float, default=0.7, help="기사 생성시 샘플링 온도")

    # generate_new_title 함수 인자
    parser.add_argument("--title-max-tokens", type=int, default=20, help="제목 생성시 최대 토큰 수")
    parser.add_argument("--title-num-sequences", type=int, default=1, help="제목 생성시 시퀀스 수")
    parser.add_argument(
        "--title-no-repeat-ngram",
        type=int,
        default=2,
        help="제목 생성시 반복하지 않을 n-gram 크기",
    )
    parser.add_argument("--title-temperature", type=float, default=1.5, help="제목 생성시 샘플링 온도")

    args = parser.parse_args()

    # max_rows가 0이면 None으로 설정
    max_rows = None if args.max_rows == 0 else args.max_rows

    # 모델 초기화 및 데이터 처리
    model, tokenizer, device = init_model(args.model_name)
    process_csv(args.input, args.output, model, tokenizer, device, args)
