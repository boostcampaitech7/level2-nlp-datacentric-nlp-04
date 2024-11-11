import argparse
from collections import defaultdict
import csv
import random
import re

from prompt import get_prompt_synonyms
import torch
from tqdm import tqdm

from .model import init_model


def get_synonyms(word, model, tokenizer, device, args):
    prompt = get_prompt_synonyms(args.prompt_version, word)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=args.max_length,
            num_return_sequences=args.num_return_sequences,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    synonyms = re.findall(r":\s*(.*?)(?:\n|$)", response)
    if synonyms:
        return [syn.strip() for syn in synonyms[0].split(",") if syn.strip()]
    return []


def augment_text(text, model, tokenizer, device, args):
    words = text.split()
    augmented_words = []
    for word in words:
        if random.random() < args.random_ratio:
            synonyms = get_synonyms(word, model, tokenizer, device, args)
            if synonyms:
                augmented_words.append(random.choice(synonyms))
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)
    return " ".join(augmented_words)


if __name__ == "__main__":
    # ArgumentParser 설정
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        help="사용할 모델 이름",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../data/aug_syn_input.csv",
        help="입력 CSV 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/aug_syn_output.csv",
        help="출력 CSV 파일 경로",
    )
    parser.add_argument("--min-row", type=int, default=3, help="처리할 최소 행 개수")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10,
        help="클래스당 증강할 샘플 수",
    )
    parser.add_argument("--prompt-version", type=int, default=1, help="사용할 프롬프트 버전")
    # get_synonyms 함수 인자들
    parser.add_argument("--max-length", type=int, default=100, help="생성할 최대 토큰 길이")
    parser.add_argument("--num-return-sequences", type=int, default=1, help="생성할 시퀀스 수")
    parser.add_argument("--random-ratio", type=float, default=0.3, help="동의어 대체 확률")

    args = parser.parse_args()

    # 모델, 토크나이저, 디바이스 설정
    model, tokenizer, device = init_model(args.model_name)

    # 타겟 클래스별 데이터 저장을 위한 딕셔너리 초기화
    data_by_target = defaultdict(list)

    with open(args.input, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header = next(reader)  # 헤더 읽기

        for row in reader:
            if len(row) < args.min_row:  # 데이터가 부족한 경우 건너뛰기
                continue
            id, text, target = row
            data_by_target[target].append((id, text))

    # CSV 파일에 증강된 데이터 쓰기
    with open(args.output, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # 헤더 쓰기

        for target, samples in data_by_target.items():
            # 각 타겟 클래스에서 샘플을 무작위로 선택하고 증강 수행
            selected_samples = random.sample(samples, min(args.samples_per_class, len(samples)))

            for id, text in tqdm(selected_samples, desc=f"Processing Target {target}"):
                writer.writerow([id, text, target])  # 원본 데이터 쓰기

                # 증강된 데이터 생성 및 쓰기
                augmented_text = augment_text(text, model, tokenizer, device, args)
                writer.writerow([f"{id}_aug", augmented_text, target])

    print(f"데이터 증강이 완료되었습니다. 결과가 '{args.output}' 파일에 저장되었습니다.")
