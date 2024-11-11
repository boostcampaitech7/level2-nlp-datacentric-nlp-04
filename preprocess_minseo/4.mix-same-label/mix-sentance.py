import pandas as pd
import yaml


# config 파일 로드 함수
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# 설정 로드
config = load_config()

# 파일 경로 설정
input_path = config["input_path"]

# 출력 파일 경로 설정 (iterations에 따라 파일명 자동 지정)
output_filename = (
    f"{input_path.replace('.csv', '')}"
    f"_aug{config['iterations']['combine']}{config['iterations']['combine_reversed']}.csv"
)

# 데이터 불러오기
df = pd.read_csv(input_path)
print("원본 데이터프레임:\n", df)

# 라벨별 데이터프레임 분리
label_dfs = {label: df[df["target"] == label].copy() for label in range(7)}


# 문장 절반 조합 함수
def combine_halves_randomly(df, label, iterations=800):
    augmented_data = []
    for i in range(iterations):
        text1, text2 = df["text"].sample(2, replace=True).tolist()

        # 문장을 스페이스바 기준으로 나눠서 절반씩 조합
        half_text1 = " ".join(text1.split()[: len(text1.split()) // 2])
        half_text2 = " ".join(text2.split()[len(text2.split()) // 2 :])
        augmented_text = f"{half_text1} {half_text2}"

        augmented_data.append({"ID": f"aug_{label}_{i}", "text": augmented_text, "target": label})

    return pd.DataFrame(augmented_data)


def combine_halves_reversed(df, label, iterations=200):
    augmented_data = []
    for i in range(iterations):
        text1, text2 = df["text"].sample(2, replace=True).tolist()

        # 문장을 스페이스바 기준으로 나눠서 절반씩 조합 (반대 순서)
        half_text1 = " ".join(text1.split()[: len(text1.split()) // 2])
        half_text2 = " ".join(text2.split()[len(text2.split()) // 2 :])
        augmented_text = f"{half_text2} {half_text1}"

        augmented_data.append({"ID": f"aug_{label}_{i}", "text": augmented_text, "target": label})

    return pd.DataFrame(augmented_data)


# 각 라벨별로 데이터 증강 수행
augmented_dfs = [
    combine_halves_randomly(df_label, label, iterations=config["iterations"]["combine"])
    for label, df_label in label_dfs.items()
]
re_augmented_dfs = [
    combine_halves_reversed(df_label, label, iterations=config["iterations"]["combine_reversed"])
    for label, df_label in label_dfs.items()
]

# 모든 증강된 데이터프레임 합치기
all_augmented_dfs = augmented_dfs + re_augmented_dfs

# 원본 데이터프레임과 합쳐서 최종 데이터 생성
df_combined = pd.concat([df] + all_augmented_dfs, ignore_index=True)
print("증강된 데이터프레임:\n", df_combined)

# 최종 결과 저장
df_combined.to_csv(output_filename, index=False)
print(f"데이터가 '{output_filename}' 파일에 저장되었습니다.")
