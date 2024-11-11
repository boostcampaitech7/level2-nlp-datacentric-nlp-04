import logging
import os

import pandas as pd

from ..csv_updater import update_csv_and_save


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 기본 파일 경로 설정
base_dir = os.path.join("..", "..", "data")
model_result = os.path.join("..", "denoised_results_relabel.csv")
train_path = os.path.join(base_dir, "train_cleaning_label_5sj.csv")

# 교체하기 로직을 실행하려면 is_replace=True, 추가하기 로직을 실행하려면 is_replace=False로 설정
is_replace = True

# `is_replace` 값에 따라 output_path 설정
if is_replace:
    suffix = "_denoise_replace_relabel.csv"
else:
    suffix = "_denoise_add_relabel.csv"
output_path = train_path.replace(".csv", suffix)

if is_replace:
    # 교체하기 로직
    logger.info("노이즈 text 교체 진행")
    update_csv_and_save(
        base_file_path=train_path,
        update_file_path=model_result,
        output_file_path=output_path,
        id_column="ID",
        target_column="text",
    )
    logger.info(f"노이즈 text 교체 진행 완료 - 결과 파일: {output_path}")
else:
    # 추가하기 로직
    logger.info("디노이즈 text 추가 진행")
    try:
        # 모델 예측 결과에서 기존 텍스트 열을 제거하고 이름을 'text'로 변경
        test_with_predictions = (
            pd.read_csv(model_result).drop(columns=["text"]).rename(columns={"denoised_text": "text"})
        )
        train = pd.read_csv(train_path)

        # 단순히 train과 test_with_predictions를 아래로 쌓기
        train_combined = pd.concat([train, test_with_predictions], ignore_index=True)
        train_combined.to_csv(output_path, index=False)
        logger.info(f"디노이즈 text 추가 진행 완료 - 결과 파일: {output_path}")

    except Exception as e:
        logger.error(f"오류 발생: {e}")
