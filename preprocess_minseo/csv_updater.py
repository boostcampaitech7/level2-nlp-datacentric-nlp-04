import logging
import os

import pandas as pd


# 모듈 전역 로거 설정
logger = logging.getLogger(__name__)


def update_csv_and_save(
    base_file_path: str, update_file_path: str, output_file_path: str, id_column: str, target_column: str
) -> None:
    """
    첫 번째 CSV 파일의 특정 칼럼 값을 두 번째 CSV 파일의 값으로 업데이트하고 저장합니다.

    :param base_file_path: 업데이트할 첫 번째 CSV 파일 경로 (예: train.csv).
    :type base_file_path: str
    :param update_file_path: 새로운 값이 있는 두 번째 CSV 파일 경로 (예: test_with_predictions.csv).
    :type update_file_path: str
    :param output_file_path: 업데이트된 CSV 파일을 저장할 경로.
    :type output_file_path: str
    :param id_column: 두 파일에서 행을 일치시킬 기준이 되는 ID 칼럼 이름.
    :type id_column: str
    :param target_column: 첫 번째 파일에서 업데이트할 칼럼 이름.
    :type target_column: str
    :return: None
    """
    try:
        # 파일 불러오기
        logger.info(f"파일 불러오기: {base_file_path}와 {update_file_path}")
        df_base = pd.read_csv(base_file_path)
        df_update = pd.read_csv(update_file_path)

        # 파일에 지정한 칼럼이 있는지 확인
        if id_column not in df_base.columns or id_column not in df_update.columns:
            logger.error(f"{id_column} 칼럼이 파일에 없습니다.")
            raise KeyError(f"{id_column} 칼럼이 파일에 없습니다.")
        if target_column not in df_base.columns:
            logger.error(f"{target_column} 칼럼이 기본 파일에 없습니다.")
            raise KeyError(f"{target_column} 칼럼이 기본 파일에 없습니다.")

        # 기준 칼럼에 따라 특정 칼럼 값을 업데이트
        logger.info(f"기준 칼럼 '{id_column}'에 따라 '{target_column}' 칼럼 값을 업데이트")
        df_base.loc[df_base[id_column].isin(df_update[id_column]), target_column] = df_base.loc[
            df_base[id_column].isin(df_update[id_column]), id_column
        ].map(df_update.set_index(id_column)[target_column])

        # 업데이트된 DataFrame을 새로운 파일로 저장
        df_base.to_csv(output_file_path, index=False)
        logger.info(f"업데이트된 파일이 {output_file_path}에 저장")

    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
    except KeyError as e:
        logger.error(f"칼럼 오류: {e}")
    except Exception as e:
        logger.error(f"예상치 못한 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # 파일 경로
    data_dir = os.path.join("..", "data")
    test_with_predictions_path = os.path.join(data_dir, "test_with_predictions.csv")
    train_path = os.path.join(data_dir, "train.csv")
    output_path = os.path.join(data_dir, "train_cleaning_AE.csv")

    # 사용 예시:
    update_csv_and_save(train_path, test_with_predictions_path, output_path, "ID", "target")
