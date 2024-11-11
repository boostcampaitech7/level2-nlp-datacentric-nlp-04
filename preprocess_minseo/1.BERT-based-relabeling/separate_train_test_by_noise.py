import logging
import os

import pandas as pd


# 로거 설정
logger = logging.getLogger(__name__)


def split_by_noise(
    input_file_path: str,
    output_train_path: str,
    output_test_path: str,
    noise_column: str = "is_noise",
) -> None:
    """
    노이즈 유무에 따라 데이터를 train과 test로 분리하고, 각각 CSV 파일로 저장합니다.

    :param input_file_path: 원본 CSV 파일 경로
    :type input_file_path: str
    :param output_train_path: 노이즈 데이터(train) 저장 경로
    :type output_train_path: str
    :param output_test_path: 비노이즈 데이터(test) 저장 경로
    :type output_test_path: str
    :param noise_column: 노이즈 여부를 나타내는 컬럼 이름 (기본값: "is_noise")
    :type noise_column: str
    :return: None
    """
    try:
        # 데이터 불러오기
        logger.info(f"파일 로드 중: {input_file_path}")
        df = pd.read_csv(input_file_path)

        # noise_column 값에 따라 데이터프레임 분리
        train_df = df[df[noise_column] == 1]
        test_df = df[df[noise_column] != 1]

        # 각 데이터프레임의 길이 로그 출력
        logger.info(f"Train 데이터프레임 길이: {len(train_df)}")
        logger.info(f"Test 데이터프레임 길이: {len(test_df)}")

        # 각각 CSV 파일로 저장
        train_df.to_csv(output_train_path, index=False)
        test_df.to_csv(output_test_path, index=False)
        logger.info(f"데이터가 각각 {output_train_path}와 {output_test_path}로 저장되었습니다.")

    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
    except Exception as e:
        logger.error(f"예상치 못한 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 디렉터리 설정
    data_dir = os.path.join("..", "data")

    # 파일 경로 설정
    input_file = os.path.join(data_dir, "5_base_noise_detected.csv")
    train_output_file = os.path.join(data_dir, "5_base_noise_detected_train.csv")
    test_output_file = os.path.join(data_dir, "5_base_noise_detected_test.csv")

    # 함수 호출
    split_by_noise(input_file, train_output_file, test_output_file)
