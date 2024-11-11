import logging
import os

import pandas as pd


# 로거 설정
logger = logging.getLogger(__name__)


def compare_target_changes(
    new_file_path: str, original_file_path: str, prev_column: str = "target", new_column: str = "target"
) -> int:
    """
    라벨이 바뀐 후 파일과 바뀌기 전 파일을 비교하여 target이 변경된 행을 식별하고,
    변경된 행의 개수를 반환합니다.

    :param new_file_path: 라벨이 바뀐 후 CSV 파일 경로
    :type new_file_path: str
    :param original_file_path: 라벨이 바뀌기 전 CSV 파일 경로
    :type original_file_path: str
    :param prev_column: 이전 라벨 컬럼 이름 (기본값: "target")
    :type prev_column: str
    :param new_column: 새 라벨 컬럼 이름 (기본값: "target")
    :type new_column: str
    :return: 변경된 행의 개수
    :rtype: int
    """
    try:
        # 파일 불러오기
        logger.info(f"파일 불러오기: {new_file_path}와 {original_file_path}")
        df_new = pd.read_csv(new_file_path)
        df_original = pd.read_csv(original_file_path)

        # prev_target 열 추가하여 이전 타겟 정보 복사
        logger.info(f"이전 라벨 컬럼 '{prev_column}' 복사하여 'prev_target' 열 생성")
        df_new["prev_target"] = df_original[prev_column]

        # is_changed 열 추가 (target과 prev_target 비교하여 다르면 1, 같으면 0)
        logger.info(f"변경 여부 확인: '{new_column}'와 'prev_target' 비교하여 'is_changed' 열 생성")
        df_new["is_changed"] = (df_new[new_column] != df_new["prev_target"]).astype(int)

        # prev_target과 target 값이 다른 행 개수 계산
        diff_count = df_new["is_changed"].sum()

        # 결과 출력 및 개수 반환
        logger.info(f"변경된 행의 개수: {diff_count}")
        return diff_count

    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
    except Exception as e:
        logger.error(f"예상치 못한 오류가 발생했습니다: {e}")
        raise


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 디렉터리 설정
    data_dir = os.path.join("..", "data")

    # 파일 경로 설정
    new_file = os.path.join(data_dir, "denoised_results_relabel.csv")  # 라벨이 바뀐 후 파일 경로
    original_file = os.path.join(data_dir, "denoised_results_wo_error.csv")  # 라벨이 바뀌기 전 파일 경로

    # 함수 호출
    changed_rows_count = compare_target_changes(new_file, original_file)
