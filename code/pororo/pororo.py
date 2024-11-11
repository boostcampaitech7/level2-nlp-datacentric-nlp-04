"""
Pororo 라이브러리를 사용하여 문장을 다른 언어로 번역 후 다시 원래 언어로 번역하는 Back Translation(역번역) 작업을 수행
readme를 먼저 읽고 pororo 라이브러를 설치해야함.
"""

import pandas as pd
from pororo import Pororo
from tqdm import tqdm


def back_translation(text, lang1, lang2, mt):
    trans_text = mt(text, src=lang1, tgt=lang2)
    backtrans_text = mt(trans_text, src=lang2, tgt=lang1)
    return backtrans_text


def apply_back_translation(
    data_path="../data/pororo/data.csv",
    output_path="../data/pororo/data_backtranslated.csv",
    proro_task="translation",
    proro_lang="multi",
    lang1="ko",
    lang2="en",
):
    df = pd.read_csv(data_path)

    mt = Pororo(task=proro_task, lang=proro_lang)

    tqdm.pandas()

    df["back_translation"] = df["text"].progress_apply(lambda x: back_translation(x, lang1=lang1, lang2=lang2, mt=mt))

    df.to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    apply_back_translation()
