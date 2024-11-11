import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import streamlit as st


# 모든 경고 무시
warnings.filterwarnings("ignore")


def str_to_np_array(s):
    numbers = s.strip("[]").split()
    return np.array([float(num) for num in numbers])


low_noise_threshold = 0.05
norm_noise_threshold = 0.2
high_noise_threshold = 0.3


def calculate_noise_ratio(df):
    df["noise_ratio"] = df["text"].apply(
        lambda x: (len(re.findall(r"[^a-zA-Z0-9\sㄱ-ㅎㅏ-ㅣ가-힣]", x)) / len(x) if len(x) > 0 else 0)
    )
    return df


def visualization(df):
    st.title("t-SNE 시각화")

    df["prob"] = df["prob"].apply(str_to_np_array)

    # t-SNE 파라미터 설정
    perplexity = st.slider("Perplexity", 5, 50, 30)
    early_exaggeration = st.slider("Early Exaggeration", 12, 30, 20)
    n_iter = st.slider("Number of Iterations", 1000, 10000, 1000)

    # t-SNE 실행
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate="auto",
        n_iter=n_iter,
    )

    coords = tsne.fit_transform(np.stack(df["prob"].values))

    # 결과를 DataFrame으로 변환
    df_plot = pd.DataFrame(data=coords, columns=["TSNE1", "TSNE2"])
    df_plot["target"] = df["target"]
    df_plot["new_label"] = df["new_label"]

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(12, 8))

    # 각 target에 대해 다른 색상으로 점 찍기
    for target in df_plot["target"].unique():
        mask = df_plot["target"] == target
        ax.scatter(
            df_plot.loc[mask, "TSNE1"],
            df_plot.loc[mask, "TSNE2"],
            label=f"Target {target}",
            alpha=0.6,
        )

    # 라벨이 변경된 포인트 강조
    changed_mask = df_plot["target"] != df_plot["new_label"]
    ax.scatter(
        df_plot.loc[changed_mask, "TSNE1"],
        df_plot.loc[changed_mask, "TSNE2"],
        color="red",
        s=100,
        facecolors="none",
        edgecolors="red",
        linewidth=2,
        label="Changed Label",
    )

    plt.title("t-SNE Visualization of Targets and Changed Labels")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(True)

    st.pyplot(fig)

    # 변경된 라벨 수 출력
    st.write(f"Number of changed labels: {changed_mask.sum()}")
    st.write("\nChanged labels details:")
    st.write(df[changed_mask][["target", "new_label"]].value_counts().sort_index())


def show(df):
    st.header("클린랩 리라벨링 노이즈 분석")

    if "new_label" not in df.columns:
        st.warning("데이터프레임에 'new_label' 열이 없습니다.")
        return

    # 업로드된 파일을 DataFrame으로 변환
    df = calculate_noise_ratio(df)

    # 새롭게 라벨링된 데이터만 선택
    new_labeled_df = df[df["target"] != df["new_label"]]

    st.header("새롭게 라벨링된 데이터의 노이즈 비율")
    low_noise = new_labeled_df[new_labeled_df["noise_ratio"] <= low_noise_threshold].shape[0]
    norm_noise = new_labeled_df[
        (low_noise_threshold < new_labeled_df["noise_ratio"]) & (new_labeled_df["noise_ratio"] <= norm_noise_threshold)
    ].shape[0]
    high_noise = new_labeled_df[new_labeled_df["noise_ratio"] > norm_noise_threshold].shape[0]

    categories = ["low_noise", "norm_noise", "high_noise"]
    values = [low_noise, norm_noise, high_noise]
    colors = ["blue", "green", "red"]

    fig, ax = plt.subplots()
    ax.bar(categories, values, color=colors)
    ax.set_xlabel("Noise Levels")
    ax.set_ylabel("Frequency")
    ax.set_title("Noise Levels for Newly Labeled Data")
    st.pyplot(fig)

    # 새롭게 라벨링된 데이터의 노이즈 비율에 따른 데이터 보기
    st.header("새롭게 라벨링된 데이터의 노이즈 비율에 따른 데이터 보기")
    noise_category = st.selectbox("노이즈 비율 카테고리 선택:", ["None", "low_noise", "norm_noise", "high_noise"])

    if noise_category == "low_noise":
        selected_data = new_labeled_df[new_labeled_df["noise_ratio"] <= low_noise_threshold].sort_values(
            by="noise_ratio"
        )

    elif noise_category == "norm_noise":
        selected_data = new_labeled_df[
            (low_noise_threshold < new_labeled_df["noise_ratio"])
            & (new_labeled_df["noise_ratio"] <= norm_noise_threshold)
        ].sort_values(by="noise_ratio")

    elif noise_category == "high_noise":
        selected_data = new_labeled_df[new_labeled_df["noise_ratio"] > high_noise_threshold].sort_values(
            by="noise_ratio"
        )

    else:
        selected_data = new_labeled_df.sort_values(by="noise_ratio")

    # 'prob' 열을 제거한 후 표시
    if "prob" in selected_data.columns:
        selected_data = selected_data.drop(columns=["prob"])

    st.dataframe(selected_data, width=1200, height=500)  # 너비 1200픽셀, 높이 500픽셀
