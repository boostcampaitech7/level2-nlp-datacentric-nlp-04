import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


low_noise_threshold = 0.05
norm_noise_threshold = 0.2
high_noise_threshold = 0.3


# 단순 노이즈 비율 계산 : Context 전체 길이 중 특수문자의 비율
def calculate_noise_ratio(df):
    df["noise_ratio"] = df["text"].apply(
        lambda x: (len(re.findall(r"[^a-zA-Z0-9\sㄱ-ㅎㅏ-ㅣ가-힣]", x)) / len(x) if len(x) > 0 else 0)
    )
    return df


def show(df):
    st.header("노이즈 분석")

    df = calculate_noise_ratio(df)

    st.header("노이즈 비율에 따른 데이터 수 Bar plot")
    low_noise = df[df["noise_ratio"] <= low_noise_threshold].shape[0]
    norm_noise = df[(low_noise_threshold < df["noise_ratio"]) & (df["noise_ratio"] <= norm_noise_threshold)].shape[0]
    high_noise = df[df["noise_ratio"] > norm_noise_threshold].shape[0]

    categories = ["low_noise", "norm_noise", "high_noise"]
    values = [low_noise, norm_noise, high_noise]
    colors = ["blue", "green", "red"]

    fig, ax = plt.subplots()
    ax.bar(categories, values, color=colors)
    ax.set_xlabel("Noise")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # 노이즈 비율에 따른 데이터 보기
    st.header("노이즈 비율에 따른 데이터 보기")
    noise_category = st.selectbox("노이즈 비율 카테고리 선택:", ["None", "low_noise", "norm_noise", "high_noise"])
    target_value = st.selectbox(
        "target값 선택 :",
        ["None"] + list(df["target"].sort_values(ascending=True).unique()),
    )

    if noise_category == "low_noise":
        selected_data = df[df["noise_ratio"] <= low_noise_threshold].sort_values(by="noise_ratio")

    elif noise_category == "norm_noise":
        selected_data = df[
            (low_noise_threshold < df["noise_ratio"]) & (df["noise_ratio"] <= norm_noise_threshold)
        ].sort_values(by="noise_ratio")

    elif noise_category == "high_noise":
        selected_data = df[df["noise_ratio"] > high_noise_threshold].sort_values(by="noise_ratio")
    else:
        selected_data = df.sort_values(by="noise_ratio")

    if target_value != "None":
        selected_data = selected_data[selected_data["target"] == target_value].sort_values(by="noise_ratio")

    st.write(selected_data)

    # 라벨별 노이즈 비율 Bar plot
    st.header("라벨별 노이즈 비율 Bar plot")
    noise_data = (
        df.groupby("target")["noise_ratio"]
        .apply(
            lambda x: pd.Series(
                {
                    "low_noise": (x <= low_noise_threshold).sum(),
                    "norm_noise": ((low_noise_threshold < x) & (x <= norm_noise_threshold)).sum(),
                    "high_noise": (x > high_noise_threshold).sum(),
                }
            )
        )
        .unstack()
    )

    fig, ax = plt.subplots()
    noise_data.plot(kind="bar", ax=ax, color=["blue", "green", "red"])
    ax.set_xlabel("Target")
    ax.set_ylabel("Frequency")
    ax.legend(title="Noise Category", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=0)
    st.pyplot(fig)
