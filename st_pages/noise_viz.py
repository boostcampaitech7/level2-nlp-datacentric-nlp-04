import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data

df = load_data()

low_noise_threshold = 0.05
norm_noise_threshold = 0.2
high_noise_threshold = 0.3


def calculate_noise_ratio(df):
    df["noise_ratio"] = df["text"].apply(
        lambda x: (
            len(re.findall(r"[^a-zA-Z0-9\sㄱ-ㅎㅏ-ㅣ가-힣]", x)) / len(x)
            if len(x) > 0
            else 0
        )
    )
    return df


# def calculate_noise_ratio(df):
#     for i in range(len(df)):
#         noise_chars = re.findall(r"[^a-zA-Z0-9\sㄱ-ㅎㅏ-ㅣ가-힣]", df["text"][i])
#         noise_ratio = (
#             len(noise_chars) / len(df["text"][i]) if len(df["text"][i]) > 0 else 0
#         )
#         df["noise_ratio"] = df["text"].apply(
#             lambda x: (
#                 len(re.findall(r"[^a-zA-Z0-9\sㄱ-ㅎㅏ-ㅣ가-힣]", x)) / len(x)
#                 if len(x) > 0
#                 else 0
#             )
#         )
#     low_noise = df[df["noise_ratio"] <= low_noise_threshold].shape[0]
#     norm_noise = df[
#         (low_noise_threshold < df["noise_ratio"])
#         & (df["noise_ratio"] <= norm_noise_threshold)
#     ].shape[0]
#     high_noise = df[df["noise_ratio"] > high_noise_threshold].shape[0]


def show(df):
    st.header("노이즈 분석")

    df = calculate_noise_ratio(df)

    st.header("노이즈 비율에 따른 데이터 수 Bar plot")
    low_noise = df[df["noise_ratio"] <= low_noise_threshold].shape[0]
    norm_noise = df[
        (low_noise_threshold < df["noise_ratio"])
        & (df["noise_ratio"] <= norm_noise_threshold)
    ].shape[0]
    high_noise = df[df["noise_ratio"] > norm_noise_threshold].shape[0]

    categories = ["low_noise", "norm_noise", "high_noise"]
    values = [low_noise, norm_noise, high_noise]
    colors = ["blue", "green", "red"]

    fig, ax = plt.subplots()
    ax.bar(categories, values, color=colors)
    ax.set_xlabel("Noise : ")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # 노이즈 비율에 따른 데이터 보기
    st.header("노이즈 비율에 따른 데이터 보기")
    noise_category = st.selectbox(
        "노이즈 비율 카테고리 선택:", ["None", "low_noise", "norm_noise", "high_noise"]
    )
    target_value = st.selectbox(
        "target값 선택 :", ["None"] + list(df["target"].unique())
    )

    if noise_category == "low_noise":
        selected_data = df[df["noise_ratio"] <= low_noise_threshold]
    elif noise_category == "norm_noise":
        selected_data = df[
            (low_noise_threshold < df["noise_ratio"])
            & (df["noise_ratio"] <= norm_noise_threshold)
        ]
    elif noise_category == "high_noise":
        selected_data = df[df["noise_ratio"] > high_noise_threshold]
    else:
        selected_data = df

    if target_value != "None":
        selected_data = selected_data[selected_data["target"] == target_value]

    st.write(selected_data)

    # 라벨별 노이즈 비율 Bar plot
    st.header("라벨별 노이즈 비율 Bar plot")
    noise_data = (
        df.groupby("target")["noise_ratio"]
        .apply(
            lambda x: pd.Series(
                {
                    "low_noise": (x <= low_noise_threshold).sum(),
                    "norm_noise": (
                        (low_noise_threshold < x) & (x <= norm_noise_threshold)
                    ).sum(),
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
    ax.legend(title="Noise Category")
    st.pyplot(fig)
