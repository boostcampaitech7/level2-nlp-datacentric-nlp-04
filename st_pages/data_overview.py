# from data_loader import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def show(df):
    # 타겟값 기준으로 데이터 직접 보기
    st.header("라벨값 기준으로 데이터 살펴보기")
    target_value = st.selectbox("label :", df["target"].sort_values(ascending=True).unique())
    filtered_data = df[df["target"] == target_value]
    st.write(filtered_data)

    # 라벨링값에 해당하는 단순 데이터 수
    st.header("라벨값 기준 데이터 수 Bar plot")
    fig, ax = plt.subplots()
    sns.countplot(x="target", data=df, ax=ax)
    ax.set_xlabel("Label")
    ax.set_ylabel("# of data")
    st.pyplot(fig)
    for i in range(6):
        st.write(f"label {i} : {df[df['target'] == i].shape[0]}개")
    st.write(f"전체 데이터 수 : {df.shape[0]}개")
