import cleanlab_noize_viz
from data_loader import save_uploaded_file_to_session
import data_overview
import noise_viz
import streamlit as st
import tokenize_viz


def select_page():
    page = st.sidebar.selectbox(
        "페이지 선택",
        [
            "단순 데이터 시각화",
            "노이즈 비율 시각화_단순 특수문자 비율",
            "클린랩 노이즈 비율 시각화",
            "토크나이징",
        ],
    )
    if page == "단순 데이터 시각화":
        if "data" in st.session_state:
            data_overview.show(st.session_state["data"])
        else:
            st.warning("세션에 저장된 데이터가 없습니다. 파일을 업로드해주세요.")

    elif page == "노이즈 비율 시각화_단순 특수문자 비율":
        if "data" in st.session_state:
            noise_viz.show(st.session_state["data"])
        else:
            st.warning("세션에 저장된 데이터가 없습니다. 파일을 업로드해주세요.")

    elif page == "클린랩 노이즈 비율 시각화":
        if "data" in st.session_state:
            cleanlab_noize_viz.show(st.session_state["data"])
            cleanlab_noize_viz.visualization(st.session_state["data"])

        else:
            st.warning("세션에 저장된 데이터가 없습니다. 파일을 업로드해주세요.")

    elif page == "토크나이징":
        if "data" in st.session_state:
            tokenize_viz.show(st.session_state["data"])


def main():
    st.title("데이터 분석 앱")

    uploaded_file = st.file_uploader("업로드할 파일:", type="csv")

    if uploaded_file is not None:
        save_uploaded_file_to_session(uploaded_file)

    # 데이터 출력
    if "data" in st.session_state:
        pass
    else:
        st.warning("세션에 저장된 데이터가 없습니다. 위에서 파일을 업로드해주세요.")
    select_page()


if __name__ == "__main__":
    main()
