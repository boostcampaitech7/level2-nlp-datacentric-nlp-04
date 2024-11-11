import pandas as pd
import streamlit as st


# 파일 업로더
uploaded_file = st.file_uploader("업로드할 파일:", type="csv")


# 데이터 로드 함수
@st.cache_data
def load_data(file=None):
    if file is not None:
        try:
            return pd.read_csv(file)
        except pd.errors.EmptyDataError:
            st.error("업로드된 파일이 비어있거나 올바른 CSV 형식이 아닙니다.")
            return None
    else:
        return None


def save_uploaded_file_to_session(uploaded_file):
    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
        st.success("파일이 성공적으로 업로드되었습니다.")

    if "uploaded_file" in st.session_state:
        data = load_data(st.session_state["uploaded_file"])
        if data is not None:
            st.session_state["data"] = data
            st.success("데이터가 성공적으로 로드되어 세션에 저장되었습니다.")
        else:
            st.error("데이터 로드에 실패했습니다. 파일을 다시 확인해주세요.")
    else:
        st.warning("파일을 업로드해주세요. 세션에 저장된 파일이 없습니다.")


# 데이터 출력
if "data" in st.session_state:
    pass
else:
    st.info("세션에 저장된 데이터가 없습니다. 파일을 업로드하고 데이터를 로드해주세요.")

# 세션 상태 표시
st.sidebar.header("세션 상태")
if "uploaded_file" in st.session_state:
    st.sidebar.success("파일 업로드 상태: 완료")
else:
    st.sidebar.warning("파일 업로드 상태: 대기 중")

if "data" in st.session_state:
    st.sidebar.success("데이터 로드 상태: 완료")
else:
    st.sidebar.warning("데이터 로드 상태: 대기 중")
