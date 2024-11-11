import streamlit as st
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")


def show(df):
    st.header("토크나이징 결과(input_ids값)")
    tokenized_df = df["text"].apply(lambda x: tokenizer(x))
    st.write(tokenized_df)

    st.header("토크나이징 결과(실제 텍스트)")
    text_tokenized_df = tokenized_df.apply(
        lambda x: " ".join([f"<{token}>" for token in tokenizer.convert_ids_to_tokens(x["input_ids"])])
    )
    text_tokenized_df = text_tokenized_df.apply(lambda x: " ".join(x))
    text_tokenized_df = text_tokenized_df.to_frame(name="tokenized_text")
    text_tokenized_df["origin_text"] = df["text"]

    st.write(text_tokenized_df)
