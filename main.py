import streamlit as st
from components import chat_component
from constants import RETRIEVAL_TOP_K
from initialize import initialize_vectorstore
from utils import get_relevant_docs

# ページ設定
st.set_page_config(
    page_title="社内情報特化型生成AI検索アプリ",
    layout="wide",
)

# --- サイドバー ---
with st.sidebar:
    st.header("利用目的")
    purpose = st.radio("",
        ["社内文書検索", "社内問い合わせ"],
        index=0,
    )

    if purpose == "社内文書検索":
        st.markdown("##### 【社内文書検索】を選択した場合")
        st.info("入力内容と関連性が高い社内文書のありかを検索できます。")
        st.markdown("**【入力例】** 社員の育成方針に関するMTGの議事録")
    else:
        st.markdown("##### 【社内問い合わせ】を選択した場合")
        st.info("質問・要望に対して、社内文書の情報をもとに回答を得られます。")
        st.markdown("**【入力例】** 人事部に所属している従業員情報を一覧化して")

# --- メインエリア ---
st.title("社内情報特化型生成AI検索アプリ")

st.success(
    "こんにちは。私は社内文書の情報をもとに回答する生成AIチャットボットです。"
    "サイドバーで利用目的を選択し、画面下部のチャット欄からメッセージを送信してください。"
)

st.warning("⚠️ 具体的に入力したほうが期待通りの回答を得やすいです。")

# セッションステートの初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = purpose

# モードの更新
st.session_state.mode = purpose

# --- RAG処理の初期化 ---
vectorstore = initialize_vectorstore()

# --- チャット履歴の表示 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- チャット欄 ---
user_input = st.chat_input("こちらからメッセージを送信してください。")

if user_input:
    # ユーザーメッセージを表示
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("社内文書を検索中..."):
        relevant_docs = get_relevant_docs(vectorstore, user_input, top_k=RETRIEVAL_TOP_K)
        response = chat_component(user_input, relevant_docs)

    # AIメッセージを表示
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})