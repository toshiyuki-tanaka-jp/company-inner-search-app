"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from typing import List
from dotenv import load_dotenv
import streamlit as st
import urllib.parse
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, Document
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct

client = OpenAI()


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def build_pdf_view_url(source_path: str) -> str:
    """GitHubのRaw URLをGoogle Docs Viewerで埋め込み/別タブ表示できるURLに変換"""
    owner  = st.secrets.get("GITHUB_OWNER")
    repo   = st.secrets.get("GITHUB_REPO")
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    rel = source_path.lstrip("./")
    raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rel}"
    return f"https://drive.google.com/viewer?embedded=1&url={urllib.parse.quote(raw, safe='')}"


def render_evidence(docs, title="情報源"):
    """関連ドキュメント（PDFはページ番号付き）を一覧表示"""
    if not docs:
        return
    st.markdown(f"### {title}")
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        page_str = f"（ページNo.{page+1}）" if isinstance(page, int) else ""
        if src.lower().endswith(".pdf"):
            url = build_pdf_view_url(src)
            # 先頭候補は淡い緑、そのほかは淡い青のバッジ風
            bg = "#e8f5e9" if i == 0 else "#e9f2ff"
            st.markdown(
                f"""
                <div style="background:{bg};padding:12px 16px;border-radius:12px;margin:8px 0;">
                  📄 <a href="{url}" target="_blank">{src}</a> {page_str}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background:#e9f2ff;padding:12px 16px;border-radius:12px;margin:8px 0;">
                  📄 {src}
                </div>
                """,
                unsafe_allow_html=True,
            )


def get_llm_response(prompt: str, docs: list, mode: str = "search") -> str:
    """LLMに要約・回答を生成させる"""
    context = "\n\n".join(
        f"Source: {d.metadata.get('source')} (page {d.metadata.get('page', 'N/A')})\n{d.page_content}"
        for d in docs
    )

    system_msg = "あなたは社内文書をもとに質問に答えるアシスタントです。"
    user_msg = f"質問:\n{prompt}\n\n参考情報:\n{context}"

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
    )

    return completion.choices[0].message.content.strip()


def get_llm_response_legacy(chat_message):
    """
    LLMからの回答取得（レガシー版・会話履歴対応）

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答（辞書形式）
    """
    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # モードによってLLMから回答を取得する用のプロンプトを変更
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # モードが「社内文書検索」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        # モードが「社内問い合わせ」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    # LLMから回答を取得する用のプロンプトテンプレートを作成
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # LLMへのリクエストとレスポンス取得
    llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    return llm_response


def get_relevant_docs(vectorstore, query, top_k=ct.RETRIEVAL_TOP_K):
    """
    ベクターストアから関連ドキュメントを取得
    
    Args:
        vectorstore: ベクターストア
        query: 検索クエリ
        top_k: 取得する文書数
        
    Returns:
        関連ドキュメントのリスト
    """
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        relevant_docs = retriever.get_relevant_documents(query)
        return relevant_docs
    except Exception as e:
        st.error(f"文書検索中にエラーが発生しました: {str(e)}")
        return []


def get_llm_response_simple(user_input, relevant_docs):
    """
    シンプルなLLM応答取得関数
    
    Args:
        user_input: ユーザー入力
        relevant_docs: 関連ドキュメント
        
    Returns:
        LLM応答
    """
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    
    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)
    
    # 文脈を作成
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # モードに応じたプロンプトを選択
    mode = st.session_state.get("mode", ct.ANSWER_MODE_1)
    if mode == ct.ANSWER_MODE_1:
        template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        template = ct.SYSTEM_PROMPT_INQUIRY
    
    # プロンプトテンプレートを作成
    prompt = ChatPromptTemplate.from_template(
        template + "\n\n入力: {input}"
    )
    
    # チェーンを作成して実行
    chain = prompt | llm
    response = chain.invoke({"input": user_input, "context": context})
    
    return {"answer": response.content}