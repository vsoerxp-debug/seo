"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import time
import unicodedata
import re
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct


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


def fix_common_input_errors(text):
    """
    よくある入力変換エラーを修正する汎用処理
    
    Args:
        text: ユーザー入力テキスト
    
    Returns:
        修正されたテキストと修正内容の辞書
    """
    if not text:
        return text, {}
    
    original_text = text
    corrections = {}
    
    # よくある誤変換パターンを修正
    correction_patterns = {
        # MEO関連
        r'してもよい': 'すればよい',
        r'何をしてもよい': '何をすればよい',
        # SEO関連  
        r'上位表示してもよい': '上位表示すればよい',
        r'対策してもよい': '対策すればよい',
        # 一般的な誤変換
        r'方法はありますか': '方法はありますか',
        r'効果的ですか': '効果的ですか',
        # 語尾の修正
        r'ですか？': 'ですか',
        r'ましょうか？': 'ましょうか',
    }
    
    for pattern, replacement in correction_patterns.items():
        if re.search(pattern, text):
            old_text = text
            text = re.sub(pattern, replacement, text)
            if old_text != text:
                corrections[pattern] = replacement
    
    # Unicode正規化
    text = unicodedata.normalize('NFC', text)
    
    # 余分な空白の除去
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text, corrections


def normalize_input_text(text):
    """
    入力テキストの正規化処理（下位互換性のため残存）

    Args:
        text: ユーザー入力テキスト

    Returns:
        正規化されたテキスト
    """
    corrected_text, _ = fix_common_input_errors(text)
    return corrected_text


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_llm_response(chat_message):
    """
    LLMからの回答取得（回答時間計測付き）

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答（回答時間情報を含む）
    """
    start_time = time.time()
    
    # デバッグ情報の出力
    print(f"Debug: chat_message type: {type(chat_message)}, content: {chat_message}")
    print(f"Debug: chat_history type: {type(st.session_state.chat_history)}, length: {len(st.session_state.chat_history) if hasattr(st.session_state.chat_history, '__len__') else 'N/A'}")
    
    # LLMのオブジェクトを用意（品質とスピードのバランス最適化）
    llm = ChatOpenAI(
        model_name=ct.MODEL, 
        temperature=ct.TEMPERATURE,
        max_tokens=ct.MAX_TOKENS,
        request_timeout=22,  # スピード維持で品質確保のバランス調整
        streaming=False,  # ストリーミング無効で処理時間短縮
        max_retries=1  # リトライ回数削減でスピード重視
    )

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # SEO専用プロンプトを使用
    question_answer_template = ct.SYSTEM_PROMPT_SEO
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

    # LLMへのリクエストとレスポンス取得（エラーハンドリング強化）
    try:
        # 会話履歴の型チェック
        if not isinstance(st.session_state.chat_history, list):
            st.session_state.chat_history = []
        
        # 会話履歴の内容をチェック・クリーニング
        cleaned_history = []
        for item in st.session_state.chat_history:
            if hasattr(item, 'content') and isinstance(item.content, str):
                cleaned_history.append(item)
            elif isinstance(item, str):
                cleaned_history.append(HumanMessage(content=item))
        
        st.session_state.chat_history = cleaned_history
        
        llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
        
    except Exception as e:
        # チェーン実行エラーの場合、会話履歴をクリアして再試行
        st.session_state.chat_history = []
        try:
            llm_response = chain.invoke({"input": chat_message, "chat_history": []})
        except Exception as retry_error:
            raise retry_error
    
    # レスポンス形式を確認・修正
    if "answer" not in llm_response or not isinstance(llm_response["answer"], str):
        # answerが存在しないか文字列でない場合の対処
        if isinstance(llm_response, dict):
            # 他の可能性のあるキーを確認
            answer_text = (llm_response.get("answer") or 
                          llm_response.get("result") or 
                          llm_response.get("response") or 
                          str(llm_response))
        else:
            answer_text = str(llm_response)
        
        llm_response = {"answer": answer_text}
    
    # 回答が文字列であることを確実に保証
    if isinstance(llm_response.get("answer"), str):
        answer_content = llm_response["answer"]
    else:
        # 文字列でない場合は強制的に文字列に変換
        answer_content = str(llm_response.get("answer", "回答の取得に失敗しました。"))
    
    llm_response["answer"] = answer_content
    
    # LLMレスポンスを会話履歴に追加（文字列であることを保証）
    st.session_state.chat_history.extend([
        HumanMessage(content=str(chat_message)), 
        HumanMessage(content=answer_content)
    ])

    # 回答時間を計算・追加
    response_time = time.time() - start_time
    llm_response["response_time"] = response_time

    return llm_response