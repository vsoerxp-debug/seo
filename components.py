"""
このファイルは、画面表示に特化した関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import streamlit as st
import utils
import constants as ct


############################################################
# 関数定義
############################################################

def display_app_title():
    """
    タイトル表示
    """
    st.markdown(f"## {ct.APP_NAME}")


def display_select_mode():
    """
    SEO専用モードの設定（モード選択は削除）
    """
    # SEO専用モードに固定
    st.session_state.mode = "SEO"


def display_initial_ai_message():
    """
    SEO専用チャットボットの初期表示
    """
    with st.chat_message("assistant"):
        st.markdown("こんにちは。私はSEO専門のチャットボットです。お客様のSEOに関するご質問にお答えします。")

        # SEO機能説明
        st.info("SEOに関するあらゆるご質問にお答えします。ページ作成指針、文字数、インデックスエラー対応など、お気軽にお聞かせください。")
        
        st.code("【入力例】\n・サブページの推奨文字数はどのくらいですか？\n・インデックスエラーの対応方法は？\n・SEOに効果的なページ構造について", wrap_lines=True, language=None)


def display_conversation_log():
    """
    会話ログの一覧表示
    """
    # 会話ログのループ処理
    for message in st.session_state.messages:
        # 「message」辞書の中の「role」キーには「user」か「assistant」が入っている
        with st.chat_message(message["role"]):

            # ユーザー入力値の場合、そのままテキストを表示するだけ
            if message["role"] == "user":
                st.markdown(message["content"])
            
            # SEOチャットボットからの回答の場合
            else:
                # 見出しサイズを調整してから表示
                formatted_answer = message["content"]["answer"]
                formatted_answer = formatted_answer.replace("# ", "#### ")  # h1 → h4
                formatted_answer = formatted_answer.replace("## ", "#### ")  # h2 → h4
                formatted_answer = formatted_answer.replace("### ", "#### ")  # h3 → h4
                st.markdown(formatted_answer)
                
                # 回答時間を表示（履歴でも確認可能）
                if "response_time" in message["content"]:
                    st.caption(f"⏱️ 回答生成時間: {message['content']['response_time']:.2f}秒")


def display_seo_llm_response(llm_response):
    """
    SEOチャットボットのLLMレスポンスを表示（回答時間表示付き）

    Args:
        llm_response: LLMからの回答

    Returns:
        LLMからの回答を画面表示用に整形した辞書データ
    """
    # 見出しサイズを調整（大きすぎる見出しを小さくする）
    formatted_answer = llm_response["answer"]
    formatted_answer = formatted_answer.replace("# ", "#### ")  # h1 → h4
    formatted_answer = formatted_answer.replace("## ", "#### ")  # h2 → h4
    formatted_answer = formatted_answer.replace("### ", "#### ")  # h3 → h4
    
    # Markdownで表示（見出しサイズ調整済み）
    st.markdown(formatted_answer)
    
    # 回答時間を表示（パフォーマンス確認用）
    if "response_time" in llm_response:
        response_time = llm_response["response_time"]
        if response_time <= 5:
            st.success(f"⚡ 高速回答: {response_time:.2f}秒", icon="🚀")
        elif response_time <= 10:
            st.info(f"⏱️ 回答時間: {response_time:.2f}秒", icon="⏱️")
        else:
            st.warning(f"⚠️ 処理時間が長めです: {response_time:.2f}秒", icon="⚠️")

    # 表示用の会話ログに格納するためのデータを用意
    content = {}
    content["answer"] = llm_response["answer"]
    if "response_time" in llm_response:
        content["response_time"] = llm_response["response_time"]

    return content