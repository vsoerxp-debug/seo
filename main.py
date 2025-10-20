"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
# 「.env」ファイルから環境変数を読み込むための関数
from dotenv import load_dotenv
# ログ出力を行うためのモジュール
import logging
# streamlitアプリの表示を担当するモジュール
import streamlit as st
# （自作）画面表示以外の様々な関数が定義されているモジュール
import utils
# （自作）アプリ起動時に実行される初期化処理が記述された関数
from initialize import initialize
# （自作）画面表示系の関数が定義されているモジュール
import components as cn
# （自作）変数（定数）がまとめて定義・管理されているモジュール
import constants as ct


############################################################
# 2. 設定関連
############################################################
# ブラウザタブの表示文言を設定
st.set_page_config(
    page_title=ct.APP_NAME
)

# ログ出力を行うためのロガーの設定
logger = logging.getLogger(ct.LOGGER_NAME)


############################################################
# 3. 初期化処理
############################################################
try:
    # 初期化処理（「initialize.py」の「initialize」関数を実行）
    initialize()
except Exception as e:
    # エラーログの出力
    logger.error(f"{ct.INITIALIZE_ERROR_MESSAGE}\n{e}")
    # エラーメッセージの画面表示
    st.error(utils.build_error_message(ct.INITIALIZE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    # 後続の処理を中断
    st.stop()

# アプリ起動時のログファイルへの出力
if not "initialized" in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)


############################################################
# 4. 初期表示
############################################################
# タイトル表示
cn.display_app_title()

# モード表示
cn.display_select_mode()

# AIメッセージの初期表示
cn.display_initial_ai_message()


############################################################
# 5. 会話ログの表示
############################################################
try:
    # 会話ログの表示
    cn.display_conversation_log()
except Exception as e:
    # エラーログの出力
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    # エラーメッセージの画面表示
    st.error(utils.build_error_message(ct.CONVERSATION_LOG_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    # 後続の処理を中断
    st.stop()


############################################################
# 6. チャット入力の受け付け
############################################################
chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)


############################################################
# 7. チャット送信時の処理
############################################################
if chat_message:
    # ==========================================
    # 7-1. ユーザーメッセージの表示と内部修正処理
    # ==========================================
    # 入力されたメッセージをそのまま表示
    with st.chat_message("user"):
        st.markdown(chat_message)
    
    # 自動修正機能を無効化 - 入力をそのまま処理
    logger.info(f"入力メッセージ: '{chat_message}'")

    # ==========================================
    # 7-2. LLMからの回答取得
    # ==========================================
    # LLM回答生成の処理
    llm_response = None
    
    try:
        # スピナー表示で回答生成中を示す
        with st.spinner(ct.SPINNER_TEXT):
            # 画面読み込み時に作成したRetrieverを使い、Chainを実行
            llm_response = utils.get_llm_response(chat_message)
            
    except Exception as e:
        # エラーログの出力
        logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
        
        # エラーの種類に応じたメッセージを表示
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            error_msg = "⏰ 処理時間が制限を超えました。より簡潔な質問でお試しください。"
        elif "rate limit" in str(e).lower():
            error_msg = "🚫 APIの利用制限に達しました。しばらく待ってからお試しください。"
        else:
            error_msg = utils.build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE)
        
        # スピナー終了後にエラーメッセージを表示
        st.error(error_msg, icon=ct.ERROR_ICON)
        
        # エラー発生時は処理を完全停止
        st.stop()
    
    # エラーでない場合のみ回答表示に進む
    if llm_response is None:
        st.error("予期しないエラーが発生しました。", icon=ct.ERROR_ICON)
        st.stop()
    
    # ==========================================
    # 7-3. LLMからの回答表示
    # ==========================================
    with st.chat_message("assistant"):
        try:
            # SEO専用回答の表示
            content = cn.display_seo_llm_response(llm_response)
            
            # AIメッセージのログ出力
            logger.info({"message": content, "application_mode": "SEO"})
        except Exception as e:
            # エラーログの出力
            logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
            # エラーメッセージの画面表示
            st.error(utils.build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            # 後続の処理を中断
            st.stop()

    # ==========================================
    # 7-4. 会話ログへの追加
    # ==========================================
    # 表示用の会話ログにユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": chat_message})
    # 表示用の会話ログにAIメッセージを追加
    st.session_state.messages.append({"role": "assistant", "content": content})