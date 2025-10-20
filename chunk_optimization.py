"""
SEOチャットボット用チャンクサイズ最適化スクリプト
回答精度と回答スピードのバランスを検証
"""

import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import constants as ct

# 環境変数読み込み
load_dotenv()

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def adjust_string(text):
    """文字化け対策用の文字列調整"""
    if isinstance(text, str):
        return text.replace('\u3000', ' ').replace('\xa0', ' ')
    return text

def load_seo_documents():
    """SEO関連ドキュメントを読み込み"""
    docs_all = []
    data_path = "./data"
    
    # PDFファイルとDOCXファイルを読み込み
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        
        if filename.endswith('.pdf'):
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            docs_all.extend(docs)
            logger.info(f"PDF読み込み: {filename} ({len(docs)}ページ)")
            
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            docs_all.extend(docs)
            logger.info(f"DOCX読み込み: {filename} ({len(docs)}ファイル)")
    
    # 文字化け対策
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    logger.info(f"総ドキュメント数: {len(docs_all)}")
    return docs_all

def generate_seo_response(chunk_size, chunk_overlap_ratio, docs_all, query, k_search=5):
    """指定されたチャンクサイズでSEO回答を生成"""
    start_time = time.time()
    
    # チャンク分割
    chunk_overlap = int(chunk_size * chunk_overlap_ratio)
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    splitted_docs = text_splitter.split_documents(docs_all)
    
    # ベクターストア作成
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(splitted_docs, embedding=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": k_search})
    
    # LLM設定（現在のシステムと同じ）
    llm = ChatOpenAI(
        model_name=ct.MODEL,
        temperature=ct.TEMPERATURE,
        max_tokens=ct.MAX_TOKENS
    )
    
    # RAGチェーン作成
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # 回答生成
    result = chain.invoke({"query": query})
    
    # 処理時間計算
    processing_time = time.time() - start_time
    
    return {
        "answer": result["result"],
        "processing_time": processing_time,
        "chunk_count": len(splitted_docs),
        "source_docs": len(result["source_documents"]),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }

def run_chunk_optimization():
    """チャンクサイズ最適化の実行"""
    
    # SEOドキュメント読み込み
    print("📁 SEOドキュメントを読み込み中...")
    docs_all = load_seo_documents()
    
    # テスト用クエリ（SEO専用）
    test_queries = [
        "MEOで上位表示させるための効果的な施策を教えてください",
        "SEOにおけるキーワード選定の重要なポイントは何ですか",
        "Googleアップデートの影響とその対策について教えてください"
    ]
    
    # チャンクサイズの検証範囲
    chunk_configs = [
        {"size": 200, "overlap_ratio": 0.2},   # 小さめ（詳細重視）
        {"size": 400, "overlap_ratio": 0.2},   # 現在より小さめ
        {"size": 500, "overlap_ratio": 0.16},  # 現在の設定
        {"size": 600, "overlap_ratio": 0.15},  # やや大きめ
        {"size": 800, "overlap_ratio": 0.125}, # 大きめ（コンテキスト重視）
        {"size": 1000, "overlap_ratio": 0.1},  # 最大サイズ
    ]
    
    results = []
    
    print("\n🔍 チャンクサイズ最適化テスト開始...")
    print("=" * 80)
    
    for config in chunk_configs:
        chunk_size = config["size"]
        overlap_ratio = config["overlap_ratio"]
        
        print(f"\n📊 チャンクサイズ: {chunk_size}, オーバーラップ比: {overlap_ratio:.1%}")
        print("-" * 60)
        
        query_results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"  クエリ {i}: {query[:30]}...")
            
            try:
                result = generate_seo_response(
                    chunk_size=chunk_size,
                    chunk_overlap_ratio=overlap_ratio,
                    docs_all=docs_all,
                    query=query
                )
                
                query_results.append(result)
                print(f"    ⏱️  処理時間: {result['processing_time']:.2f}秒")
                print(f"    📄 チャンク数: {result['chunk_count']}")
                print(f"    🔍 参照文書: {result['source_docs']}")
                print(f"    💬 回答長: {len(result['answer'])}文字")
                
            except Exception as e:
                print(f"    ❌ エラー: {e}")
                continue
        
        # 平均値計算
        if query_results:
            avg_time = sum(r['processing_time'] for r in query_results) / len(query_results)
            avg_chunks = query_results[0]['chunk_count']  # チャンク数は同じ
            avg_answer_length = sum(len(r['answer']) for r in query_results) / len(query_results)
            
            summary = {
                "chunk_size": chunk_size,
                "overlap_ratio": overlap_ratio,
                "avg_processing_time": avg_time,
                "chunk_count": avg_chunks,
                "avg_answer_length": avg_answer_length,
                "speed_score": 10 - min(avg_time, 10),  # 10秒以内で高スコア
                "detail_score": min(avg_answer_length / 100, 10)  # 回答の詳細度
            }
            
            # バランススコア計算（速度50%, 詳細度50%）
            summary["balance_score"] = (summary["speed_score"] * 0.5 + 
                                      summary["detail_score"] * 0.5)
            
            results.append(summary)
            
            print(f"  📈 平均処理時間: {avg_time:.2f}秒")
            print(f"  📊 バランススコア: {summary['balance_score']:.2f}/10")
    
    # 結果まとめ
    print("\n" + "=" * 80)
    print("🏆 最適化結果サマリー")
    print("=" * 80)
    
    # スコア順にソート
    results.sort(key=lambda x: x["balance_score"], reverse=True)
    
    print(f"{'ランク':<4} {'チャンク':<6} {'重複比':<6} {'時間':<6} {'スコア':<6} {'推奨理由'}")
    print("-" * 60)
    
    for i, result in enumerate(results[:3], 1):
        reason = ""
        if i == 1:
            if result["avg_processing_time"] <= 8:
                reason = "最適バランス🥇"
            else:
                reason = "高品質重視🥇"
        elif i == 2:
            reason = "準最適🥈"
        else:
            reason = "代替案🥉"
            
        print(f"{i:<4} {result['chunk_size']:<6} {result['overlap_ratio']:.1%}{'':>1} "
              f"{result['avg_processing_time']:.1f}s{'':>1} {result['balance_score']:.1f}{'':>2} {reason}")
    
    # 推奨設定
    best_config = results[0]
    print(f"\n💡 推奨設定:")
    print(f"  chunk_size = {best_config['chunk_size']}")
    print(f"  chunk_overlap = {int(best_config['chunk_size'] * best_config['overlap_ratio'])}")
    print(f"  期待処理時間: {best_config['avg_processing_time']:.1f}秒")

if __name__ == "__main__":
    run_chunk_optimization()