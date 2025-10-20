"""
SEOチャットボット用チャンクサイズ最適化スクリプト（高速版）
"""

import os
import time
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import constants as ct

# 環境変数読み込み
load_dotenv()

def quick_chunk_test():
    """高速チャンクサイズテスト"""
    
    # サンプルドキュメント読み込み（PDFファイル3つのみ）
    sample_files = [
        "2025-2026_SEO1-3_local_seo_meo.pdf",
        "2025-2026_SEO2-1_content_assets.pdf", 
        "2025-2026_SEO3-1_keyword_research.pdf"
    ]
    
    docs_all = []
    for filename in sample_files:
        file_path = f"./data/{filename}"
        if os.path.exists(file_path):
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            docs_all.extend(docs)
            print(f"読み込み: {filename} ({len(docs)}ページ)")
    
    print(f"サンプルドキュメント数: {len(docs_all)}")
    
    # テスト用クエリ（1つのみ）
    query = "MEOで上位表示させるための効果的な施策を教えてください"
    
    # チャンクサイズ候補
    chunk_sizes = [300, 500, 700, 1000]
    
    results = []
    
    print("\n🔍 高速チャンクサイズテスト開始...")
    print("=" * 60)
    
    for chunk_size in chunk_sizes:
        print(f"\n📊 チャンクサイズ: {chunk_size}")
        
        start_time = time.time()
        
        # チャンク分割
        chunk_overlap = int(chunk_size * 0.15)
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n"
        )
        splitted_docs = text_splitter.split_documents(docs_all)
        
        # ベクターストア作成（一時的）
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(splitted_docs, embedding=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # LLM設定
        llm = ChatOpenAI(
            model_name=ct.MODEL,
            temperature=ct.TEMPERATURE,
            max_tokens=ct.MAX_TOKENS
        )
        
        # RAGチェーン
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        # 回答生成
        result = chain.invoke({"query": query})
        
        processing_time = time.time() - start_time
        
        # 結果記録
        chunk_result = {
            "chunk_size": chunk_size,
            "chunk_count": len(splitted_docs),
            "processing_time": processing_time,
            "answer_length": len(result["result"]),
            "answer": result["result"][:200] + "..." if len(result["result"]) > 200 else result["result"]
        }
        
        results.append(chunk_result)
        
        print(f"  ⏱️  処理時間: {processing_time:.2f}秒")
        print(f"  📄 チャンク数: {len(splitted_docs)}")
        print(f"  💬 回答長: {len(result['result'])}文字")
        print(f"  📝 回答例: {chunk_result['answer']}")
    
    # 結果まとめ
    print("\n" + "=" * 60)
    print("🏆 最適化結果サマリー")
    print("=" * 60)
    
    print(f"{'チャンク':<8} {'時間':<8} {'チャンク数':<10} {'回答長':<8} {'評価'}")
    print("-" * 50)
    
    for result in results:
        # 簡易評価
        if result["processing_time"] <= 8 and result["answer_length"] >= 300:
            evaluation = "🥇 最適"
        elif result["processing_time"] <= 10:
            evaluation = "🥈 良好"
        else:
            evaluation = "🥉 改善要"
            
        print(f"{result['chunk_size']:<8} {result['processing_time']:.1f}s{'':>3} "
              f"{result['chunk_count']:<10} {result['answer_length']:<8} {evaluation}")
    
    # 推奨設定
    best = min(results, key=lambda x: x["processing_time"] if x["answer_length"] >= 300 else float('inf'))
    
    print(f"\n💡 推奨設定:")
    print(f"  chunk_size = {best['chunk_size']}")
    print(f"  chunk_overlap = {int(best['chunk_size'] * 0.15)}")
    print(f"  予想処理時間: {best['processing_time']:.1f}秒")
    
    return results

if __name__ == "__main__":
    quick_chunk_test()