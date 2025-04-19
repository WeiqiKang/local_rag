from flask import request, Flask
app = Flask(__name__)

import os
from openai import OpenAI
from llama_index.core import StorageContext,load_index_from_storage,Settings
from create_kb import *
DB_PATH = "VectorStore"

# 若使用本地嵌入模型，请取消以下注释：
from langchain_community.embeddings import ModelScopeEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
embeddings = ModelScopeEmbeddings(model_id="modelscope/iic/nlp_gte_sentence-embedding_chinese-large")
EMBED_MODEL = LangchainEmbedding(embeddings)

# 设置嵌入模型
Settings.embed_model = EMBED_MODEL

# from rerankers import Reranker
# ranker = Reranker('mixedbread-ai/mxbai-rerank-large-v1', model_type='cross-encoder')

from pymilvus.model.reranker import BGERerankFunction
bge_rf = BGERerankFunction(
    model_name="BAAI/bge-reranker-v2-m3",  # Specify the model name. Defaults to `BAAI/bge-reranker-v2-m3`.
    device="cuda:0" # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)

@app.route('/get_knowledge', methods=["POST"])
def get_knowledge():
    '''
    chunk_cnt
    prompt
    similarity_threshold
    '''
    chunk_cnt = int(request.form["chunk_cnt"])
    prompt = request.form["prompt"]
    similarity_threshold = float(request.form["similarity_threshold"])
    db_name = request.form["db_name"]

    if chunk_cnt is None or prompt is None or similarity_threshold is None:
        return "Missing one or more params!", 200
    

    storage_context = StorageContext.from_defaults(
        persist_dir=os.path.join(DB_PATH, db_name)
    )

    index = load_index_from_storage(storage_context)
    print("index获取完成")


    retriever_engine = index.as_retriever(
        similarity_top_k=20,
    )

    # 获取chunk
    retrieve_chunk = retriever_engine.retrieve(prompt)
    print(f"原始chunk为：{retrieve_chunk}")

    docss = [x.text for x in retrieve_chunk]

    rerank_res = bge_rf(prompt, docss)

    chunk_text = ""
    chunk_show = ""
    for i in range(len(rerank_res)):
        chunk_text = chunk_text + f"## {i+1}:\n {rerank_res[i].text}\n"
        chunk_show = chunk_show + f"## {i+1}:\n {rerank_res[i].text}\nscore: {round(rerank_res[i].score,2)}\n"
    print(f"已获取chunk：{chunk_text}")

    return chunk_text

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)