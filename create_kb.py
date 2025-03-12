#####################################
######       创建知识库         #######
#####################################
import csv
import chardet
import gradio as gr
import os
import shutil
from llama_index.core import VectorStoreIndex,Settings,SimpleDirectoryReader
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.core.schema import TextNode, Document
from upload_file import *
DB_PATH = "VectorStore"
STRUCTURED_FILE_PATH = "File/Structured"
UNSTRUCTURED_FILE_PATH = "File/Unstructured"
TMP_NAME = "tmp_abcd"
EMBED_MODEL = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
)
# 若使用本地嵌入模型，请取消以下注释：
from langchain_community.embeddings import ModelScopeEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
embeddings = ModelScopeEmbeddings(model_id="modelscope/iic/nlp_gte_sentence-embedding_chinese-large")
EMBED_MODEL = LangchainEmbedding(embeddings)


# 设置嵌入模型
Settings.embed_model = EMBED_MODEL
Settings.chunk_size = 3000

# 刷新知识库
def refresh_knowledge_base():
    return os.listdir(DB_PATH)

# 创建非结构化向量数据库
def create_unstructured_db(db_name:str,label_name:list):
    print(f"知识库名称为：{db_name}，类目名称为：{label_name}")
    if label_name is None:
        gr.Info("没有选择类目")
    elif len(db_name) == 0:
        gr.Info("没有命名知识库")
    # 判断是否存在同名向量数据库
    elif db_name in os.listdir(DB_PATH):
        gr.Info("知识库已存在，请换个名字或删除原来知识库再创建")
    else:
        gr.Info("正在创建知识库，请等待知识库创建成功信息显示后前往RAG问答")
        documents = []
        for label in label_name:
            label_path = os.path.join(UNSTRUCTURED_FILE_PATH,label)
            documents.extend(SimpleDirectoryReader(label_path).load_data())
        index = VectorStoreIndex.from_documents(
            documents
        )

        db_path = os.path.join(DB_PATH,db_name)
        if not os.path.exists(db_path):
            os.mkdir(db_path)
            index.storage_context.persist(db_path)
        elif os.path.exists(db_path):
            pass
        gr.Info("知识库创建成功，可前往RAG问答进行提问")
    
# 创建结构化向量数据库
def create_structured_db(db_name:str,data_table:list):
    if not data_table:
        gr.Info("没有选择数据表")
        return
    if len(db_name) == 0:
        gr.Info("没有命名知识库")
        return
    if db_name in os.listdir(DB_PATH):
        gr.Info("知识库已存在，请换个名字或删除原来知识库再创建")
        return

    gr.Info("正在创建知识库，请等待知识库创建成功信息显示后前往RAG问答")

    nodes = []
    for label in data_table:

        file_path = os.path.join(STRUCTURED_FILE_PATH, label)
        for file_name in os.listdir(file_path):
            if file_name.endswith(".csv"):
                df = pd.read_csv(os.path.join(file_path, file_name), dtype=str)  # 确保所有列作为字符串处理
            elif file_name.endswith(".xlsx"):
                df = pd.read_excel(os.path.join(file_path, file_name), dtype=str)
            else:
                continue  # 只处理 csv/xlsx 文件
            
            # 逐行读取 CSV，并将每行作为一个独立的 chunk
            for idx, row in df.iterrows():
                row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
                node = Document(
                    text=row_text,
                    metadata = {"file_name": file_name},
                )
                nodes.append(node)

    # 构建索引
    index = VectorStoreIndex.from_documents(nodes)

    db_path = os.path.join(DB_PATH, db_name)
    os.makedirs(db_path, exist_ok=True)
    index.storage_context.persist(db_path)

    gr.Info("知识库创建成功，可前往 RAG 问答进行提问")

# 删除指定名称知识库
def delete_db(db_name:str):
    if db_name is not None:
        folder_path = os.path.join(DB_PATH, db_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            gr.Info(f"已成功删除{db_name}知识库")
            print(f"已成功删除{db_name}知识库")
        else:
            gr.Info(f"{db_name}知识库不存在")
            print(f"{db_name}知识库不存在")

# 实时更新知识库列表
def update_knowledge_base():
    return gr.update(choices=os.listdir(DB_PATH))

# 临时文件创建知识库
def create_tmp_kb(files):
    if not os.path.exists(os.path.join("File",TMP_NAME)):
        os.mkdir(os.path.join("File",TMP_NAME))
    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file,os.path.join("File",TMP_NAME,file_name))
    documents = SimpleDirectoryReader(os.path.join("File",TMP_NAME)).load_data()
    index = VectorStoreIndex.from_documents(
        documents
    )
    db_path = os.path.join(DB_PATH,TMP_NAME)
    if not os.path.exists(db_path):
        os.mkdir(db_path)
    index.storage_context.persist(db_path)

# 清除tmp文件夹下内容 
def clear_tmp():
    if os.path.exists(os.path.join("File",TMP_NAME)):
        shutil.rmtree(os.path.join("File",TMP_NAME))
    if os.path.exists(os.path.join(DB_PATH,TMP_NAME)):
        shutil.rmtree(os.path.join(DB_PATH,TMP_NAME))