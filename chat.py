import os
from openai import OpenAI
from llama_index.core import StorageContext,load_index_from_storage,Settings
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from create_kb import *
DB_PATH = "VectorStore"
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

from rerankers import Reranker
ranker = Reranker('mixedbread-ai/mxbai-rerank-large-v1', model_type='cross-encoder')

# 针对不同角色设定的不同prompt模板
PROMPT_FOR_ROLE = {
    "数学解题专家": 
        '''请参考以下内容：{chunk_text}，接下来给你输入一道线性代数题目: {prompt}，请你在知识库中检索该题的详细解答过程和答案，原封不动输出出来，不要对公式、换行、转义符号进行任何修改,请逐字逐符号输出 LaTeX 公式，保持原始格式，不要将反斜杠 `\` 变成 `\\`，先输出题目的详细解答过程，再输出答案，从知识库中检索.''',
    "数学讲题老师":'''你现在是一位非常擅长用语言表达数学问题的老师。
你面前有一道线性代数的题目:{prompt}

请你完成以下要求：

不要使用任何数学符号或数学公式（包括LaTeX格式）。
请你对知识库的解答，使用类似口述讲解的方法将知识库的内容讲出来，给出讲稿。
其中的数学符号，如加号，减号，平方根等符号，请用中文描述出来，不要使用英文单词。
请你保证你的回答能被TTS（文本转语音）朗读出来，并且没有不可读的字符
不需要过于生活化的例子，请始终保持原数学题目所给出的情景或背景，避免过多的延伸或改变。
# 知识库
请记住以下材料，他们可能对回答问题有帮助。
{chunk_text}
''',
    "时光机客服": "你是时光机的智能客服，请参考{chunk_text}，用最简洁的语言概括回答{prompt}。若问题无关，礼貌拒绝。不回答‘上一句是什么’类问题。",

}

def get_model_response(multi_modal_input,history,model,role,temperature,max_tokens,history_round,db_name,similarity_threshold,chunk_cnt):
    
    # prompt = multi_modal_input['text']
    prompt = history[-1][0]
    tmp_files = multi_modal_input['files']
    if os.path.exists(os.path.join("File",TMP_NAME)):
        db_name = TMP_NAME
    else:
        if tmp_files:
            create_tmp_kb(tmp_files)
            db_name = TMP_NAME
    # 获取index
    print(f"prompt:{prompt},tmp_files:{tmp_files},db_name:{db_name}")
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=os.path.join(DB_PATH,db_name)
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

        try:
            if len(docss) > 1:
                results = ranker.rank(query=prompt, docs=docss)
                results = results.top_k(chunk_cnt)
                print(f"rerank成功，重排后的chunk为：{results}")
            else:
                results = results.document
        except:
            results = retrieve_chunk[:chunk_cnt]
            print(f"rerank失败，chunk为：{results}")
        chunk_text = ""
        chunk_show = ""

        for i in range(len(results)):
            # if results[i].score >= similarity_threshold:
            chunk_text = chunk_text + f"## {i+1}:\n {results[i].text}\n"
            chunk_show = chunk_show + f"## {i+1}:\n {results[i].text}\nscore: {round(results[i].score,2)}\n"
        print(f"已获取chunk：{chunk_text}")


        prompt_template = PROMPT_FOR_ROLE[role].format(chunk_text=chunk_text, prompt=prompt)

    except Exception as e:
        print(f"异常信息：{e}")
        prompt_template = prompt
        chunk_show = ""
    history[-1][-1] = ""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    system_message = {'role': 'system', 'content': 'You are a helpful assistant.'}
    messages = []
    history_round = min(len(history),history_round)
    for i in range(history_round):
        messages.append({'role': 'user', 'content': history[-history_round+i][0]})
        messages.append({'role': 'assistant', 'content': history[-history_round+i][1]})
    messages.append({'role': 'user', 'content': prompt_template})
    messages = [system_message] + messages
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
        )
    assistant_response = ""
    reasoning_content = ""  # 定义完整思考过程

    for chunk in completion:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        else:
            assistant_response += delta.content
            print(chunk.choices[0].delta.content, end='')
            history[-1][-1] = assistant_response
            yield history,chunk_show