# 本地知识库方案（基于阿里云百炼local_rag项目）

项目特色：

1. 针对于阿里云local_rag的reranker模型需要依赖云端模型的问题，我们将其完全本地化，embedding和ranker模型都采用本地模型，可以完全离线部署（只有LLM是云端的，如果有条件可以换成本地部署的）。
2. 支持自定义角色，对不同的角色设定不同的提示词
3. 提供一个可供外界调用的知识库检索接口，供外界LLM程序调用获取知识库

请确保你的项目中包含以下文件结构：

```
local_rag/
├── File/
│   ├── Structured
│   ├── Unstructured
├── VectorStore/
│
```

### 环境配置

```
pip install -r requirements.txt
```

### 运行指南

1. 在终端中输入和阿里云百炼API-KEY相关的环境变量

```
export DASHSCOPE_API_KEY=
```

2. 启动知识库后端

```
uvicorn main:app --port 7866
```

然后访问`http://127.0.0.1:7866`，这是一个基于阿里云百炼的知识库管理系统

3. （可选）如需启动知识库检索后端，请启动下面flask程序，即可在5002端口启动一个知识库检索程序

```
python backend.py
```