"""
hotpotqa - 处理 HotpotQA 数据集的验证集，将其存入向量数据库

Author - hahahajw
Date - 2025-06-04 
"""
from loguru import logger as log
from typing import (
    List,
    Literal,
    Dict,
    Any,
    Tuple
)
import os
import json
import random
from uuid import uuid4
from pymilvus.client.types import LoadState

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_openai import OpenAIEmbeddings

from modules.Index import Index


def json_to_documents(file_path: str,
                      sample_size: int,
                      output_file_path: str) -> List[Document]:
    """
    从 HotpotQA 中加载原始数据，随机抽取 sample_size 个将其存入 Document 对象中
    同时将选中的样本重新写入一个 JSON 文件中，以便后续试验使用
    Args:
        file_path: 存放 HotpotQA 原始数据的文件位置
        sample_size: 要抽取多少样本来构成向量数据集
        output_file_path: 抽样样本的存放位置

    Returns:
        List[Document]: 存放着样本数据的 Document 对象，样本中的一段被加载到一个 Document 对象中
    """
    # 加载原始文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data: List[Dict[str, Any]] = json.load(f)
        log.info(f"成功加载HotpotQA数据集，共{len(raw_data)}个样本")
    except Exception as e:
        log.error(f"加载HotpotQA数据集失败: {e}")
        raise

    # 从中抽取 sample_size 和样本
    sample_data = random.sample(raw_data, sample_size) if sample_size < len(raw_data) else raw_data
    # 将抽取的样本存入 hotpotqa_{sample_size}.json 文件中，以便后续试验
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False)
    log.info(f"从数据集中随机抽取了{len(sample_data)}个样本，写入 {output_file_path} 文件中")

    # 将数据存入 Document 对象中
    documents = []
    for sample in sample_data:
        # 获取样本的元数据（先这么写）
        sample_id = sample.get('_id', 'unknown_id')
        question = sample.get('question', '')
        context = sample.get('context', ['', []])
        for title, sentences in context:
            # 把句子合并为一个完整的段落（就像 Index 时 PDF 中的一页）
            paragraph = ''.join(sentences)
            doc = Document(
                page_content=paragraph,
                metadata={
                    'hotpotqa_id': sample_id,
                    'question': question,
                    'title': title,
                    'dataset': 'HotpotQA',
                    'source': f'HotpotQA_{title}_{sample_id}',  # 为了应付文档格式化的情况
                    'page': 0
                }
            )
            documents.append(doc)

    return documents


def get_hotpotqa_vector_store(embed_model: OpenAIEmbeddings,
                              file_path: str,
                              sample_size: int,
                              output_file_path: str,
                              vector_store_name: str,
                              chunk_size: int,
                              chunk_overlap: int,
                              is_hybrid=Literal['hybrid', 'dense', 'sparse']) -> Tuple[Milvus, str]:
    """
    将数据存入向量数据库中
    Args:
        embed_model: 要使用的嵌入模型
        file_path: 存放 HotpotQA 原始数据的文件位置
        sample_size: 抽取的样本数量
        output_file_path: 抽样出的样本的存放位置
        vector_store_name: 这次创建的向量数据库名称
        chunk_size: chunk 中文本块的大致长度
        chunk_overlap: chunk 间重叠字符的数量
        is_hybrid: 要创建的向量数据库类型
                   hybrid: 可以进行混合搜索的 Milvus 数据库实例
                   dense: 只能进行密集搜索的 Milvus 数据库实例
                   sparse: 只能进行稀疏搜索的 Milvus 数据库实例

    Returns:
        保存着 HotpotQA 数据集的 Index 实例
    """
    # 看要创建的数据库是否已经存在，存在则直接返回向量数据库
    # 如果已经存在了一个重名的数据库但没被加载到内存中，则下面的代码会将其加载到内存中
    # 如果已经存在了一个重名的数据库并且已经被加载到内存中，则下面的代码只是为对应的向量数据库创建了一个新的引用
    if is_hybrid == 'hybrid':
        vector_store = Milvus(
            collection_name=vector_store_name,
            embedding_function=embed_model,
            # 定义一个可以进行混合搜索的 Milvus 实例
            builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
            vector_field=['dense', 'sparse']
        )
    elif is_hybrid == 'dense':
        vector_store = Milvus(
            collection_name=vector_store_name,
            embedding_function=embed_model,
        )
    else:
        vector_store = Milvus(
            collection_name=vector_store_name,
            embedding_function=None,
            builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
            vector_field='sparse'
        )
    client = vector_store.client
    state = client.get_load_state(collection_name=vector_store_name)
    # 当前的向量数据库还没有被创建过
    if state['state'] == LoadState.NotExist:
        pass
    else:
        log.info(f'{vector_store_name} 向量数据库已存在，成功加载到内存中')
        return vector_store, vector_store_name

    # 为向数据库中添加数据做准备
    # 把 document 切分为 chunk
    all_documents = json_to_documents(
        file_path=file_path,
        sample_size=sample_size,
        output_file_path=output_file_path
    )
    all_chunks = Index.documents_to_chunks(
        docs=all_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # 向数据库中添加数据
    for i in range(0, len(all_chunks), 10):
        cur_chunks = all_chunks[i: i + 10]
        vector_store.add_documents(
            documents=cur_chunks,
            ids=[str(uuid4()) for _ in range(len(cur_chunks))]
        )
    log.info(f'HotpotQA 数据集处理完成，共添加了 {len(all_chunks)} 个 chunk 到向量数据库 {vector_store_name} 中')

    return vector_store, vector_store_name


if __name__ == '__main__':
    # 1. 您可以使用下面的函数替换 Index.documents_to_chunks
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # def documents_to_chunks(docs: List[Document],
    #                         chunk_size: int,
    #                         chunk_overlap: int) -> List[Document]:
    #     """
    #     将 context 较长的 Document 划分为较小的 chunk，可以直接存入向量数据库

    #     Args:
    #         docs: contex 较长的 Document
    #         chunk_size: chunk 中文本块的大致长度
    #         chunk_overlap: chunk 间重叠字符的数量

    #     Returns:
    #         List[Document]: 最终的 chunk 列表
    #     """
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=chunk_size,
    #         chunk_overlap=chunk_overlap,
    #         add_start_index=True,
    #         length_function=len,  # 计算文本长度的方式，最后划分出的 chunk 不严谨满足这个长度。因为不是以长度为标准进行的划分
    #         separators=[
    #             "\n\n",
    #             "\n",
    #             "。",
    #             "，"
    #             ".",
    #             ",",
    #             " ",
    #             ""
    #         ]  # 划分文本所使用的分隔符。添加了中文的句号和逗号，以便更好的划分中文文本。此外，分隔符的属性也是有意义的
    #     )

    #     return text_splitter.split_documents(docs)

    # 2. 替换后，您应该只需要修改 get_hotpotqa_vector_store 函数中的 file_path 参数，就可以运行下面的代码

    # 3. 当成功创建向量数据库后，您就可以使用对应的向量数据库的名称利用 Milvus() 函数加载对应的向量数据库了


    import os
    from dotenv import load_dotenv

    from langchain_openai import OpenAIEmbeddings
    

    load_dotenv()

    embed_model = OpenAIEmbeddings(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model="text-embedding-v3",
        dimensions=1024,
        check_embedding_ctx_length=False
    )

    v_s, v_s_name = get_hotpotqa_vector_store(
        embed_model=embed_model,
        file_path='../data/hotpotqa_10.json',
        sample_size=500,
        output_file_path='hotpotqa/qa_100.json',
        vector_store_name='hotpotqa_test',
        chunk_size=500,
        chunk_overlap=10
    )


    
