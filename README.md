# files-for-milvus-help
*2025.07.23 更新：*

`client_search_diff.json` 是在相同的问题和搜索参数下， `dense_vs.client.search` 和 `hybrid_vs.client.search` 所得到的不一致结果（文件内容和 `diff.json` 一致）

`client_search_in_vs.ipynb` 是在相同的问题和搜索参数下， 使用 `dense_vs.client.search` 和 `hybrid_vs.client.search` 进行搜索的代码

------
`diff.json` 是 186 个问题在两个向量数据库上完整的检索结果和分数

`hotpotqa.py` 是我使用的将 HotpotQA 中的问题加载到 Document 对象中并将得到的 Document 对象存入向量数据库的脚本

`search_in_vector_store.ipynb` 中包括在两个向量数据库上的完整搜索过程 和 尝试使用「ICI House is now named after the company that provides what type of item?」问题的搜索结果复现问题的代码

`test_data_500.json` 是我使用的从 HotpotQA 开发集中随机抽取的 500 个问题

`top1_diff.json` 是 12 个 top1 不同问题在两个向量数据库上的检索结果和分数（只取了 top1）

`try_to_reproduce.ipynb` 中是尝试使用「ICI House is now named after the company that provides what type of item?」问题搜索结果复现问题的代码
