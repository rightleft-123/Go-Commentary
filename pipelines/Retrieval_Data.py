# -*- coding: utf-8 -*-
import multiprocessing.managers
import os
import re
import requests
import pandas as pd
import json
import threading
import pickle
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DataFrameLoader
import multiprocessing

configs = json.loads(open("./configs/config.json").read())

subscription_key = configs['bing_search_v7_subscription_key']
endpoint = configs['bing_search_v7_endpoint']


shared_dict = {}

def get_web_search_result(query):
    # Construct a request
    mkt = 'en-US'
    params = { 'q': query, 'mkt': mkt }
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

    # Call the API
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()

        # print("\nHeaders:\n")
        # print(response.headers)

        print("\nJSON Response:\n")
        # pprint(response.json())
    except Exception as ex:
        raise ex

    current_data = response.json()
    current_data1 = current_data.get('webPages', {}).get('value', [])

    df = {'news name': [],
        'news url': [],
        'news content': []}
    for i, data in enumerate(current_data1):
        print(data)
        df['news name'].append(data['name'])
        df['news url'].append(data['url'])
        df['news content'].append(data['snippet'])

    return pd.DataFrame(df)


def data_preparation():
    print('####################### Loading Database #######################')
    # Document preparation
    loader1 = PyPDFLoader("./data/External_Knowledge/围棋史话.pdf")
    loader2 = PyPDFLoader("./data/External_Knowledge/围棋历史对决.pdf")
    go_history_books = loader1.load()
    go_classic_matches = loader2.load()

    replace_list = [' ', '@', '〇', '®', '¥', '+', '-']
    for i in range(len(go_classic_matches)):
        for j in range(len(replace_list)):
            go_classic_matches[i].page_content = go_classic_matches[i].page_content.replace(replace_list[j], '')
        chinese_text = re.sub(r'[^\u4e00-\u9fff,\.]', '', go_classic_matches[i].page_content)
        go_classic_matches[i].page_content = ''.join(chinese_text)

    FIXED_QUERY = "2024年3月到6月的围棋比赛结果, 新闻"
    info = get_web_search_result(query=FIXED_QUERY)
    go_recent_matches = DataFrameLoader(info, page_content_column="news name").load()

    # Store in Vectorstore
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_for_go_history = Chroma.from_documents(documents=go_history_books, embedding=embedding_function)
    vectorstore_for_classic_match = Chroma.from_documents(documents=go_classic_matches, embedding=embedding_function)
    vectorstore_for_recent_match = Chroma.from_documents(documents=go_recent_matches, embedding=embedding_function)

    global shared_dict
    shared_dict['vectorstore1'] = vectorstore_for_go_history
    shared_dict['vectorstore2'] = vectorstore_for_classic_match
    shared_dict['vectorstore3'] = vectorstore_for_recent_match
    print('####################### Finish Database Loading #######################')


if __name__ == '__main__':
    ### RAG for commentary(Data preparation)
    data_preparation()

    try:
        while True:
            # 模拟一些长期运行的任务，比如等待一段时间
            print("Holding...")
            time.sleep(10)  # 等待10秒
    except KeyboardInterrupt:
        # 捕获Ctrl+C信号以优雅地退出循环
        print("\nRetrieval Data Stop")

    # vectorstore1, vectorstore2, vectorstore3 = data_preparation()
