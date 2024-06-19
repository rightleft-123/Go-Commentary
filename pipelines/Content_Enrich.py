# -*- coding: utf-8 -*-
import json
import os
import re
import requests
import pandas as pd
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DataFrameLoader

# from Retrieval_Data import shared_dict

configs = json.loads(open("../configs/config.json").read())

subscription_key = configs['bing_search_v7_subscription_key']
endpoint = configs['bing_search_v7_endpoint']
os.environ["OPENAI_API_KEY"] = configs['openai_api_key']
os.environ["OPENAI_API_BASE"] = configs['openai_api_url']
commentary_script_json = []


def get_web_search_result(query):
    # Construct a request
    mkt = 'en-US'
    params = {'q': query, 'mkt': mkt, 'count': 2}
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

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
    loader1 = PyPDFLoader("../data/External_Knowledge/go_history.pdf")
    loader2 = PyPDFLoader("../data/External_Knowledge/go_history_match.pdf")
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

    # shared_dict['vectorstore1'] = pickle.dumps(vectorstore_for_go_history)
    # shared_dict['vectorstore2'] = pickle.dumps(vectorstore_for_classic_match)
    # shared_dict['vectorstore3'] = pickle.dumps(vectorstore_for_recent_match)

    print('####################### Finish Database Loading #######################')
    return vectorstore_for_go_history, vectorstore_for_classic_match, vectorstore_for_recent_match


class RouterQuery(BaseModel):
    datasource: Literal["no_need_for_knowledge", "recent_go_match_news", "classic_go_match"] = Field(
        description="Given a description of the Go board question choose which datasource would be most relevant for answering their question"
    )


def routing_database(description):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_llm = llm.with_structured_output(RouterQuery)

    # Prompt for routing
    system = """You are an expert at routing a user question to the appropriate data source.
    Based on the programming language the question is referring to, route it to the relevant data source."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{description}")
        ]
    )

    router = prompt | structured_llm
    routing_result = router.invoke({"description": description})
    return routing_result


def reciprocal_rank_fusion(results: list[list], k=60):
    '''
        Reciprocal_rank_fusion that takes multiple lists of ranked documents 
            and an optional parameter k used in the RRF formula 
    '''

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


# Create a retriever for the Vectorstore
def get_docs(retriever, description):
    # RAG-Fusion to generate the similar queries for better retrieval
    multi_query_template = """ I would like you to act as an AI big language modelling assistant. Your task is to generate five different versions of five given user descriptions in order to retrieve relevant documents from a vector database.
    By generating multiple sayings of user descriptions, your goal is to help users overcome some of the limitations of distance-based search. Please provide these alternative questions separated by line breaks.
    Original Description: {description}
    Your respond (In Chinese):
    """

    prompt_query_translation = ChatPromptTemplate.from_template(template=multi_query_template)

    generate_multi_query = (
            prompt_query_translation
            | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
    )

    retrieval_chain_rag_fusion = (
            generate_multi_query
            | retriever.map()
            | reciprocal_rank_fusion
    )

    docs = retrieval_chain_rag_fusion.invoke({"description": description})
    return docs


def match_level(level_num):
    if 0 <= level_num <= 2:
        return "新手"
    elif 3 <= level_num <= 5:
        return "了解"
    elif 6 <= level_num <= 8:
        return "熟悉"
    elif 9 <= level_num <= 10:
        return "精通"
    else:
        return "新手"


def get_commentary_script(step, time_sep, time, docs, short_com, desc, level=2):
    # Combine the context(RAG) and the question to generate a templete for prompt
    final_template = """I want you to act as a Go commentator. I will give you a description of a Go game in progress, and some context that enriches the content of your commentary, and you will comment on the situation of the game and add to the contextual content provided to you in a sensible and correct way. You are commentating to a {level} of people, and you should have a {level} of knowledge of Go terminology.
    Here is a description of a Go game in progress: {description}
    Here's the context information provided to you: {context}
    You need to generate about {time_sep} seconds of commentary, you don't need to say the opening remarks as well as the closing remarks, just generate it for the available information.
    Explanation script (In Chinese):
    """

    prompt = ChatPromptTemplate.from_template(final_template)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    chain = prompt | llm

    commentary_script = chain.invoke(
        {"time_sep": time_sep, "level": match_level(level_num=level), "description": desc, "context": docs})
    # print(commentary_script)
    global commentary_script_json
    commentary_script_json.append(
        {"step": step, "Time_sep": time_sep, "Time": time, "Commentary": commentary_script.content,
         "Short_Commentary": short_com})


def retrieval_aug(lev):
    ### RAG for commentary
    vectorstore1, vectorstore2, vectorstore3 = data_preparation()
    # print(shared_dict)
    with open(configs['go_expert_knowledge_description_json_path'], mode='r', encoding='utf-8') as f:
        json_file = json.load(f)
        for i in range(len(json_file)):
            add_history_knowladge = False
            time_sep = 0
            if i == 0:
                json_file[i]['Time'] = 0
            if i == len(json_file) - 1:
                with open(configs['write_go_list_path'], mode='r', encoding='utf-8') as f:
                    json_file1 = json.load(f)
                    end_time = int(json_file1[len(json_file1) - 1]['video_length']) / 60
                    time_sep = (end_time - json_file[i]['Time'])
            else:
                time_sep = (json_file[i + 1]['Time'] - json_file[i]['Time'])
            add_history_knowladge == True if time_sep > 35 else add_history_knowladge == False
            description = json_file[i]['Commentary']
            context = []

            routing_db = routing_database(description=description)
            print('####################### Finish Database Routing {} #######################'.format(i + 1))
            print('Routing to database: ', routing_db.datasource)

            if add_history_knowladge == True:
                retriever = vectorstore1.as_retriever(search_kwargs={"k": 1})
                context = get_docs(retriever=retriever, description=description)
                print("Add context: ", context[0])
            else:
                if routing_db.datasource == 'classic_go_match':
                    retriever = vectorstore2.as_retriever(search_kwargs={"k": 2})
                    context = get_docs(retriever=retriever, description=description)
                    print("Add context: ", context[0])
                elif routing_db.datasource == 'recent_go_match_news':
                    retriever = vectorstore3.as_retriever(search_kwargs={"k": 2})
                    context = get_docs(retriever=retriever, description=description)
                    print("Add context: ", context[0])

            print('####################### Finish Database Retrieval {} #######################'.format(i + 1))
            get_commentary_script(step=i + 1, time_sep=round(time_sep, 2), time=round(json_file[i]['Time'], 2),
                                  docs=context, desc=description, short_com=json_file[i]['Short_commentary'], level=lev)

    with open(configs['go_description_rag_json_path'], 'w', encoding='utf-8') as f:
        json.dump(commentary_script_json, f, indent=4, ensure_ascii=False)
    print('####################### Finish RAG #######################')


if __name__ == '__main__':
    ### RAG for commentary
    retrieval_aug(lev=2)
