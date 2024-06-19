# -*- coding: utf-8 -*-
import json
import time
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from tqdm import trange

configs = json.loads(open("../configs/config.json").read())

# device = torch.device('cuda:1')
# model_path = configs['go_expert_model_path']
# model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

client = OpenAI(
    api_key=configs['openai_api_key'],
    base_url=configs['openai_api_url'],
    timeout=configs['openai_api_timeout']
)


def use_gpt(client, messages, model, max_try, key_list):
    cnt = 0
    while cnt < max_try:
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
            )
            print(f'{chat_completion.usage.total_tokens} prompt tokens counted by the OpenAI API.')
            bot_text = chat_completion.choices[0].message.content.strip()
            if key_list != []:
                bot_text = bot_text.replace('```', '')

                if bot_text[:4] == 'json':
                    bot_text = bot_text[4:]
                bot_text = json.loads(bot_text)

                for key in key_list:
                    assert key in bot_text
            return bot_text

        except Exception as e:
            e = str(e)
            e = e.replace('\n', '//n')
            print(e)
            cnt += 1
            time.sleep(0.1)
            print(f'Try {cnt} of {max_try}... ')

    return f"Error: Failed after {max_try} attempts"


def convert_numbers_to_chinese(s):
    num_dict = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
                '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}
    unit_dict = {2: '十', 3: '百', 4: '千', 5: '万'}

    def convert(num_str):
        if not num_str:
            return ''
        num_str = num_str[::-1]
        res = ''
        for idx, char in enumerate(num_str):
            if idx > 0 and char != '0':
                res += unit_dict[idx + 1]
            res += num_dict[char]
        return res[::-1]

    import re
    res = ''
    for part in re.split(r'(\d+)', s):
        if part.isdigit():
            res += convert(part)
        else:
            res += part
    return res


def gen_descript(goList_file):
    go_records = []
    with open(goList_file, 'r', encoding='utf-8') as f:
        f = json.load(f)
        for line in f:
            go_records.append(line)
        # for line in f.readlines():
        #     dic = json.loads(line)
        #     go_records.append(dic)

    go_records_cleaned = [{"step": 0, "black": [], "white": []}]
    go_records_cleaned.append(go_records[0])

    for i in range(1, len(go_records) - 1):
        if set(go_records[i]['black']) != set(go_records[i - 1]['black']) or set(go_records[i]['white']) != set(
                go_records[i - 1]['white']):
            go_records_cleaned.append(go_records[i])
            print(go_records[i]['step'])

    ###
    # go_records_cleaned=go_records_cleaned[:5]
    ###

    description = []
    for i in trange(1, len(go_records_cleaned)):
        prompt = (
            "You are a Go expert, I will give you the previous chess board and the current chess board together with the places for white pieces and black pieces. You should first find the move of the current step, which new piece has been put on or which pieces were taken off the board."
            "\nFirst please recall some basic knowledge of the Go game, black should play first and each turn black and white play one after another. The chess board is 19*19, whose row numbers are: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], line numbers are: [a,b,c,d,e,f,g,h,j,k,l,m,n,o,p,q,r,s,t], the lower left corner is A1."
            "\nThe inputs are as follows:"
            f"\nPrevious board with the pieces:{go_records_cleaned[i - 1]}"
            f"\nCurrent step board with the pieces:{go_records_cleaned[i]}"
            f"\nPlease give your output in JSON format, with keys are \"New_changed_pieces\", \"Description\", \"Questions\"."
            f"\nUnder the content of \"New_changed_pieces\", you should carefully look at the previous and current board, find the differences between them find the new adding pieces or romoved pieces. Tell me the places also in python List format."
            f"\nUnder the content of \"Description\", generate a description of the current board's pieces. Your description should be detailed, don't omit any information or detail. The description should contain two parts: 1. The description of the specific area of the new adding pieces. 2.The description of the general situation of black and white pieces. Please generate the \"Description\" in the format of a python String!"
            f"\nUnder the content of \"Questions\", generate at most 3 expertise questions of the current board's pieces. Your questions should be about some fundmental Go knowledge, just think due to current Go board what Go knowledge may you ask about? Generate the questions in python list."
            f"\nGenerate all your output in Chinese!")

        messages = [{'role': 'user', 'content': prompt}]
        response = use_gpt(client, messages, 'gpt-3.5-turbo', 10, ["New_changed_pieces", "Description", "Questions"])

        black_set1 = set(go_records_cleaned[i - 1]['black'])
        black_set2 = set(go_records_cleaned[i]['black'])

        # 找出list1中有但list2中没有的元素
        black_missing_in_list2 = list(black_set1.difference(black_set2))

        # 找出list2中有但list1中没有的元素
        black_new_in_list2 = list(black_set2.difference(black_set1))

        white_set1 = set(go_records_cleaned[i - 1]['white'])
        white_set2 = set(go_records_cleaned[i]['white'])

        # 找出list1中有但list2中没有的元素
        white_missing_in_list2 = list(white_set1.difference(white_set2))

        # 找出list2中有但list1中没有的元素
        white_new_in_list2 = list(white_set2.difference(white_set1))

        short_dscrpt = '在这一步中，'

        if black_new_in_list2 != []:
            short_dscrpt += '黑棋下在了 '
            for j in range(len(black_new_in_list2)):
                short_dscrpt += black_new_in_list2[j]
                short_dscrpt += '，'
            short_dscrpt += ' '

        if white_new_in_list2 != []:
            short_dscrpt += '白棋下在了 '
            for j in range(len(white_new_in_list2)):
                short_dscrpt += white_new_in_list2[j]
                short_dscrpt += '，'
            short_dscrpt += ' '

        if black_missing_in_list2 != []:
            short_dscrpt += '黑棋在 '
            for j in range(len(black_missing_in_list2)):
                short_dscrpt += black_missing_in_list2[j]
                short_dscrpt += '，'
            short_dscrpt += '的位置的棋子被吃掉了。 '

        if white_missing_in_list2 != []:
            short_dscrpt += '白棋在 '
            for j in range(len(white_missing_in_list2)):
                short_dscrpt += white_missing_in_list2[j]
                short_dscrpt += '，'
            short_dscrpt += '的位置的棋子被吃掉了。 '

        record = {}
        record['Step'] = go_records_cleaned[i]['step']
        record['Time'] = int(go_records_cleaned[i]['time']) / 60
        record["Description"] = convert_numbers_to_chinese(response["Description"])
        record["Questions"] = response["Questions"]
        record["Short_Description"] = short_dscrpt
        print(record)
        description.append(record)

    return description


def knowledge_enhance(description, tokenizer, model):
    go_knowledge = {}
    for i in trange(len(description)):
        go_knowledge[description[i]["Step"]] = []
        questions = description[i]["Questions"]
        for question in questions:
            # prompt = f"请回答下面有关围棋知识的问题：{question}"
            # messages = [
            #     {"role": "system", "content": "你是一个很棒的助手."},
            #     {"role": "user", "content": prompt}
            # ]
            # text = tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )
            # model_inputs = tokenizer([text], return_tensors="pt").to(device)
            #
            # generated_ids = model.generate(
            #     model_inputs.input_ids,
            #     max_new_tokens=128
            # )
            # generated_ids = [
            #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            # ]
            #
            # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response = 'lwy'
            go_knowledge[description[i]["Step"]].append(response)

    return go_knowledge


def merge_go_expert_knowledge(descriptions, go_knowledge):
    enhanced_descriptions = []
    for i in trange(len(descriptions)):
        step = descriptions[i]["Step"]
        description = descriptions[i]["Description"]
        knowledge = str(go_knowledge[step])
        prompt = (
            "You are a Go commentator, I will give you the description of the current chess board together with some Go knowledge for you to use for generating your commentary."
            "\nThe inputs are as follows:"
            f"\nDescription of the current chess board:{description}"
            f"\nGo knowledge:{knowledge}"
            f"\nPlease directly give your output commentary, generate the commentary by enhancing the description with the Go knowledge. Don't explain, don't say any other words."
            f"\nGenerate all your output in Chinese!")

        messages = [{'role': 'user', 'content': prompt}]
        response = use_gpt(client, messages, 'gpt-3.5-turbo', 10, [])
        enhanced_descriptions.append({'Step': step, 'Time': descriptions[i]["Time"], 'Commentary': response,
                                      'Short_commentary': descriptions[i]["Short_Description"]})

    return enhanced_descriptions


def knowledge_aug():
    ### Description to Go Commentary
    goList_file = configs['write_go_list_path']
    descriptions = gen_descript(goList_file)

    with open(configs['go_description_json_path'], 'w+', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=4, ensure_ascii=False)

    go_knowledge = knowledge_enhance(descriptions, tokenizer='lwy', model='lwy')

    enhanced_descriptions = merge_go_expert_knowledge(descriptions, go_knowledge)
    with open(configs['go_expert_knowledge_description_json_path'], 'w+', encoding='utf-8') as f:
        json.dump(enhanced_descriptions, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    ### Description to Go Commentary
    knowledge_aug()
