# -*- codeing = utf-8 -*-
import copy
import json
import os
import shutil

import requests
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
from pydub import AudioSegment

# 输入json文件

configs = json.loads(open("../configs/config.json").read())
os.environ["OPENAI_API_KEY"] = configs['openai_api_key']
os.environ["OPENAI_API_BASE"] = configs['openai_api_url']


def delete_folder_contents(folder_path):
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 使用 shutil.rmtree 删除文件夹中的所有内容
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 递归删除目录
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The directory {folder_path} does not exist.')


def get_audio_length(file_path):
    audio = AudioSegment.from_file(file_path)
    duration = audio.duration_seconds
    return duration


def change_voice(person):
    if person == "Azuma":
        data = {
            "gpt_model_path": "GPT_weights/Azuma-e10.ckpt",
            "sovits_model_path": "SoVITS_weights/Azuma_e35_s1435.pth"
        }
        response = requests.post("http://127.0.0.1:9880/set_model", json=data)

        if response.status_code == 400:
            raise Exception(f'请求GPTSOVITS出现错误:{response.message}')
        print(response.content)
    if (person == "dingzhen"):
        data = {
            "gpt_model_path": "GPT_weights/dingzhen-e15.ckpt",
            "sovits_model_path": "SoVITS_weights/Azuma_e35_s1435.pth"
        }
        response = requests.post("http://127.0.0.1:9880/set_model", json=data)

        if response.status_code == 400:
            raise Exception(f'请求GPTSOVITS出现错误:{response.message}')
        print(response.content)
    if (person == "Carol"):
        data = {
            "gpt_model_path": "GPT_weights/Carol-e15.ckpt",
            "sovits_model_path": "SoVITS_weights/Carol_e40_s2160.pth"
        }
        response = requests.post("http://127.0.0.1:9880/set_model", json=data)

        if response.status_code == 400:
            raise Exception(f'请求GPTSOVITS出现错误:{response.message}')
        print(response.content)
    if (person == "xuan"):
        data = {
            "gpt_model_path": "GPT_weights/xuan-e15.ckpt",
            "sovits_model_path": "SoVITS_weights/xuan_e12_s408.pth"
        }
        response = requests.post("http://127.0.0.1:9880/set_model", json=data)

        if response.status_code == 400:
            raise Exception(f'请求GPTSOVITS出现错误:{response.message}')
        print(response.content)


def get_audio(file_path, person, video_path, output_path):
    delete_folder_contents('../middle_result/audio_result')
    InputPath = open(file_path, encoding="utf-8")
    # 设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
    temp = json.load(InputPath)
    # json格式数据转换为python字典类型
    copy1 = []
    dic = {"Commentary": "", "Time_sep": 0, "Time": 0}
    k = 0
    if (temp[0]["Time_sep"] <= 2):
        temp[0]["Commentary"] = ""
    elif (temp[0]["Time_sep"] > 2 and temp[0]["Time_sep"] <= 8):
        temp[0]["Commentary"] = "这是一场正在进行的围棋比赛。"
    elif (temp[0]["Time_sep"] <= 10):
        temp[0]["Commentary"] = "这是一场正在进行的围棋比赛,现在双方都在寻找最佳的下法，以争夺胜利。"
    while True:
        if (dic["Time_sep"] >= 5):
            temp_dict = copy.deepcopy(dic)
            copy1.append(temp_dict)
            dic["Commentary"] = ""
            dic["Time"] = dic["Time"] + dic["Time_sep"]
            dic["Time_sep"] = 0
        if (k == len(temp)):
            break
        # 五秒之内的用short
        if (temp[k]["Time_sep"] <= 4):
            dic["Commentary"] = dic["Commentary"] + temp[k]["Short_Commentary"]
            dic["Time_sep"] = dic["Time_sep"] + temp[k]["Time_sep"]
        # 超过十秒放长comment
        else:
            dic["Commentary"] = dic["Commentary"] + temp[k]["Commentary"]
            dic["Time_sep"] = dic["Time_sep"] + temp[k]["Time_sep"]
        k += 1
    for i in range(1, len(copy1)):
        if (copy1[i]["Time_sep"] <= 15):
            copy1[i]["Commentary"] = summarize(copy1[i]["Commentary"], copy1[i]["Time_sep"])
        elif (copy1[i]["Time_sep"] > 15):
            copy1[i]["Commentary"] = summarize1(copy1[i]["Commentary"], copy1[i]["Time_sep"])
    print("summarize")
    if person == "dingzhen":
        change_voice("dingzhen")
        count = 1
        for i in range(len(copy1)):
            base1 = copy1[i]["Commentary"]  # 因为此时已经转换为了字典类型
            data = {
                "refer_wav_path": "./dataset/dingzhen/dingzhen_46.wav",
                "prompt_text": "今天我想跟你们说说。我学到的东西，先给大家讲讲我过去的生活。",
                "prompt_language": "zh",
                "text": base1,
                "text_language": "zh",
            }

            response = requests.post("http://127.0.0.1:9880", json=data)

            if response.status_code == 400:
                raise Exception(f'请求GPTSOVITS出现错误:{response.message}')

            path = configs['go_commentary_audio_path'] + f"commentary{count}.wav"
            # print(path)
            with open(path, 'wb') as f:
                f.write(response.content)
            count += 1
    if person == "Azuma":
        change_voice("Azuma")
        count = 1
        for i in range(len(copy1)):
            base1 = copy1[i]["Commentary"]  # 因为此时已经转换为了字典类型
            data = {
                "refer_wav_path": "./dataset/Azuma Max/Azuma_131.wav",
                "prompt_text": "不愧是黄毛，嗯反正我觉得不行。",
                "prompt_language": "zh",
                "text": base1,
                "text_language": "zh",
            }

            response = requests.post("http://127.0.0.1:9880", json=data)

            if response.status_code == 400:
                raise Exception(f'请求GPTSOVITS出现错误:{response.message}')

            path = configs['go_commentary_audio_path'] + f"commentary{count}.wav"
            # print(path)
            with open(path, 'wb') as f:
                f.write(response.content)
            count += 1
    if person == "xuan":
        change_voice("xuan")
        count = 1
        for i in range(len(copy1)):
            base1 = copy1[i]["Commentary"]  # 因为此时已经转换为了字典类型
            data = {
                "refer_wav_path": "./dataset/xuan/xuan_11.wav",
                "prompt_text": "不烧脑不烧脑我已经完全明白了已经不烧脑了，我完全懂了，我他妈砍疯了都。",
                "prompt_language": "zh",
                "text": base1,
                "text_language": "zh",
            }

            response = requests.post("http://127.0.0.1:9880", json=data)

            if response.status_code == 400:
                raise Exception(f'请求GPTSOVITS出现错误:{response.message}')

            path = configs['go_commentary_audio_path'] + f"commentary{count}.wav"
            # print(path)
            with open(path, 'wb') as f:
                f.write(response.content)
            count += 1
    if person == "Carol":
        change_voice("Carol")
        count = 1
        for i in range(len(copy1)):
            base1 = copy1[i]["Commentary"]  # 因为此时已经转换为了字典类型
            data = {
                "refer_wav_path": "./dataset/Carol/Carol_161.wav",
                "prompt_text": "睁开你你那个看到玉米肠就会变成星星眼的眼睛看看这可是粉色啊。",
                "prompt_language": "zh",
                "text": base1,
                "text_language": "zh",
            }

            response = requests.post("http://127.0.0.1:9880", json=data)

            if response.status_code == 400:
                raise Exception(f'请求GPTSOVITS出现错误:{response.message}')

            path = configs['go_commentary_audio_path'] + f"commentary{count}.wav"
            # print(path)
            with open(path, 'wb') as f:
                f.write(response.content)
            count += 1
    time_sep = [entry["Time_sep"] for entry in copy1]
    audios = os.listdir(configs['go_commentary_audio_path'])
    audios = [configs['go_commentary_audio_path'] + i for i in audios]
    insert_audios_to_video(video_path, audios, time_sep, output_path)


def summarize(text, time_sep):
    template = f"现在我有一段围棋比赛解说文本如下：{text}\n\n解说员用正常语速解说时，大概是3~4个字/秒(包括标点符号)，现在我的要求如下：\n我需要将这段解说词压缩到正常语速解说下持续{time_sep}秒；\n改写后只要求文本量缩减，对棋局的正确描述不得改变，并且出现的字母尽可能少。\n\n请给我改写后的文本。"
    summarize_prompt = ChatPromptTemplate.from_template(template=template)
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    chain = summarize_prompt | llm

    response = chain.invoke({"time_sep": time_sep - 2, "text": text})
    return response.content


def summarize1(text, time):
    keep_time = int(time / 4)
    text1 = text[:keep_time*6*2]
    text2 = text[keep_time * 6*2:]

    template = (f"你是一个围棋解说员，我会给你提供一个上下文和一段话，请你根据上下文对这一段话根据我的要求进行压缩。"
                f"具体要求如下："
                f"1. 请你将等待压缩的话压缩到 {keep_time*6} 个字以内， 注意每一个汉字、标点符号、数字都算一个字，总共加起来不能超过 {keep_time*6} 个字。"
                f"2. 请你仔细阅读给你的上下文和待压缩的话，他们都是很好的围棋解说词，请保证你的压缩后的词仍然具有相同的解说词风格。"
                f"3. 上下文和给你的这一段等待压缩的话是前后文，同属于一段话，因此你产生的压缩后的话应当能够和上下文完美的衔接起来，没有纰漏，很流畅。"
                f"你的输入如下："
                f"上下文：<<<{text1}>>>"
                f"等待压缩的话：<<<{text2}>>>"
                f"不要解释，不要说任何其他的词，严格按照我的要求来，直接给出对于等待压缩的话的压缩后的结果。")

    summarize_prompt = ChatPromptTemplate.from_template(template=template)
    llm = ChatOpenAI(model='gpt-4-1106-preview', temperature=0)
    chain = summarize_prompt | llm

    response = chain.invoke({"text1": text1, "text2": text2, "keep_time": keep_time})

    comment = text1+response.content
    return comment


def insert_audios_to_video(video_path, audios, time_sep, output_path):
    with VideoFileClip(video_path) as video:
        audio = AudioFileClip(video_path)
        # 初始化音频片段列表
        audio_clips = []

        # 遍历所有音频片段
        for audio_file, time_sep in zip(audios, time_sep):
            new_audio = AudioFileClip(audio_file)
            duration = new_audio.duration
            print(duration)
            if (duration > time_sep):
                new_audio1 = new_audio.subclip(0, time_sep)
                audio_clips.append(new_audio1)
            else:
                silence_duration = time_sep - duration
                silence = audio.subclip(0, silence_duration)
                silence = silence.volumex(0)
                audio_clips.append(new_audio)
                audio_clips.append(silence)

        # 将所有音频片段和原始音频合成一个音频
        final_audio = concatenate_audioclips(audio_clips)
        # 设置合成后的音频为视频的音频
        final_video = video.set_audio(final_audio)
        # 保存输出视频
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')


def text2speech():
    file_path = configs['go_description_rag_json_path']
    audio_path = configs['go_commentary_audio_path']
    video_path = configs['video_path']  # 替换为你的无声视频文件路径
    output_path = configs['go_commentary_output_video_path']  # 替换为你希望保存的输出视频文件路径
    person = "Carol"
    person_list = ["dingzhen", "Azuma", "xuan", "Carol"]
    get_audio(file_path, person, video_path, output_path)


if __name__ == '__main__':
    text2speech()
