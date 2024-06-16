# -*- codeing = utf-8 -*-
import requests
import json
import os
import openai

from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydub import AudioSegment
#输入json文件

configs = json.loads(open("./configs/config.json").read())
os.environ["OPENAI_API_KEY"] = configs['openai_api_key']
os.environ["OPENAI_API_BASE"] = configs['openai_api_url']


def get_audio_length(file_path):
    audio = AudioSegment.from_file(file_path)
    duration = audio.duration_seconds
    return duration


def change_voice(person):
    if(person=="Azuma"):
        data = {
            "gpt_model_path": "GPT_weights/Azuma-e10.ckpt",
            "sovits_model_path":"SoVITS_weights/Azuma_e35_s1435.pth"
        }
        response = requests.post("http://127.0.0.1:9880/set_model", json=data)

        if response.status_code == 400:
            raise Exception(f'请求GPTSOVITS出现错误:{response.message}')
        print(response.content)
    if(person=="dingzhen"):
        data = {
            "gpt_model_path": "GPT_weights/dingzhen-e15.ckpt",
            "sovits_model_path":"SoVITS_weights/Azuma_e35_s1435.pth"
        }
        response = requests.post("http://127.0.0.1:9880/set_model", json=data)

        if response.status_code == 400:
            raise Exception(f'请求GPTSOVITS出现错误:{response.message}')
        print(response.content)
    if(person=="Carol"):
        data = {
            "gpt_model_path": "GPT_weights/Carol-e15.ckpt",
            "sovits_model_path":"SoVITS_weights/Carol_e40_s2160.pth"
        }
        response = requests.post("http://127.0.0.1:9880/set_model", json=data)

        if response.status_code == 400:
            raise Exception(f'请求GPTSOVITS出现错误:{response.message}')
        print(response.content)
    if(person=="xuan"):
        data = {
            "gpt_model_path": "GPT_weights/xuan-e15.ckpt",
            "sovits_model_path":"SoVITS_weights/xuan_e12_s408.pth"
        }
        response = requests.post("http://127.0.0.1:9880/set_model", json=data)

        if response.status_code == 400:
            raise Exception(f'请求GPTSOVITS出现错误:{response.message}')
        print(response.content)


def get_audio(file_path,person,video_path,output_path):
    InputPath = open(file_path, encoding="utf-8")
    # 设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
    temp = json.load(InputPath)
    #json格式数据转换为python字典类型
    for i in range(len(temp)):
        temp[i]["Commentary"]=summarize(temp[i]["Commentary"],temp[i]["Time_sep"])
    if person == "dingzhen":
        change_voice("dingzhen")
        count = 1
        for i in range(len(temp)):
            base1 = temp[i]["Commentary"]  # 因为此时已经转换为了字典类型
            data={
                "refer_wav_path": "./dataset/dingzhen/dingzhen_46.wav",
                "prompt_text": "今天我想跟你们说说。我学到的东西，先给大家讲讲我过去的生活。",
                "prompt_language": "zh",
                "text": base1,
                "text_language": "zh",
            }

            response=requests.post("http://127.0.0.1:9880",json=data)

            if response.status_code==400:
                raise Exception(f'请求GPTSOVITS出现错误:{response.message}')
            
            path = configs['go_commentary_audio_path'] + f"commentary{count}.wav"
            # print(path)
            with open(path, 'wb') as f:
                f.write(response.content)
            count += 1
    if person == "Azuma":
        change_voice("Azuma")
        count = 1
        for i in range(len(temp)):
            base1 = temp[i]["Commentary"]  # 因为此时已经转换为了字典类型
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
            count+=1
    if person == "xuan":
        change_voice("xuan")
        count = 1
        for i in range(len(temp)):
            base1 = temp[i]["Commentary"]  # 因为此时已经转换为了字典类型
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
            count+=1
    if person == "Carol":
        change_voice("Carol")
        count = 1
        for i in range(len(temp)):
            base1 = temp[i]["Commentary"]  # 因为此时已经转换为了字典类型
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
            count+=1
    audios = os.listdir(configs['go_commentary_audio_path'])
    audios = [configs['go_commentary_audio_path']+i for i in audios]
    insert_time = [entry["Time"] for entry in temp]
    time_sep = [entry["Time_sep"] for entry in temp]
    insert_audios_to_video(video_path,audios,insert_time,time_sep,output_path)


def summarize(text,time_sep):
    template = """ 帮我概括这段话到{time_sep}之内\n:
    Q:{text}
    A:
    """
    summarize_prompt = ChatPromptTemplate.from_template(template=template)
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    chain = summarize_prompt | llm

    response = chain.invoke({"time_sep": time_sep, "text": text})
    return response.content


def insert_audios_to_video(video_path, audio_segments, insert_times,time_sep ,output_path):
    video = VideoFileClip(video_path)
    original_audio = video.audio

    # 初始化音频片段列表
    audio_clips = []

    # 遍历所有音频片段
    for audio_file, start_time, time_sep in zip(audio_segments,insert_times,time_sep):
        new_audio = AudioFileClip(audio_file).subclip(0, time_sep)
        new_audio = new_audio.set_start(start_time)
        audio_clips.append(new_audio)

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
    person = "Azuma"
    person_list = ["dingzhen","Azuma","xuan","Carol"]
    get_audio(file_path, person,video_path, output_path)

if __name__ == '__main__':
    file_path = configs['go_description_rag_json_path']
    audio_path = configs['go_commentary_audio_path']
    video_path = configs['video_path']  # 替换为你的无声视频文件路径
    output_path = configs['go_commentary_output_video_path']  # 替换为你希望保存的输出视频文件路径
    person = "Azuma"
    person_list = ["dingzhen","Azuma","xuan","Carol"]
    get_audio(file_path, person,video_path, output_path)
