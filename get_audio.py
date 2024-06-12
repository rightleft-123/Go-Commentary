# -*- codeing = utf-8 -*-
# @Time : 2024/5/26 17:59
# @Yuwendi
# @File : get_audio.py
# @Software: PyCharm
import requests
import json
import os
import openai

from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
#输入json文件

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
    if(person=="xuan"):
        data = {
            "gpt_model_path": "GPT_weights/-e15.ckpt",
            "sovits_model_path":"SoVITS_weights/xuan_e12_s408.pth"
        }
        response = requests.post("http://127.0.0.1:9880/set_model", json=data)

        if response.status_code == 400:
            raise Exception(f'请求GPTSOVITS出现错误:{response.message}')
        print(response.content)


def get_audio(file_path,person):
    InputPath = open(file_path, encoding="utf-8")
    # 设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
    temp = json.load(InputPath)      #json格式数据转换为python字典类型
    for i in range(len(temp)):
        temp[i]["Commentary"]=summarize(temp[i]["Commentary"],temp[i]["Time"])
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

            with open(f"./result/commentary{count}.wav",'wb') as f:
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

            with open(f"./result/commentary{count}.wav", 'wb') as f:
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

            with open(f"./result/commentary{count}.wav", 'wb') as f:
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

            with open(f"./result/commentary{count}.wav", 'wb') as f:
                f.write(response.content)
            count+=1
def summarize(text,time):
    API_SECRET_KEY = "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"
    url="https://api.openai.com/v1/chat/completions"
    header = {"Content-Type": "application/json", "Authorization": "Bearer " + API_SECRET_KEY}
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"帮我概括这段话到{time}之内\n"+text
            }
        ],
        "temperature": 0,
        "stream": False
    }
    response = requests.post(url=url, headers=header, json=data).json()
    print(response)
    return response

def add_audio_to_video(video_path, audio_path, insert_time, output_path):
    """
    在无声视频的指定时间点插入音频。
    :param video_path: 无声视频文件路径
    :param audio_path: 要插入的音频文件夹路径
    :param insert_time: 插入时间点的列表（以秒为单位）
    :param output_path: 输出视频文件路径
    """
    copy= VideoFileClip(video_path)
    copy.write_videofile(output_path, codec='libx264', audio_codec='aac')
    audios = os.listdir(audio_path)
    audios = [audio_path+"/"+i for i in audios]
    for i, j in zip(audios[0:len(insert_time)],insert_time):
        video = VideoFileClip(output_path)
        original_audio = video.audio
        new_audio = AudioFileClip(i)
        # 将新的音频插入到原音频的指定时间点
        # 截取原音频前半部分和后半部分
        before_insert = original_audio.subclip(0, j)
        after_insert = original_audio.subclip(j)

        # 计算新的音频和插入后的音频长度
        combined_audio = concatenate_audioclips(
            [before_insert, new_audio, after_insert.set_start(j + new_audio.duration)])
        combined_audio = combined_audio.set_duration(original_audio.duration)

        # 将合并后的音频设置为视频的音频
        video = video.set_audio(combined_audio)
        # 保存输出视频
        video.write_videofile(output_path, codec='libx264', audio_codec='aac')

file_path = "C:/Users/ywd/Desktop/commentary.json"
audio_path = "./result"
video_path = "C:/Users/ywd/Desktop/1718016213782.mp4"  # 替换为你的无声视频文件路径
insert_time = [5.0,10.0]  # 插入时间点（以秒为单位）
output_path = "./output_video.mp4"  # 替换为你希望保存的输出视频文件路径
person = "xuan"
person_list = ["dingzhen","Azuma","xuan","Carol"]
get_audio(file_path,person)
