# -*- codeing = utf-8 -*-
# @Time : 2024/5/26 17:59
# @Yuwendi
# @File : get_audio.py
# @Software: PyCharm
import requests
import json
import os
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
#输入json文件
def get_audio(file_path):
    InputPath = open(file_path, encoding="utf-8")
    # 设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
    temp = json.load(InputPath)      #json格式数据转换为python字典类型
    count = 1
    for i in range(len(temp)):
        base1 = temp[i]["Commentary"]  # 因为此时已经转换为了字典类型
        data={
            "refer_wav_path": "./dataset/Azuma Max/Azuma_17.wav",
            "prompt_text": "主播聊八卦，还有一些粉丝聊八卦，都会听到很多那种逆天主播",
            "prompt_language": "zh",
            "text": base1,
            "text_language": "zh",
            "speed":1.5
        }

        response=requests.post("http://127.0.0.1:9880",json=data)

        if response.status_code==400:
            raise Exception(f'请求GPTSOVITS出现错误:{response.message}')

        with open(f"./result/commentary{count}.wav",'wb') as f:
            f.write(response.content)
        count += 1

def add_audio_to_video(video_path, audio_path, insert_time, output_path):
    """
    在无声视频的指定时间点插入音频。
    :param video_path: 无声视频文件路径
    :param audio_path: 要插入的音频文件夹路径
    :param insert_time: 插入时间点的列表（以秒为单位）
    :param output_path: 输出视频文件路径
    """
    audios = os.listdir(audio_path)
    audios = [audio_path+"/"+i for i in audios]
    for i, j in zip(audios[0:len(insert_time)],insert_time):
        video = VideoFileClip(video_path)
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
get_audio(file_path)
add_audio_to_video(video_path,audio_path,insert_time,output_path)