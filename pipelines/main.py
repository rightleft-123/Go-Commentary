import base64
import os
import streamlit as st


# 将图片转换为 base64 编码
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


# 设置页面配置
# 设置页面配置
current_dir = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(current_dir, "../figure/gogo.jpg")  # 请确保这个路径下有一个icon.png文件

# 获取base64编码后的图片
icon_base64 = get_base64_image(icon_path)

# 设置页面配置
st.set_page_config(
    page_title="Go to Go",
    page_icon=f"data:image/png;base64,{icon_base64}"
)

import time
from PIL import Image
import json
from langchain_openai import ChatOpenAI


from Image2Text import image2text
from Expert_Knowledge import knowledge_aug
from Content_Enrich import retrieval_aug
from Text2Speech import text2speech


# 显示图片和文字
def show_image_and_text(images, texts, index):
    try:
        with st.spinner("加载中..."):
            st.write(f"正在加载图片：{images[index]}")
            image = Image.open(images[index])
            st.image(image, caption=f"图片 {index + 1}")
            st.write(texts[index])
    except Exception as e:
        st.error(f"加载图片 {images[index]} 出错：{e}")


def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


configs = json.loads(open("../configs/config.json").read())
os.environ["OPENAI_API_KEY"] = configs['openai_api_key']
os.environ["OPENAI_API_BASE"] = configs['openai_api_url']

if __name__ == '__main__':

    # 定义相对路径
    image_path = os.path.join(current_dir, '../figure/chess.png')  # 当前文件夹下的图

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # 使用 HTML 和 CSS 设置背景图片
    page_bg_img = f'''
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{encoded_string}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0);
    }}

    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)

    # 标题和描述
    st.markdown(
        f'<div style="display: flex; align-items: center;">'
        f'<h1 style="margin: 0;">Go Commentary</h1>'
        f'<img src="data:image/png;base64,{icon_base64}" width="50" style="margin-left: 10px;"/>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.write("Go Commentary 这是一个围棋解说应用程序。")

    # 交互式小部件
    st.sidebar.header('hello，欢迎来到GO-Commentary')
    user_input = st.sidebar.text_input("请输入你的名字", "GO-Commentary 用户")
    st.sidebar.write(f"你好, {user_input}!")

    level = 2
    level = st.sidebar.slider("选择你的围棋水平", 0, 10, 0)
    st.sidebar.write(f"你的围棋水平是: {level}")
    st.sidebar.write(
        "0-2说明你是围棋新手，\n刚接触围棋;3-5说明你是围棋爱好者，\n具有一定的围棋基础;6-8说明你已经是围棋高手，\n具有很强的围棋功底;9-10围棋大师")

    # 将获取的信息组合成一个字符串
    info_string = f"名字: {user_input}, 围棋水平: {level}"

    # 定义保存信息的文件路径
    file_path = os.path.join(current_dir, '../output/user_level.txt')  # 修改为实际保存信息的路径

    # 将信息保存到本地文件
    if st.sidebar.button("提交信息"):
        try:
            with open(file_path, 'a') as f:
                f.write(info_string + "\n")
            st.sidebar.success("信息已成功上传")
        except Exception as e:
            st.sidebar.error(f"信息保存失败: {e}")

    # 图片和文字数据
    folder_path = configs['folder_go_list_path'][0:-1]
    file_list = os.listdir(folder_path)

    # 创建一个空列表来存储图片
    image_paths_for_keyframe = []

    # 遍历文件列表，获取图片路径
    for file_name in file_list:
        image_paths_for_keyframe.append(os.path.join(folder_path, file_name))

    text_list_for_commentary = []
    with open(configs['go_description_rag_json_path'], 'r', encoding='utf-8') as f:
        json_file = json.load(f)
        for i in range(len(json_file)):
            text_list_for_commentary.append(json_file[i]['Commentary'])

    # 文件上传
    # 视频上传和展示
    st.write("上传和展示视频：")
    uploaded_video = st.file_uploader("上传一个围棋视频文件，将为您提供围棋讲解", type=["mp4", "mov", "avi"])

    # 定义保存路径
    # if os.path.exists('../output/output_video.mp4'):
    #     os.remove('../output/output_video.mp4')
    uploaded_video_path = os.path.join(current_dir, '../video/new_video.mp4')
    processed_video_path = os.path.join(current_dir, '../output/output_video.mp4')

    if uploaded_video is not None:
        # 保存上传的视频文件
        st.write("正在保存上传的视频文件...")
        save_uploaded_file(uploaded_video, uploaded_video_path)
        st.success(f"正在处理")

        with st.spinner("这可能需要一些时间，请稍候..."):
            while not os.path.exists(processed_video_path):
                image2text()
                print(11111)
                # knowledge_aug()
                print(22222)
                retrieval_aug(lev=level)
                print(33333)
                text2speech()
                print(44444)
                time.sleep(5)
        # 假设处理完后生成新的视频文件和字幕文件
        # 这里你可以添加实际的处理逻辑，例如调用本地处理脚本
        # 示例代码假设处理后的文件已经存在于指定路径

        if os.path.exists(processed_video_path):
            # 读取处理后的视频文件
            with open(processed_video_path, "rb") as video_file:
                video_bytes = video_file.read()

            # 编码视频文件为 Base64
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')

            # 嵌入 HTML 播放视频
            video_html = f'''
            <video width="700" controls>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            '''
            st.markdown(video_html, unsafe_allow_html=True)
            st.session_state['video_uploaded'] = True
        else:
            st.error("视频处理失败，请重试。")
            st.session_state['video_uploaded'] = False

        # 初始化 session state 变量
        if 'image_index' not in st.session_state:
            st.session_state['image_index'] = 0
        if 'show_image' not in st.session_state:
            st.session_state['show_image'] = False
        # 如果视频已上传，显示解析和下一步按钮
        if st.session_state.get('video_uploaded', False):
            if st.button("解析首页"):
                st.session_state['show_image'] = True
                show_image_and_text(images=image_paths_for_keyframe, texts=text_list_for_commentary,
                                    index=st.session_state['image_index'])
            if st.button("下一步"):
                st.session_state['image_index'] = (st.session_state['image_index'] + 1) % len(image_paths_for_keyframe)
                show_image_and_text(images=image_paths_for_keyframe, texts=text_list_for_commentary,
                                    index=st.session_state['image_index'])
    else:
        st.write("请上传视频文件。")

    # 文本解释框
    st.write("### 提问")
    question = st.text_area("在此输入你的问题和疑惑：", "围棋解答窗口。你可以在这里输入任何问题。")

    if st.button("提交问题"):
        # 获取用户的答案（这里可以是你想要的任何处理逻辑）
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        answer = llm.invoke(question).content

        # 定义保存文件的路径
        if os.path.exists('../output/answer.txt'):
            os.remove('../output/answer.txt')

        file_path1 = os.path.join(current_dir, '../output/question.txt')
        file_path2 = os.path.join(current_dir, '../output/answer.txt')

        # 模拟处理时间
        st.write("正在处理，请稍候...")
        time.sleep(2)  # 模拟处理时间

        # 将问题和答案保存到文件
        try:
            with open(file_path1, 'w', encoding='utf-8') as f:
                f.write(f"问题: {question}\n")

            with st.spinner("正在提问ing，请稍候..."):
                while not os.path.exists(file_path2):
                    with open(file_path2, 'w', encoding='utf-8') as f:
                        f.write(f"解答: {answer}\n")
                        st.success(f"问题和答案已上传")
                    time.sleep(1)  # 等待处理完成           

            # 假设处理完后生成新的视频文件和字幕文件
            # 这里你可以添加实际的处理逻辑，例如调用本地处理脚本
            # 示例代码假设处理后的文件已经存在于指定路径

            if os.path.exists(file_path2):
                # 读取文件内容并显示在新的文本组件中
                st.write("### 解答")
                with open(file_path2, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                st.write(file_content)

        except Exception as e:
            st.error(f"保存文件失败: {e}")

    # 底部注释
    st.markdown("---")
    st.markdown("# 广告界面")
    st.markdown("""
    携手共创辉煌，共赢未来市场！如果您对我们的项目感兴趣，我们诚挚邀请您加入我们Go Commentary的开发中，共同打造更加出色的品牌！

    联系方式：100100011
    """)
