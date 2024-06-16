# -*- coding: utf-8 -*-
from roboflow import Roboflow
import cv2
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import os
import json


configs = json.loads(open("./configs/config.json").read())
rf = Roboflow(api_key=configs['go_detect_model_api_key'])
project = rf.workspace().project("go-positions")
go_detect_model = project.version(4).model

#对棋盘区域截图
def clipImg(rect, image):
    # 计算变换后的宽度和高度
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 目标矩形的顶点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)

    # 应用透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


# 找到图片中棋盘区域
def findClip(image):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 形态学操作，连接断开的边缘
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 假设最大的轮廓是棋盘
    contour = max(contours, key=cv2.contourArea)

    # cv2.imshow("Cropped Chessboard", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 近似多边形
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 如果近似多边形的顶点数为4，认为找到了棋盘
    if len(approx) == 4:
        # 获取顶点
        pts = approx.reshape(4, 2)

        # 对顶点进行排序，顺序为：[top-left, top-right, bottom-right, bottom-left]
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    else:
        print("棋盘检测失败，未能找到棋盘的四个顶点。")


# 抽关键帧帧并切割棋盘区域图像保存
def extract_key_frames(video_path, threshold=30, interval=30, output_folder='./keyframes/'):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read video file.")
        return

        # 存储第一帧
    frame_count = 1
    clipArea = findClip(prev_frame)
    clipPrev_frame = clipImg(clipArea, prev_frame)
    filename = f"{output_folder}/{frame_count:04d}.jpg"
    cv2.imwrite(filename, clipPrev_frame)
    # cv2.imwrite("r2.jpg", prev_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % interval == 0:

            # 计算均方误差（MSE）
            clipframe = clipImg(clipArea, frame)
            mse = np.mean((clipframe.astype("float") - clipPrev_frame.astype("float")) ** 2, axis=(0, 1))
            mse_value = np.mean(mse)  # 计算RGB三个通道的MSE平均值
            print(mse_value)

            # 如果差异超过阈值，保存这一帧
            if mse_value > threshold:
                filename = f"{output_folder}/{frame_count:04d}.jpg"
                cv2.imwrite(filename, clipframe)
                print(f"Mean Squared Error (MSE): {mse_value}")
                print(f"Saved keyframe: {filename}")

                # 更新前一帧
            clipPrev_frame = clipframe

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


#写入json文件
def WriteInJson(lst, write_path):
    json_string = json.dumps(lst)
    with open(write_path, 'w', encoding='utf-8') as file:
        file.write(json_string)


#计算棋盘间隔
def getMargin(result):
    img_width = result["image"]["width"]
    img_height = result["image"]["height"]
    board_len = (float(img_height) + float(img_width))/2
    return board_len/38


#棋盘行位转换
def row_pos_trans(pos):
    alphalst = ["A","B","C","D","E","F","G","H","J","K","L","M","N","O","P","Q","R","S","T"]
    numlst = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    if pos in numlst:
        idx = numlst.index(pos)
        return(alphalst[idx])
    return pos


#单步信息写入字典
def WriteInDic(Bp, Wp, n, t):
    data = {
        "step": n,
        "black": Bp,
        "white": Wp,
        "time": t
    }
    return data


# 计算棋子相对棋盘位置
def get_stone(prediction, Margin):
    x = prediction["x"]
    y = prediction["y"]
    n1 = Decimal(x / Margin)
    n2 = Decimal(y / Margin)
    x = n1.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    y = n2.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    n3 = Decimal((x + 1) / 2)
    tpx = n3.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    s1 = row_pos_trans(tpx)
    s2 = 20 - int((y + 1) / 2)
    Class = prediction["class"]
    info = [s1, s2, Class]

    return (info)


# 获取棋盘所有棋子位置信息
def goPositionPredict(img_path, step, t):
    Result = go_detect_model.predict(img_path, confidence=45, overlap=50).json()
    Margin = getMargin(Result)
    black_pos = []
    white_pos0 = []
    white_pos = []

    for i in Result["predictions"]:
        tmp = get_stone(i, Margin)
        if (tmp[2] == 'blackStone'):
            black_pos.append(tmp[0] + str(tmp[1]))
        if (tmp[2] == 'whiteStone'):
            white_pos0.append(tmp[0] + str(tmp[1]))

    for i in white_pos0:
        if i not in black_pos:
            white_pos.append(i)

    return WriteInDic(black_pos, white_pos, step, t)


def image2text():
    ### Video to Go Position
    # specified_video_name = "test3.mp4"
    video_path = configs['video_path']
    folder_path = configs['folder_go_list_path']
    write_path = configs['write_go_list_path']
    
    # Video extraction
    extract_key_frames(video_path, output_folder=folder_path)
    dic_list = []
    for root, dirs, files in os.walk(folder_path):
        step = 1
        for name in files:
            img_path = os.path.join(root, name)
            dic_list.append(goPositionPredict(img_path, step, name[0:4]))

            step = step + 1

    # Write to json
    WriteInJson(dic_list, write_path)
    

if __name__ == '__main__':
    ### Video to Go Position
    # specified_video_name = "test3.mp4"
    video_path = configs['video_path']
    folder_path = configs['folder_go_list_path']
    write_path = configs['write_go_list_path']
    
    # Video extraction
    extract_key_frames(video_path, output_folder=folder_path)
    dic_list = []
    for root, dirs, files in os.walk(folder_path):
        step = 1
        for name in files:
            img_path = os.path.join(root, name)
            dic_list.append(goPositionPredict(img_path, step, name[0:4]))

            step = step + 1

    # Write to json
    WriteInJson(dic_list, write_path)
