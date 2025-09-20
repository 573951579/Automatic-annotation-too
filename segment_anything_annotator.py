import os
import argparse
import sys
from PyQt5.QtWidgets import QApplication
from salt.editor import Editor
from salt.interface import ApplicationInterface
from segment_anything import sam_model_registry, SamPredictor
import cv2
from tqdm import tqdm
import numpy as np


def main(checkpoint_path, model_type, device, images_folder, embeddings_folder):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 获取图像文件夹中所有文件，并过滤出常见图像格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_names = [
        name for name in os.listdir(images_folder)
        if name.lower().endswith(image_extensions)
    ]

    # 遍历图像文件，仅处理不存在embedding的文件
    for image_name in tqdm(image_names, desc="生成图像嵌入"):
        # 构建输入图像路径和输出embedding路径
        image_path = os.path.join(images_folder, image_name)
        base_name = os.path.splitext(image_name)[0]
        embedding_path = os.path.join(embeddings_folder, f"{base_name}.npy")

        # 检查embedding文件是否已存在，存在则跳过
        if os.path.exists(embedding_path):
            continue

        # 读取并预处理图像
        image = cv2.imread(image_path)
        if image is None:  # 处理图像读取失败的情况
            print(f"警告：无法读取图像 {image_path}，已跳过")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 生成并保存嵌入
        predictor.set_image(image)
        image_embedding = predictor.get_image_embedding().cpu().numpy()
        np.save(embedding_path, image_embedding)    

if __name__ == "__main__":
    #os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/west-dj/anaconda3/envs/py38/lib/python3.8/site-packages/PyQt5/Qt5/plugins" #虚拟环境使用pyqt5需要添加并修改此路径

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-model-path", type=str, default="./sam_onnx.onnx")
    parser.add_argument("--dataset-path", type=str, default="datasets")
    parser.add_argument("--checkpoint-path", type=str, default="./sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type", type=str, default="default")
    parser.add_argument("--device", type=str, default="cuda")
    

    args = parser.parse_args()
    print("如果本批图像初次标注需要删除上次残留的/SAM-Tool-main/datasets/路径下的embeddings文件夹与/annotations.json文件防止报错，并耐心等待embeddings的生成")
    #npy文件生成启动部分,如果本批图像初次标注需要删除上次残留的/SAM-Tool-main/datasets/路径下的embeddings文件夹与/annotations.json文件防止报错，并耐心等待embeddings的生成
    checkpoint_path = args.checkpoint_path
    model_type = args.model_type
    device = args.device
    dataset_folder = args.dataset_path
    images_folder = os.path.join(dataset_folder, "images")
    embeddings_folder = os.path.join(dataset_folder, "embeddings")
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)
    main(checkpoint_path, model_type, device, images_folder, embeddings_folder)
    print("快捷键：")
    print("esc：退出")
    print("A：上一张")
    print("D：下一张")
    print("F：打框")
    print("ctrl+Z：撤回上一个框")
    print("ctrl+S：保存内容")
    #标注启动部分
    onnx_model_path = args.onnx_model_path
    dataset_path = args.dataset_path

    coco_json_path = os.path.join(dataset_path,"annotations.json")
    categories=""

    editor = Editor(
        onnx_model_path,
        dataset_path,
        categories=categories,
        coco_json_path=coco_json_path
    )
    
    app = QApplication(sys.argv)
    window = ApplicationInterface(app, editor)
    window.show()
    sys.exit(app.exec_())
