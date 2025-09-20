# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache-2.0 license found in the LICENSE file in the root directory of segment_anything repository and source tree.
# Adapted from onnx_model_example.ipynb in the segment_anything repository.
# Please see the original notebook for more details and other examples and additional usage.
import os
import argparse
import cv2
from tqdm import tqdm
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default="/home/west-dj/obb/sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type", type=str, default="default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset-folder", type=str, default="./example_dataset")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    model_type = args.model_type
    device = args.device
    dataset_folder = args.dataset_folder

    images_folder = os.path.join(dataset_folder, "images")
    embeddings_folder = os.path.join(dataset_folder, "embeddings")
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    main(checkpoint_path, model_type, device, images_folder, embeddings_folder)
