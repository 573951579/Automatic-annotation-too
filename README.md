# SAM-Labelimg
利用Segment Anything(SAM)模型进行快速标注
本项目基于修改 https://github.com/zhouayi/SAM-Tool 库 github开源
https://github.com/facebookresearch/segment-anything 库 也为开源项目
此项目始终免费服务于广大数据标注、yolo等有关爱好者，所有和此项目相关的使用、维护、更新、传播、二次开发（其他博主的二次开发除外）等服务不会收取任何费用，如有索取关注、加群等引流、盈利性行为果断逃离
####项目部署与使用（使用者）
#### 1.下载项目
###1.1 源参考项目1：https://github.com/zhouayi/SAM-Tool -----可以不git，感兴趣的可以做参考
####   up主项目1：                                      -----视频中项目
###1.2 项目2：https://github.com/facebookresearch/segment-anything  -----sam1基本库
```bash
git clone 
###
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
###python环境下import segment_anything不报错则安装成功
```
####1.3
下载`SAM`模型：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  （2.6G）
####1.4
非本项目环境缺失如torch、PyQt5等有关缺失、错误自行检索有关教程

#### 2.把数据放置在`<dataset_path>/images/*`这个路径中，并创建空文件夹`<dataset_path>/embeddings`
####  xavier文件夹中为测试图片数据集
#### 此项目相较于源项目，已删去手动生成npy文件步骤


##### 3.2 运行`generate_onnx.py`将`pth`文件转换为`onnx`模型文件

```bash
# cd到项目1的主目录下
python helpers\generate_onnx.py --checkpoint-path sam_vit_h_4b8939.pth --onnx-model-path ./sam_onnx.onnx --orig-im-size 1080 1920
```

- `checkpoint-path`：同样的`SAM`模型路径代码中已添加相对路径

- `onnx-model-path`：得到的`onnx`模型保存路径

- `orig-im-size`：数据中图片的尺寸大小`（height, width）`

【**注意1：提供给的代码转换得到的`onnx`模型并不支持动态输入大小，所以如果你的数据集中图片尺寸不一，那么可选方案是以不同的`orig-im-size`参数导出不同的`onnx`模型供后续使用**】
【**注意2：onnx模型的生成每台电脑配置还有python环境不同可能无法完全适配，onnx生成如有问题可自行检索onnx有关解答**】
#### 4.生成的`sam_onnx.onnx`模型应该会自动保存到项目1的主目录下，运行`segment_anything_annotator.py`进行标注

```bash
# cd到项目1的主目录下
python segment_anything_annotator.py
```
在对象位置出点击鼠标左键为增加掩码，点击右键为去掉该位置掩码。
其他使用快捷键有：

| `Esc`：退出     | `a`：前一张图片| `d`：下一张图片 |
| `r`  ： 重置    | `f`：打框      | `Ctrl+s`：保存 |
| `Ctrl+z`：撤销上一次|


最后生成的标注文件为`coco`格式，保存在`<dataset_path>/annotations.json`。
手动导出两种格式yolov8.txt、yolov11-obb.txt 
yolov8.txt为所有yolo系列基础训练使用
yolov11-obb.txt 为yolov11之后有关旋转估计系列（obb）基础训练使用
训练时如有txt数据集使用有关问题反馈给我以及时修复有关bug

####项目库解释（开发者）
./datasets 数据集文件夹
./helpers 有关extract_embeddings文件夹（已附属到主函数串行运行） 、sam-onnx文件生成
./labels txt标注文件生成路径
./xavier 参考标注图片
./salt 使用工具库
    ./dataset_explorer.py 有关json文件读取、写入
    ./display_utils.py  有关标注画面中框的绘制
    ./editor.py 主系统、ui、标注交互和数据集成
    ./interface.py ui设计与逻辑类
    ./json2txt.py json转txt函数
    ./onnx_model.py onnx模型调用
    ./utils.py 服务于onnx
./cocoviewer.py json文件查看
./environment.yaml 推荐环境配置
./LICENSE 使用许可（非本项目）
./README.md 使用建议

## Reference
https://github.com/facebookresearch/segment-anything 

https://github.com/anuragxel/salt

https://github.com/trsvchn/coco-viewer
