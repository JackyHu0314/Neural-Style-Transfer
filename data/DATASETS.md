# Training Dataset Plan

训练图片风格化模型时，建议使用两类数据集：

```text
内容图数据集：真实照片，负责学习保留图像结构
风格图数据集：艺术画作，负责学习颜色、纹理、笔触
```

## 推荐组合

### 1. 内容图：MS COCO 2017

用途：作为 content images。

官方页面：

```text
https://cocodataset.org/#download
https://cocodataset.org/dataset/detection-2017.htm
```

常用下载：

```text
2017 train images: http://images.cocodataset.org/zips/train2017.zip
2017 val images:   http://images.cocodataset.org/zips/val2017.zip
```

建议：

```text
先下 val2017.zip，大约 5k 张图，约 1GB，适合先验证训练代码。
训练稳定后再下 train2017.zip，大约 118k 张图，约 18GB。
```

风格迁移只需要图片本身，不需要 COCO 的检测标注。

### 2. 风格图：WikiArt

用途：作为 style images。

可选来源：

```text
Kaggle WikiArt:
https://www.kaggle.com/datasets/steubk/wikiart

Hugging Face WikiArt 81K:
https://huggingface.co/datasets/Dant33/WikiArt-81K-BLIP_2-1024x1024

Hugging Face huggan/wikiart:
https://huggingface.co/datasets/huggan/wikiart
```

建议：

```text
如果只是课程实验，先用 1k-5k 张 WikiArt 风格图训练。
如果要完整训练，再使用 80k 级别的完整 WikiArt。
```

注意：WikiArt 常见数据集一般只适合非商业研究或课程实验，报告里应注明数据来源。

## 建议本地目录

```text
data/
├─ train/
│  ├─ content/
│  │  ├─ coco_val2017/
│  │  └─ coco_train2017/
│  └─ style/
│     └─ wikiart/
```

## 下载优先级

推荐按这个顺序来：

```text
第一阶段：COCO val2017 + 少量 WikiArt
第二阶段：COCO train2017 + WikiArt 子集
第三阶段：COCO train2017 + 完整 WikiArt
```

原因是你的当前 PyTorch 是 CPU 版，完整数据集会非常慢。先用小一点的数据验证训练流程，后面再扩大数据量更稳。

