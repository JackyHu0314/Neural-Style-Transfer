# models

这里用于放可选的本地模型权重，例如：

```text
vgg19.pth
```

当前版本默认使用 `torchvision` 的 VGG19 预训练权重。如果下载失败，可以把权重文件放到本目录，并通过 `--vgg-weights` 指定。

