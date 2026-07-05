# 视频检索与中文检索升级说明

本轮新增两个能力：

- 视频素材上传、抽帧、CLIP 向量索引与检索。
- 中文查询增强：对象词、颜色词、动作词自动改写为英文视觉 prompt。

## 视频检索方案

当前实现是轻量级方案：

```text
视频文件 -> 抽取 8 帧 -> 每帧 CLIP 图像向量 -> 平均池化 -> 视频语义向量
```

优点：

- 工程简单。
- 可以复用现有图片检索链路。
- 对短视频、素材片段、场景类视频检索有效。

局限：

- 不理解复杂时序动作。
- 不做音频和字幕。
- 长视频只靠少量帧，可能漏掉关键片段。

## 已验证样例

本地生成了两个测试视频：

```text
data/video_samples/red_square_motion.mp4
data/video_samples/blue_sky_grass_motion.mp4
data/video_samples/green_block_motion.mp4
data/video_samples/yellow_ball_motion.mp4
```

已导入素材库：

```text
red_square_motion.mp4
blue_sky_grass_motion.mp4
green_block_motion.mp4
yellow_ball_motion.mp4
```

测试结果：

```text
查询：红色方块移动的视频
Top1：red_square_motion.mp4

查询：蓝天草地的视频
Top1：blue_sky_grass_motion.mp4

查询：a video of grass and sky
Top1：blue_sky_grass_motion.mp4
```

## 中文检索增强

新增识别类型：

- 对象词：狗、猫、公交车、汽车、摩托车、飞机、披萨、香蕉、人、孩子等。
- 场景词：草地、蓝天、夜景、建筑、风景等。
- 颜色词：红色、蓝色、绿色、黄色、白色、黑色。
- 动作词：移动、跑、行驶、骑、飞、吃。

示例：

```text
狗在草地上
-> a photo of dog
-> a photo of dog in grass field

公交车在街道上
-> a photo of bus
-> a photo of bus at street

红色方块移动的视频
-> a video of red moving square
-> a photo of red moving square
```

注意：只有查询中明确出现“视频/短视频/片段”时，Agent 才会生成 video prompt，避免普通图片查询被视频素材干扰。

## 推荐视频数据集

### UCF101

适合动作识别和视频分类，包含 101 类动作、超过 1.3 万个视频片段。适合后续测试“骑车、打球、跑步”等动作类检索。

参考：UCF101 论文介绍了 101 类动作和超过 13k clips 的数据规模。来源见 UCF101 paper。  
https://arxiv.org/abs/1212.0402

### VATEX

适合中英视频描述和多语言视频检索。它包含英文和中文 caption，适合我们后续升级中文视频检索。

参考：VATEX paper 提到其包含超过 41,250 个视频和中英文 captions。  
https://arxiv.org/abs/1904.03493

### MSR-VTT

适合文本到视频检索和视频描述任务，但标注质量存在一定噪声。可作为后续评测集。

参考：MSR-VTT 常用于 video captioning / text-to-video retrieval。  
https://arxiv.org/abs/2102.06448

## 下一步建议

1. 下载 UCF101 的少量类别子集，例如 dog / horse / biking / basketball / cooking 类别，用于动作检索演示。
2. 接入 PaddleOCR/EasyOCR 处理视频关键帧文字。
3. 接入 Whisper 处理视频音频。
4. 把一个视频拆成多个 clip/frame 索引，而不是只保存一个平均向量。
5. 引入 Chinese-CLIP 或中文视频文本模型，提高中文视频检索稳定性。
