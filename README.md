# VideoAgent-simple
CS4316 Final Project

## 项目结构
```
VideoAgent/
├── milestone1.py          # 基础视频问答
├── milestone2.py          # 模块化工具设计
├── milestone3.py          # 分层搜索 + CLIP 增强
├── test/                  # 测试脚本
│   ├── run_benchmark.py       # 统一测试入口
│   ├── test_classification.py # 分类能力测试
│   └── test_grounding.py      # 时刻定位测试
├── test_results/          # 测试结果输出
├── datasets/              # 数据集 (UCF101, HMDB51, Charades)
└── test_videos/           # 测试视频
```

## Milestone 1
### 25/12/15
熟悉了一下 langchain 的基本操作，然后在 demo_multimodel_stream.py 中借助 AI agent 实现了最基本的根据输入的问题进行画面描述功能。整体结构看上去复杂而繁琐，也并没有好好利用 tools 进行灵活的工具调用。之后会更改整体代码架构使之更为合理，目前就先这样吧。
ps：视频放在 test_videos 文件夹下，因为太大没有传上来。

## Milestone 2
### 25/12/21
在 Milestone 1 的基础上，重新（让 AI）设计了整体代码架构，使之更为合理和模块化。新增了多种工具函数，支持更多样化的视频理解任务，包括视频时间线生成、场景检测、时刻检索等。
添加了 test_task.py 用于测试各个工具函数的功能正确性。

## Milestone 3
### 25/12/29
实现了两项核心改进，并构建了标准化的测试评估框架：

### 📊 Benchmark 测试框架
测试脚本位于 `test/` 目录下，结果输出到 `test_results/`

- **视频分类测试** (`test/test_classification.py`)：基于 UCF-101 和 HMDB51 数据集的 10 个共有类别
  - Multiple Choice：多选题形式，VLM 直接选择答案
  - LLM-As-Judger：VLM 描述 + LLM 匹配判断
- **视频时刻定位测试** (`test/test_grounding.py`)：基于 Charades-STA 数据集（150 样本）
  - 评估指标：Mean IoU, R@0.3, R@0.5, R@0.7
- **统一测试入口** (`test/run_benchmark.py`)：一键运行所有测试

```bash
# 运行方式
cd test
python run_benchmark.py --task all
python test_classification.py --method multiple_choice --samples 3
python test_grounding.py --samples 50
```

### 🔧 Moment Retrieval 增强
1. **分层搜索 (Hierarchical Search)**
   - 三阶段由粗到细：30s 粗粒度 → 10s 中粒度 → 精确定位
   - 适用于中长视频，减少 VLM 调用次数，提高效率

2. **CLIP 特征搜索**
   - 视频帧 CLIP 特征提取与缓存
   - 文本-视频相似度快速匹配
   - 混合检索：CLIP 预筛选 + VLM 精确验证

### 🛠️ 新增工具函数
| 工具名 | 功能描述 |
|--------|----------|
| `hierarchical_moment_retrieval` | 分层递进式时刻检索 |
| `clip_moment_search` | 纯 CLIP 特征快速搜索 |
| `clip_guided_moment_retrieval` | CLIP 引导 + VLM 验证混合检索 |