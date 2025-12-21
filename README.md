# VideoAgent-simple
CS4316 Final Project

## Milestone 1
### 25/12/15
熟悉了一下 langchain 的基本操作，然后在 demo_multimodel_stream.py 中借助 AI agent 实现了最基本的根据输入的问题进行画面描述功能。整体结构看上去复杂而繁琐，也并没有好好利用 tools 进行灵活的工具调用。之后会更改整体代码架构使之更为合理，目前就先这样吧。
ps：视频放在 test_videos 文件夹下，因为太大没有传上来。

## Milestone 2
### 25/12/21
在 Milestone 1 的基础上，重新（让 AI）设计了整体代码架构，使之更为合理和模块化。新增了多种工具函数，支持更多样化的视频理解任务，包括视频时间线生成、场景检测、时刻检索等。
添加了 test_task.py 用于测试各个工具函数的功能正确性。