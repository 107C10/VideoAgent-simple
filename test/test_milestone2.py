# -*- coding: utf-8 -*-
"""
测试 Milestone 2 的各项功能
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
import milestone2

def test_task(question, task_name):
    """测试单个任务"""
    print("=" * 70)
    print(f"测试: {task_name}")
    print(f"问题: {question}")
    print("=" * 70)
    
    agent = milestone2.create_video_qa_agent()
    config = {"configurable": {"thread_id": f"test_{task_name}"}}
    inputs = {"messages": [HumanMessage(content=question)]}
    
    print("\n--- Agent 开始运行 ---")
    for chunk in agent.stream(inputs, config, stream_mode="values"):
        message = chunk["messages"][-1]
        content = message.content if message.content else "[调用工具中...]"
        print(f"[{message.type}]: {content}")
        print("-" * 50)
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    # 获取测试编号
    test_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    if test_num == 2:
        # Task 2: 视频 Captioning（时间范围）
        test_task("总结30秒到60秒的视频内容", "Task2_Captioning")
    
    elif test_num == 3:
        # Task 3: 完整时间线
        test_task("生成视频的完整时间线", "Task3_Timeline")
    
    elif test_num == 4:
        # Task 4: Moment Retrieval
        test_task("找到视频中两个老太太的片段，以及主角鞠躬的片段", "Task4_MomentRetrieval")
    
    else:
        print("用法: python test_milestone2.py <task_number>")
        print("  2 - 测试视频Captioning（时间范围）")
        print("  3 - 测试完整时间线")
        print("  4 - 测试Moment Retrieval")
