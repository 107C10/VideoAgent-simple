"""
视频问答Agent - 能够回答"视频第X秒展示了什么画面？"
使用LangChain构建，基于多模态大模型

使用 @tool 装饰器 + create_agent 实现自动工具调用
"""

import os
import cv2
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# ============ 环境配置 ============
# os.environ["OPENAI_API_KEY"] = "your_api_key_here"
# os.environ["OPENAI_API_BASE"] = "your_api_base_here"

# Agent主模型（支持 function call 的文本模型）
AGENT_MODEL = "Qwen/Qwen3-32B"
# 视觉分析模型（多模态 VLM）
VL_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"

# 初始化模型
llm = ChatOpenAI(model=AGENT_MODEL, temperature=0)
vl_llm = ChatOpenAI(model=VL_MODEL, temperature=0)

# 默认视频路径
DEFAULT_VIDEO_PATH = "test_videos/Truman.MP4"


# ============ Tool 1: 解析秒数 ============
@tool
def get_second(user_query: str) -> float:
    """
    从用户的问题中提取秒数。
    当用户询问视频某个时间点的画面时，首先调用此工具提取秒数。
    
    Args:
        user_query: 用户的原始问题，例如"视频第15秒展示了什么？"
    
    Returns:
        提取出的秒数（浮点数），例如 15.0
    """
    print(f"[Tool 1] get_second: 正在解析问题中的秒数...")
    
    messages = [
        SystemMessage(content="请你找到里面表示秒数的数字，直接返回数字结果，不要额外的输出。如果有小数也保留。只输出数字。"),
        HumanMessage(content=user_query)
    ]
    
    response = llm.invoke(messages)
    result = response.content.strip()
    
    print(f"[Tool 1] 解析得到秒数: {result}")
    
    # 提取数字
    import re
    match = re.search(r'[\d.]+', result)
    if match:
        return float(match.group())
    
    raise ValueError(f"无法从'{user_query}'中提取秒数")


# ============ Tool 2: 提取视频帧 ============
@tool
def get_frame_at_second(second: float, video_path: str = DEFAULT_VIDEO_PATH) -> str:
    """
    根据指定的秒数从视频中提取对应的帧图片，并保存到本地。
    在获取秒数后调用此工具提取视频帧。
    
    Args:
        second: 要提取帧的时间点（秒），例如 10.0
        video_path: 视频文件路径，默认为 test_videos/Truman.MP4
    
    Returns:
        保存的图片文件路径
    """
    print(f"[Tool 2] get_frame_at_second: 正在提取第 {second} 秒的画面...")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"[Tool 2] 视频信息: FPS={fps:.2f}, 时长={duration:.2f}秒")
    
    if second < 0 or second > duration:
        cap.release()
        raise ValueError(f"秒数 {second} 超出视频范围 (0 - {duration:.2f}秒)")
    
    # 计算帧位置并跳转
    frame_number = int(second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"无法读取第 {second} 秒的帧")
    
    # 保存图片
    output_path = f"images/frame_at_{second}s.jpg"
    os.makedirs("images", exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"[Tool 2] 帧已保存到: {output_path}")
    
    return output_path


# ============ Tool 3: 分析图片内容 ============
@tool
def analyze_frame(image_path: str, question: str = "请详细描述这张图片中的画面内容") -> str:
    """
    使用多模态大模型分析图片内容。
    在获取视频帧后调用此工具分析画面。
    
    Args:
        image_path: 图片文件路径
        question: 关于图片的问题，例如"请描述画面内容"
    
    Returns:
        模型对图片的分析结果
    """
    print(f"[Tool 3] analyze_frame: 正在分析画面内容...")
    
    # 读取并编码图片
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
    
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        )
    ]
    
    response = vl_llm.invoke(messages)
    print(f"[Tool 3] 分析完成")
    
    return response.content


# ============ 创建Agent ============
def create_video_qa_agent():
    """创建视频问答Agent"""
    tools = [get_second, get_frame_at_second, analyze_frame]
    
    memory = MemorySaver()
    
    system_prompt = """你是一个专业的视频内容分析助手。当用户询问视频某个时间点的画面内容时，你需要按顺序调用以下工具：

1. get_second: 从用户问题中提取具体的秒数
2. get_frame_at_second: 获取该时间点的视频帧图片
3. analyze_frame: 分析这张图片的内容

请依次调用这三个工具，并将最终的图片分析结果返回给用户。回答时请用中文。"""
    
    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        system_prompt=system_prompt
    )
    
    return agent


# ============ 主函数 ============
def main():
    print("=" * 60)
    print("视频问答Agent启动")
    print(f"Agent模型: {AGENT_MODEL}")
    print(f"视觉模型: {VL_MODEL}")
    print(f"默认视频: {DEFAULT_VIDEO_PATH}")
    print("=" * 60)
    
    # 创建Agent
    agent = create_video_qa_agent()
    
    # 测试问题
    test_questions = [
        "视频第153秒展示了什么画面？",
    ]
    
    config = {"configurable": {"thread_id": "video_qa_thread"}}
    
    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 50)
        
        inputs = {
            "messages": [HumanMessage(content=question)]
        }
        
        print("--- Agent 开始运行 ---")
        for chunk in agent.stream(inputs, config, stream_mode="values"):
            message = chunk["messages"][-1]
            content = message.content if message.content else "[调用工具中...]"
            print(f"[{message.type}]: {content}")
            print("-" * 30)
        
        print("=" * 60)


# ============ 交互式模式 ============
def interactive_mode():
    """交互式问答模式"""
    print("=" * 60)
    print("视频问答Agent - 交互模式")
    print(f"Agent模型: {AGENT_MODEL}")
    print(f"视觉模型: {VL_MODEL}")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    agent = create_video_qa_agent()
    
    config = {"configurable": {"thread_id": "interactive_thread"}}
    
    while True:
        question = input("\n请输入问题: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        
        if not question:
            continue
        
        inputs = {
            "messages": [HumanMessage(content=question)]
        }
        
        print("\n--- Agent 开始运行 ---")
        for chunk in agent.stream(inputs, config, stream_mode="values"):
            message = chunk["messages"][-1]
            content = message.content if message.content else "[调用工具中...]"
            print(f"[{message.type}]: {content}")
            print("-" * 30)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        # 交互模式: python ChatGPTexample.py -i
        interactive_mode()
    else:
        # 测试模式
        main()
