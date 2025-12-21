import os
import base64
import cv2
from langchain_classic.agents import Agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

def encode_image(image_path):
    """将图片转换为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@tool
def analyze_video_at_second(second: float, query: str, model: Agent):
    """
    读取视频并在指定秒数截取一帧，然后根据问题描述画面内容。
    当用户询问视频某时刻的画面时，必须调用此工具。
    参数:
    - second: 视频的秒数 (例如 29.0)
    - query: 关于画面的问题 (例如 "画面里有什么？")
    - model: 自己使用多模态来分析图片
    """
    # 假设视频路径固定
    video_path = os.path.join(os.path.dirname(__file__), "test_videos", "Truman.MP4")
    output_path = "frame_agent.jpg"

    if not os.path.exists(video_path):
        return f"Error: Video file not found at {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video."

    # 计算帧的位置
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = int(fps * second)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return f"Error: Could not read frame at {second}s."

    cv2.imwrite(output_path, frame)

    # 使用 VLM 模型分析图片
    try:
        base64_image = encode_image(output_path)

        msg = HumanMessage(content=[
            {"type": "text", "text": query},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])

        # 调用 VLM 生成描述
        response = model.invoke([msg])
        return f"Frame at {second}s analysis: {response.content}"
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def main():
    # 1. 定义主 Agent 模型 (负责思考和调用工具)
    # 使用支持工具调用的文本模型
    main_llm = ChatOpenAI(model="Qwen/Qwen2.5-32B-Instruct")

    # 2. 定义工具列表
    tools = [analyze_video_at_second]

    # 3. 创建 ReAct Agent (使用 LangGraph)
    memory = MemorySaver()
    # 注意：不同版本的 langgraph 参数可能不同，这里使用最通用的方式，将 system prompt 放入 input messages
    agent = create_agent(
        model=main_llm,
        tools=tools,
        checkpointer=memory
    )

    # 4. 用户提问
    question = input("请以类似格式提问：视频第29秒展示了什么画面？\n")
    print(f"用户提问: {question}")

    # 将 System Prompt 作为第一条消息传入
    system_prompt = "你是一个视频助手。当用户问及视频某秒的内容时，请使用 analyze_video_at_second 工具来查看并回答。其中 model 参数为你自己"
    inputs = {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=question)]}

    config = {"configurable": {"thread_id": "demo_thread_3"}}

    print("--- Agent 开始运行 ---")
    # stream_mode="values" 会返回每一步的消息列表
    for chunk in agent.stream(inputs, config, stream_mode="values"):
        # 获取最新的一条消息
        message = chunk["messages"][-1]
        # 打印消息类型和内容
        print(f"[{message.type}]: {message.content}")
        print("-----------------------")

if __name__ == "__main__":
    main()
