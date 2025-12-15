import os

from langchain_core.prompts import ChatPromptTemplate

# 设置 API 密钥
os.environ["OPENAI_API_KEY"] = "sk-yuknxsbirvgfjucumekpjaytgbgsvgvgdyztihhcqmtwlafu"
os.environ["OPENAI_API_BASE"] = "https://api.siliconflow.cn/v1"
os.environ["TAVILY_API_KEY"] = "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"

# 导入 langchain 相关包
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch

import cv2
@tool
def get_frame_at_second(second: float, video_path: str = "./test_videos/Truman.MP4", output_path: str = "./images/frame.jpg"):
    """
    读取视频并在指定秒数截取一帧保存。
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "Error: Could not open video."

    # 计算帧的位置
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = int(fps * second)

    # 设置当前帧位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
        cap.release()
        return f"Success: Frame at {second}s saved to {output_path}"
    else:
        cap.release()
        return "Error: Could not read frame."

vlm = ChatOpenAI(model="Qwen/Qwen2.5-32B-Instruct")
# memory = MemorySaver()
tools = [get_frame_at_second]
second_template = """
请你从下面的问题中找到里面表示秒数的数字，直接返回数字结果，不要额外的输出。
问题：{question}
"""
prompt = ChatPromptTemplate.from_template(second_template)
# question = input("想要了解视频里第几秒的画面？尽管提问：")
question = "视频第29秒展示了什么画⾯？"
messages = prompt.format_messages(question=question)

agent = create_agent(
    model=vlm,
    tools=tools,
    system_prompt="You are a assistant who helps people extract frames from videos based on their questions."
)

# response = agent.invoke(messages)
# print(response.content)
inputs = {"messages": messages}

config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent.stream(inputs, config):
    print(chunk)
    print("-----chunk break-----")
    # for step, data in chunk.items():
    #     print(f"step: {step}")
    #     print(f"content: {data['messages'][-1].content_blocks[-1].type()}")
    #

