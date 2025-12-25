"""
视频问答Agent - Milestone 2
功能扩展：
1. 视频分段 (Scene Detection) - 使用 PySceneDetect
2. 视频 Captioning - 对视频片段进行内容总结
3. 时间线总结 - 生成完整的视频时间线

使用 @tool 装饰器 + create_agent 实现自动工具调用
"""

import os
import cv2
import base64
from typing import List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from scenedetect import detect, ContentDetector

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

# 全局缓存场景列表
_cached_scenes = None
_cached_video_path = None


# ============ Task 1: 视频分段 ============
@tool
def detect_scenes(video_path: str = DEFAULT_VIDEO_PATH, threshold: float = 27.0) -> str:
    """
    使用 PySceneDetect 检测视频中的场景切换点，将视频分割成多个段落。
    场景切换通常发生在画面内容发生显著变化的时刻。
    
    Args:
        video_path: 视频文件路径
        threshold: 场景检测阈值，越低检测越敏感，默认27.0
    
    Returns:
        检测到的场景列表，包含每个场景的起止时间
    """
    global _cached_scenes, _cached_video_path
    
    print(f"[Tool] detect_scenes: 正在检测视频场景...")
    print(f"[Tool] 视频路径: {video_path}, 阈值: {threshold}")
    
    # 检测场景
    scene_list = detect(video_path, ContentDetector(threshold=threshold))
    
    # 缓存结果
    _cached_scenes = scene_list
    _cached_video_path = video_path
    
    # 格式化输出
    result_lines = [f"检测到 {len(scene_list)} 个场景：\n"]
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_timecode()
        end_time = scene[1].get_timecode()
        start_sec = scene[0].get_seconds()
        end_sec = scene[1].get_seconds()
        duration = end_sec - start_sec
        result_lines.append(
            f"场景 {i+1}: {start_time} - {end_time} (时长: {duration:.1f}秒)"
        )
    
    result = "\n".join(result_lines)
    print(f"[Tool] detect_scenes: 完成，共检测到 {len(scene_list)} 个场景")
    
    return result


@tool
def get_scene_info(scene_index: int) -> str:
    """
    获取指定场景的详细信息，包括起止时间和帧号。
    需要先调用 detect_scenes 进行场景检测。
    
    Args:
        scene_index: 场景编号（从1开始）
    
    Returns:
        场景的详细信息
    """
    global _cached_scenes
    
    if _cached_scenes is None:
        return "错误：请先调用 detect_scenes 检测场景"
    
    if scene_index < 1 or scene_index > len(_cached_scenes):
        return f"错误：场景编号 {scene_index} 超出范围 (1-{len(_cached_scenes)})"
    
    scene = _cached_scenes[scene_index - 1]
    start_time = scene[0].get_timecode()
    end_time = scene[1].get_timecode()
    start_frame = scene[0].frame_num
    end_frame = scene[1].frame_num
    start_sec = scene[0].get_seconds()
    end_sec = scene[1].get_seconds()
    
    return (f"场景 {scene_index} 详情:\n"
            f"  起始: {start_time} (帧 {start_frame}, {start_sec:.2f}秒)\n"
            f"  结束: {end_time} (帧 {end_frame}, {end_sec:.2f}秒)\n"
            f"  时长: {end_sec - start_sec:.2f}秒")


# ============ Task 2: 视频 Captioning（指定时间范围总结） ============
@tool
def caption_video_segment(start_second: float, end_second: float, video_path: str = DEFAULT_VIDEO_PATH) -> str:
    """
    对指定时间范围的视频内容进行总结（Captioning）。
    会从该时间范围内均匀提取多个关键帧，使用多模态大模型分析并生成内容描述。
    
    Args:
        start_second: 起始秒数，例如 30.0
        end_second: 结束秒数，例如 60.0
        video_path: 视频文件路径
    
    Returns:
        该时间范围内视频内容的详细描述
    """
    print(f"[Tool] caption_video_segment: 正在总结 {start_second}s - {end_second}s 的视频内容...")
    
    if start_second >= end_second:
        return f"错误：起始时间 {start_second}s 必须小于结束时间 {end_second}s"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"错误：无法打开视频文件 {video_path}"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    if start_second < 0 or end_second > duration:
        cap.release()
        return f"错误：时间范围超出视频长度 (0 - {duration:.2f}秒)"
    
    # 根据时间范围长度决定提取帧数
    segment_duration = end_second - start_second
    if segment_duration <= 5:
        num_frames = 3
    elif segment_duration <= 15:
        num_frames = 5
    elif segment_duration <= 30:
        num_frames = 8
    else:
        num_frames = 10
    
    # 计算均匀分布的帧位置
    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps)
    step = (end_frame - start_frame) / (num_frames + 1)
    frame_positions = [int(start_frame + step * (i + 1)) for i in range(num_frames)]
    
    # 读取并编码图片
    image_contents = []
    frame_times = []
    for frame_num in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
            frame_times.append(frame_num / fps)
    
    cap.release()
    
    if not image_contents:
        return "错误：无法提取视频帧"
    
    # 构建多图消息，请求详细描述
    time_info = ", ".join([f"{t:.1f}s" for t in frame_times])
    content = [
        {"type": "text", "text": f"""这是视频中 {start_second:.1f}秒 到 {end_second:.1f}秒 时间段的 {len(image_contents)} 个关键帧（分别位于 {time_info}）。

请详细描述这段视频的内容，包括：
1. 场景环境描述
2. 出现的人物及其特征
3. 发生的动作和事件
4. 整体叙事或情节发展

请用中文回答，尽可能详细完整。"""}
    ] + image_contents
    
    messages = [HumanMessage(content=content)]
    response = vl_llm.invoke(messages)
    
    result = f"📹 视频片段总结 ({start_second:.1f}s - {end_second:.1f}s, 时长{segment_duration:.1f}秒):\n\n{response.content}"
    print(f"[Tool] caption_video_segment: 总结完成")
    
    return result


@tool
def extract_time_range_frames(start_second: float, end_second: float, num_frames: int = 5, video_path: str = DEFAULT_VIDEO_PATH) -> str:
    """
    从指定时间范围内均匀提取多个关键帧图片并保存。
    
    Args:
        start_second: 起始秒数
        end_second: 结束秒数
        num_frames: 要提取的帧数，默认5帧
        video_path: 视频文件路径
    
    Returns:
        提取的帧图片路径列表
    """
    print(f"[Tool] extract_time_range_frames: 正在从 {start_second}s - {end_second}s 提取 {num_frames} 帧...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"错误：无法打开视频文件 {video_path}"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps)
    
    step = (end_frame - start_frame) / (num_frames + 1)
    frame_positions = [int(start_frame + step * (i + 1)) for i in range(num_frames)]
    
    saved_paths = []
    os.makedirs("images", exist_ok=True)
    
    for i, frame_num in enumerate(frame_positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            second = frame_num / fps
            output_path = f"images/range_{start_second}-{end_second}_frame{i+1}_{second:.1f}s.jpg"
            cv2.imwrite(output_path, frame)
            saved_paths.append(output_path)
    
    cap.release()
    
    result = ", ".join(saved_paths)
    print(f"[Tool] extract_time_range_frames: 已提取 {len(saved_paths)} 帧")
    
    return result


# ============ Task 3: 时间线总结（完整输出所有场景） ============
# 全局变量缓存时间线，供 Moment Retrieval 使用
_cached_timeline = None

@tool
def generate_timeline(video_path: str = DEFAULT_VIDEO_PATH) -> str:
    """
    生成视频的【完整】时间线总结。
    会自动检测场景，并为【每个场景】生成描述，一次性输出所有场景的时间线。
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        视频的完整时间线总结（包含所有场景）
    """
    global _cached_scenes, _cached_video_path, _cached_timeline
    
    print(f"[Tool] generate_timeline: 正在生成完整视频时间线...")
    
    # Step 1: 检测场景
    scene_list = detect(video_path, ContentDetector(threshold=27.0))
    _cached_scenes = scene_list
    _cached_video_path = video_path
    
    total_scenes = len(scene_list)
    print(f"[Tool] 检测到 {total_scenes} 个场景，将处理全部场景")
    
    # Step 2: 为【每个】场景生成描述（不做截断）
    timeline_entries = []
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    for i in range(total_scenes):
        scene = scene_list[i]
        start_sec = scene[0].get_seconds()
        end_sec = scene[1].get_seconds()
        start_frame = scene[0].frame_num
        end_frame = scene[1].frame_num
        
        print(f"[Tool] 正在处理场景 {i+1}/{total_scenes} ({start_sec:.1f}s - {end_sec:.1f}s)...")
        
        # 取场景中间帧
        mid_frame = (start_frame + end_frame) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # 调用VLM生成简短描述
            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": "用一句话简洁描述这个画面的主要内容（20字以内）："},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]
            
            response = vl_llm.invoke(messages)
            description = response.content.strip()
            
            timeline_entries.append(
                f"场景{i+1} [{start_sec:.1f}s - {end_sec:.1f}s]: {description}"
            )
    
    cap.release()
    
    # 汇总完整时间线
    result = f"📽️ 视频完整时间线总结 (共 {total_scenes} 个场景):\n\n"
    result += "\n".join(timeline_entries)
    
    # 缓存时间线供 Moment Retrieval 使用
    _cached_timeline = result
    
    print(f"[Tool] generate_timeline: 完整时间线生成完成，共 {total_scenes} 个场景")
    
    return result


# ============ Task 4: Moment Retrieval（视频时刻检索） ============
@tool
def moment_retrieval(query: str, video_path: str = DEFAULT_VIDEO_PATH) -> str:
    """
    根据自然语言描述查询视频中对应的时间范围（Moment Retrieval）。
    会先生成时间线（如果还没有），然后根据查询内容找到匹配的时间段。
    
    Args:
        query: 自然语言查询，例如"主角在吃饭的片段"、"出现日落的画面"
        video_path: 视频文件路径
    
    Returns:
        匹配的时间范围及相关描述
    """
    global _cached_timeline, _cached_scenes, _cached_video_path
    
    print(f"[Tool] moment_retrieval: 正在检索视频时刻，查询: '{query}'...")
    
    # 如果没有时间线缓存，需要先生成
    if _cached_timeline is None:
        print("[Tool] 时间线未缓存，正在生成...")
        # 直接调用底层逻辑而非工具
        scene_list = detect(video_path, ContentDetector(threshold=27.0))
        _cached_scenes = scene_list
        _cached_video_path = video_path
        
        cap = cv2.VideoCapture(video_path)
        timeline_entries = []
        
        for i, scene in enumerate(scene_list):
            start_sec = scene[0].get_seconds()
            end_sec = scene[1].get_seconds()
            start_frame = scene[0].frame_num
            end_frame = scene[1].frame_num
            
            mid_frame = (start_frame + end_frame) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                base64_image = base64.b64encode(buffer).decode('utf-8')
                
                messages = [
                    HumanMessage(content=[
                        {"type": "text", "text": "用一句话简洁描述这个画面的主要内容（20字以内）："},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ])
                ]
                response = vl_llm.invoke(messages)
                description = response.content.strip()
                timeline_entries.append(f"场景{i+1} [{start_sec:.1f}s - {end_sec:.1f}s]: {description}")
        
        cap.release()
        _cached_timeline = "\n".join(timeline_entries)
    
    # 使用LLM在时间线中检索匹配的时刻
    retrieval_prompt = f"""你是一个视频时刻检索助手。下面是视频的完整时间线：

{_cached_timeline}

用户想要查找的内容是：{query}

请根据时间线内容，找出与用户查询最相关的时间范围。返回格式：
1. 匹配的时间范围（可能有多个）
2. 每个匹配的置信度（高/中/低）
3. 匹配理由

如果没有找到匹配内容，请说明原因并给出最接近的结果。"""
    
    messages = [HumanMessage(content=retrieval_prompt)]
    response = llm.invoke(messages)
    
    result = f"🔍 Moment Retrieval 结果\n查询: \"{query}\"\n\n{response.content}"
    print(f"[Tool] moment_retrieval: 检索完成")
    
    return result


@tool
def moment_retrieval_with_timeline(query: str, timeline: str) -> str:
    """
    根据已有的时间线和自然语言查询，找到视频中对应的时间范围。
    当用户已经有时间线时，可以直接使用此工具进行检索。
    
    Args:
        query: 自然语言查询，例如"主角在吃饭的片段"
        timeline: 视频时间线文本
    
    Returns:
        匹配的时间范围及相关描述
    """
    print(f"[Tool] moment_retrieval_with_timeline: 正在基于时间线检索，查询: '{query}'...")
    
    retrieval_prompt = f"""你是一个视频时刻检索助手。下面是视频的时间线：

{timeline}

用户想要查找的内容是：{query}

请根据时间线内容，找出与用户查询最相关的时间范围。返回格式：
1. 最匹配的时间范围：[开始时间] - [结束时间]
2. 置信度：高/中/低
3. 匹配理由
4. 如果有其他可能匹配的时间段，也列出来

如果没有找到明确匹配，请给出最接近的结果。"""
    
    messages = [HumanMessage(content=retrieval_prompt)]
    response = llm.invoke(messages)
    
    result = f"🔍 Moment Retrieval 结果\n查询: \"{query}\"\n\n{response.content}"
    print(f"[Tool] moment_retrieval_with_timeline: 检索完成")
    
    return result


# ============ milestone1 工具 ============
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
    print(f"[Tool] get_second: 正在解析问题中的秒数...")
    
    messages = [
        SystemMessage(content="请你找到里面表示秒数的数字，直接返回数字结果，不要额外的输出。如果有小数也保留。只输出数字。"),
        HumanMessage(content=user_query)
    ]
    
    response = llm.invoke(messages)
    result = response.content.strip()
    
    print(f"[Tool] get_second: 解析得到秒数: {result}")
    
    import re
    match = re.search(r'[\d.]+', result)
    if match:
        return float(match.group())
    
    raise ValueError(f"无法从'{user_query}'中提取秒数")


@tool
def get_frame_at_second(second: float, video_path: str = DEFAULT_VIDEO_PATH) -> str:
    """
    根据指定的秒数从视频中提取对应的帧图片，并保存到本地。
    
    Args:
        second: 要提取帧的时间点（秒），例如 10.0
        video_path: 视频文件路径
    
    Returns:
        保存的图片文件路径
    """
    print(f"[Tool] get_frame_at_second: 正在提取第 {second} 秒的画面...")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"[Tool] 视频信息: FPS={fps:.2f}, 时长={duration:.2f}秒")
    
    if second < 0 or second > duration:
        cap.release()
        raise ValueError(f"秒数 {second} 超出视频范围 (0 - {duration:.2f}秒)")
    
    frame_number = int(second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"无法读取第 {second} 秒的帧")
    
    output_path = f"images/frame_at_{second}s.jpg"
    os.makedirs("images", exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"[Tool] 帧已保存到: {output_path}")
    
    return output_path


@tool
def analyze_frame(image_path: str, question: str = "请详细描述这张图片中的画面内容") -> str:
    """
    使用多模态大模型分析图片内容。
    
    Args:
        image_path: 图片文件路径
        question: 关于图片的问题
    
    Returns:
        模型对图片的分析结果
    """
    print(f"[Tool] analyze_frame: 正在分析画面内容...")
    
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
    
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )
    ]
    
    response = vl_llm.invoke(messages)
    print(f"[Tool] analyze_frame: 分析完成")
    
    return response.content


# ============ 创建Agent ============
def create_video_qa_agent():
    """创建视频问答Agent，包含所有视频分析工具"""
    tools = [
        # milestone1 工具
        get_second, 
        get_frame_at_second, 
        analyze_frame,
        # Task 1: 视频分段工具
        detect_scenes,
        get_scene_info,
        # Task 2: 视频Captioning工具（支持时间范围）
        caption_video_segment,
        extract_time_range_frames,
        # Task 3: 时间线总结
        generate_timeline,
        # Task 4: Moment Retrieval
        moment_retrieval,
        moment_retrieval_with_timeline,
    ]
    
    memory = MemorySaver()
    
    system_prompt = """你是一个专业的视频内容分析助手。你可以帮助用户分析视频内容，包括：

1. **查看特定时刻**: 用户问"视频第X秒展示了什么？"时，依次调用 get_second → get_frame_at_second → analyze_frame

2. **视频分段**: 用户想了解视频结构时，调用 detect_scenes 检测场景切换点

3. **视频片段总结 (Captioning)**: 用户想了解某个时间范围的内容时（如"总结30秒到60秒的内容"），使用 caption_video_segment(start_second, end_second) 工具

4. **时间线总结**: 用户想要完整视频概览时，调用 generate_timeline 生成【完整】时间线（所有场景都会输出）

5. **时刻检索 (Moment Retrieval)**: 用户想要根据描述查找视频片段时（如"找到主角吃饭的片段"），使用 moment_retrieval 工具

重要提示：
- 时间线总结会输出【所有】场景，不要只输出部分
- 视频Captioning支持任意时间范围查询
- Moment Retrieval会根据时间线内容匹配用户查询

请根据用户需求选择合适的工具。回答时请用中文。"""
    
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
    print("视频问答Agent - Milestone 2")
    print(f"Agent模型: {AGENT_MODEL}")
    print(f"视觉模型: {VL_MODEL}")
    print(f"默认视频: {DEFAULT_VIDEO_PATH}")
    print("=" * 60)
    
    # 创建Agent
    agent = create_video_qa_agent()
    
    # 测试问题 - 可以测试不同功能
    test_questions = [
        # "视频第10秒展示了什么画面？",
        # "请帮我检测这个视频有几个场景？",
        "请生成这个视频的时间线总结",
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
    print("视频问答Agent - Milestone 2 交互模式")
    print(f"Agent模型: {AGENT_MODEL}")
    print(f"视觉模型: {VL_MODEL}")
    print("\n支持的功能:")
    print("  - 视频第X秒展示了什么？")
    print("  - 检测视频场景")
    print("  - 总结30秒到60秒的视频内容 (Captioning)")
    print("  - 生成完整视频时间线")
    print("  - 找到视频中xxx的片段 (Moment Retrieval)")
    print("\n输入 'quit' 或 'exit' 退出")
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
        
        inputs = {"messages": [HumanMessage(content=question)]}
        
        print("\n--- Agent 开始运行 ---")
        for chunk in agent.stream(inputs, config, stream_mode="values"):
            message = chunk["messages"][-1]
            content = message.content if message.content else "[调用工具中...]"
            print(f"[{message.type}]: {content}")
            print("-" * 30)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive_mode()
    else:
        main()
