"""
视频问答Agent - Milestone 3
功能扩展：
1. 视频分段 (Scene Detection) - 使用 PySceneDetect
2. 视频 Captioning - 对视频片段进行内容总结
3. 时间线总结 - 生成完整的视频时间线
4. 渐进式时刻检索 (Hierarchical Moment Retrieval) - 从粗到细逐步定位
5. CLIP特征搜索 - 使用CLIP进行高效视频-文本匹配

使用 @tool 装饰器 + create_agent 实现自动工具调用
"""

import os
import cv2
import base64
import numpy as np
from typing import List, Tuple, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from scenedetect import detect, ContentDetector

# 尝试导入CLIP（可选依赖）
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[Warning] CLIP未安装，CLIP特征搜索功能将不可用。安装方式: pip install torch transformers")

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

# ============ CLIP模型管理 ============
_clip_model = None
_clip_processor = None
_clip_device = None

# CLIP特征缓存
_clip_frame_features = None  # 视频帧的CLIP特征
_clip_frame_times = None     # 对应的时间点
_clip_cached_video = None    # 缓存对应的视频路径


def get_clip_model():
    """懒加载CLIP模型"""
    global _clip_model, _clip_processor, _clip_device
    
    if not CLIP_AVAILABLE:
        raise RuntimeError("CLIP未安装，请运行: pip install torch transformers")
    
    if _clip_model is None:
        print("[CLIP] 正在加载CLIP模型...")
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_clip_device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        print(f"[CLIP] 模型加载完成，使用设备: {_clip_device}")
    
    return _clip_model, _clip_processor, _clip_device


def extract_clip_features(video_path: str, sample_interval: float = 1.0) -> Tuple[np.ndarray, List[float]]:
    """
    提取视频帧的CLIP视觉特征
    
    Args:
        video_path: 视频路径
        sample_interval: 采样间隔（秒）
    
    Returns:
        (特征矩阵, 时间点列表)
    """
    global _clip_frame_features, _clip_frame_times, _clip_cached_video
    
    # 检查缓存
    if _clip_cached_video == video_path and _clip_frame_features is not None:
        print("[CLIP] 使用缓存的视频特征")
        return _clip_frame_features, _clip_frame_times
    
    model, processor, device = get_clip_model()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # 按间隔采样帧
    sample_times = []
    current_time = 0
    while current_time < duration:
        sample_times.append(current_time)
        current_time += sample_interval
    
    print(f"[CLIP] 提取视频特征: {len(sample_times)} 帧, 间隔 {sample_interval}s")
    
    features_list = []
    valid_times = []
    
    with torch.no_grad():
        for t in sample_times:
            frame_num = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 处理并提取特征
                inputs = processor(images=frame_rgb, return_tensors="pt").to(device)
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                features_list.append(image_features.cpu().numpy())
                valid_times.append(t)
    
    cap.release()
    
    # 合并特征
    features = np.vstack(features_list)
    
    # 缓存结果
    _clip_frame_features = features
    _clip_frame_times = valid_times
    _clip_cached_video = video_path
    
    print(f"[CLIP] 特征提取完成: shape={features.shape}")
    
    return features, valid_times


def clip_text_search(query: str, video_path: str, top_k: int = 5, sample_interval: float = 1.0) -> List[Tuple[float, float]]:
    """
    使用CLIP进行文本-视频匹配
    
    Args:
        query: 文本查询
        video_path: 视频路径
        top_k: 返回top-k个最匹配的时间点
        sample_interval: 采样间隔
    
    Returns:
        [(时间点, 相似度分数), ...]
    """
    model, processor, device = get_clip_model()
    
    # 获取视频特征
    frame_features, frame_times = extract_clip_features(video_path, sample_interval)
    
    # 提取文本特征
    with torch.no_grad():
        text_inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()
    
    # 计算相似度
    similarities = np.dot(frame_features, text_features.T).flatten()
    
    # 获取top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = [(frame_times[i], float(similarities[i])) for i in top_indices]
    
    return results


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


# ============ Task 5: 渐进式时刻检索 (Hierarchical Moment Retrieval) ============
@tool
def hierarchical_moment_retrieval(query: str, video_path: str = DEFAULT_VIDEO_PATH) -> str:
    """
    【渐进式时刻检索】从粗粒度到细粒度逐步定位视频片段。
    
    工作流程：
    1. 粗粒度阶段：将视频分成几个大的时间段（如每30秒一段），快速定位大致范围
    2. 中粒度阶段：在候选范围内进行更细的分析（如每10秒一段）
    3. 细粒度阶段：在最终候选范围内精确定位
    
    这种方法比一次性总结所有内容更高效，特别适合长视频。
    
    Args:
        query: 自然语言查询，例如"主角在吃饭的片段"
        video_path: 视频文件路径
    
    Returns:
        匹配的时间范围及搜索过程详情
    """
    print(f"[Tool] hierarchical_moment_retrieval: 开始渐进式检索...")
    print(f"[Tool] 查询: '{query}'")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"错误：无法打开视频文件 {video_path}"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"[Tool] 视频时长: {duration:.1f}秒")
    
    results = []
    results.append(f"📽️ 渐进式时刻检索\n查询: \"{query}\"\n视频时长: {duration:.1f}秒\n")
    
    # ========== 阶段1: 粗粒度扫描 ==========
    coarse_interval = max(30.0, duration / 10)  # 至少分成10段，或每30秒一段
    coarse_segments = []
    
    current = 0
    while current < duration:
        end = min(current + coarse_interval, duration)
        coarse_segments.append((current, end))
        current = end
    
    print(f"[Tool] 阶段1 - 粗粒度扫描: {len(coarse_segments)} 个段落")
    results.append(f"\n🔍 阶段1: 粗粒度扫描 ({len(coarse_segments)}个段落，每段约{coarse_interval:.0f}秒)")
    
    # 为每个粗粒度段落生成简短描述
    coarse_descriptions = []
    for i, (start, end) in enumerate(coarse_segments):
        mid_time = (start + end) / 2
        mid_frame = int(mid_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": "用10个字以内描述画面主要内容："},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]
            response = vl_llm.invoke(messages)
            desc = response.content.strip()
            coarse_descriptions.append((start, end, desc))
            results.append(f"  [{start:.0f}s-{end:.0f}s]: {desc}")
    
    # 使用LLM筛选候选段落
    coarse_timeline = "\n".join([f"[{s:.0f}s-{e:.0f}s]: {d}" for s, e, d in coarse_descriptions])
    
    filter_prompt = f"""根据以下视频段落描述，找出最可能包含"{query}"的1-3个时间段。

{coarse_timeline}

只返回时间范围，格式如: [0s-30s], [60s-90s]
如果都不太相关，返回最可能的1个。"""
    
    response = llm.invoke([HumanMessage(content=filter_prompt)])
    candidate_text = response.content.strip()
    results.append(f"\n  → 候选范围: {candidate_text}")
    
    # 解析候选范围
    import re
    candidates = re.findall(r'\[?(\d+)s?\s*[-–到]\s*(\d+)s?\]?', candidate_text)
    if not candidates:
        # 如果解析失败，使用前3个段落
        candidates = [(str(int(s)), str(int(e))) for s, e, _ in coarse_descriptions[:3]]
    
    candidate_ranges = [(float(s), float(e)) for s, e in candidates]
    
    # ========== 阶段2: 中粒度分析 ==========
    print(f"[Tool] 阶段2 - 中粒度分析: {len(candidate_ranges)} 个候选范围")
    results.append(f"\n🔍 阶段2: 中粒度分析 (候选范围内每10秒)")
    
    mid_descriptions = []
    for range_start, range_end in candidate_ranges:
        # 确保范围有效
        range_start = max(0, range_start)
        range_end = min(duration, range_end)
        
        mid_interval = 10.0  # 每10秒一段
        current = range_start
        
        while current < range_end:
            end = min(current + mid_interval, range_end)
            mid_time = (current + end) / 2
            mid_frame = int(mid_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                base64_image = base64.b64encode(buffer).decode('utf-8')
                
                messages = [
                    HumanMessage(content=[
                        {"type": "text", "text": f"这个画面是否包含'{query}'的相关内容？简短描述（15字以内）："},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ])
                ]
                response = vl_llm.invoke(messages)
                desc = response.content.strip()
                mid_descriptions.append((current, end, desc))
                results.append(f"  [{current:.1f}s-{end:.1f}s]: {desc}")
            
            current = end
    
    # ========== 阶段3: 精确定位 ==========
    print(f"[Tool] 阶段3 - 精确定位")
    results.append(f"\n🔍 阶段3: 精确定位")
    
    # 让LLM基于中粒度描述做最终判断
    mid_timeline = "\n".join([f"[{s:.1f}s-{e:.1f}s]: {d}" for s, e, d in mid_descriptions])
    
    final_prompt = f"""根据以下详细的视频片段描述，精确定位"{query}"的时间范围。

{mid_timeline}

请返回：
1. 最匹配的时间范围：[开始秒数, 结束秒数]
2. 置信度：高/中/低
3. 简短理由"""
    
    response = llm.invoke([HumanMessage(content=final_prompt)])
    final_result = response.content.strip()
    results.append(f"\n📍 最终结果:\n{final_result}")
    
    cap.release()
    
    print(f"[Tool] hierarchical_moment_retrieval: 完成")
    
    return "\n".join(results)


# ============ Task 6: CLIP特征搜索 ============
@tool
def clip_moment_search(query: str, video_path: str = DEFAULT_VIDEO_PATH, top_k: int = 5) -> str:
    """
    使用CLIP模型进行高效的视频时刻检索。
    
    CLIP (Contrastive Language-Image Pre-training) 可以直接计算文本和图像的相似度，
    无需逐帧调用VLM，大大提高搜索效率。
    
    工作流程：
    1. 提取视频帧的CLIP视觉特征（会缓存以供复用）
    2. 提取查询文本的CLIP文本特征
    3. 计算特征相似度，返回最匹配的时间点
    
    Args:
        query: 自然语言查询，例如"a person eating food"
        video_path: 视频文件路径
        top_k: 返回前k个最匹配的时间点
    
    Returns:
        最匹配的时间点列表及相似度分数
    """
    if not CLIP_AVAILABLE:
        return "错误：CLIP未安装。请运行: pip install torch transformers"
    
    print(f"[Tool] clip_moment_search: 使用CLIP搜索...")
    print(f"[Tool] 查询: '{query}'")
    
    try:
        # 进行CLIP搜索
        results = clip_text_search(query, video_path, top_k=top_k, sample_interval=1.0)
        
        # 格式化输出
        output_lines = [f"🔍 CLIP Moment Search 结果", f"查询: \"{query}\"", f"Top-{top_k} 匹配时间点:\n"]
        
        for i, (time_point, score) in enumerate(results):
            output_lines.append(f"  {i+1}. {time_point:.1f}s (相似度: {score:.4f})")
        
        # 分析匹配区间
        if results:
            times = [t for t, _ in results]
            min_time = min(times)
            max_time = max(times)
            output_lines.append(f"\n📍 匹配区间估计: [{min_time:.1f}s, {max_time:.1f}s]")
            output_lines.append(f"   (基于top-{top_k}匹配点的范围)")
        
        print(f"[Tool] clip_moment_search: 完成")
        
        return "\n".join(output_lines)
    
    except Exception as e:
        return f"CLIP搜索出错: {str(e)}"


@tool
def clip_guided_moment_retrieval(query: str, video_path: str = DEFAULT_VIDEO_PATH) -> str:
    """
    【CLIP引导的混合检索】结合CLIP快速定位和VLM精确分析。
    
    工作流程：
    1. 使用CLIP快速扫描整个视频，找到最相关的时间点
    2. 在CLIP定位的范围附近，使用VLM进行精确分析
    3. 输出最终的时间范围
    
    这种方法结合了CLIP的效率和VLM的准确性。
    
    Args:
        query: 自然语言查询
        video_path: 视频文件路径
    
    Returns:
        匹配的时间范围及详细分析
    """
    if not CLIP_AVAILABLE:
        return "错误：CLIP未安装。请运行: pip install torch transformers\n将使用渐进式检索替代..."
    
    print(f"[Tool] clip_guided_moment_retrieval: CLIP引导的混合检索...")
    print(f"[Tool] 查询: '{query}'")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"错误：无法打开视频文件 {video_path}"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    results = [f"🔍 CLIP引导的混合检索", f"查询: \"{query}\"", f"视频时长: {duration:.1f}秒\n"]
    
    # ========== Step 1: CLIP快速定位 ==========
    print(f"[Tool] Step 1: CLIP快速定位")
    results.append("📌 Step 1: CLIP快速定位")
    
    try:
        clip_results = clip_text_search(query, video_path, top_k=10, sample_interval=2.0)
        
        # 显示CLIP结果
        results.append("  CLIP匹配的时间点:")
        for i, (time_point, score) in enumerate(clip_results[:5]):
            results.append(f"    {i+1}. {time_point:.1f}s (score: {score:.3f})")
        
        # 确定候选范围（取top-5结果的时间范围，并扩展5秒）
        if clip_results:
            top_times = [t for t, _ in clip_results[:5]]
            range_start = max(0, min(top_times) - 5)
            range_end = min(duration, max(top_times) + 5)
            results.append(f"\n  → 候选范围: [{range_start:.1f}s, {range_end:.1f}s]")
        else:
            range_start, range_end = 0, duration
    
    except Exception as e:
        results.append(f"  CLIP定位失败: {e}，使用全视频范围")
        range_start, range_end = 0, duration
    
    # ========== Step 2: VLM精确分析 ==========
    print(f"[Tool] Step 2: VLM精确分析 [{range_start:.1f}s - {range_end:.1f}s]")
    results.append(f"\n📌 Step 2: VLM精确分析 (范围: {range_start:.1f}s - {range_end:.1f}s)")
    
    # 在候选范围内每5秒采样分析
    analysis_interval = 5.0
    analyses = []
    
    current = range_start
    while current < range_end:
        end = min(current + analysis_interval, range_end)
        mid_time = (current + end) / 2
        mid_frame = int(mid_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": f"这个画面是否展示了'{query}'？回答Yes/No并简述内容（10字以内）："},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]
            response = vl_llm.invoke(messages)
            analysis = response.content.strip()
            analyses.append((current, end, analysis))
            results.append(f"  [{current:.1f}s-{end:.1f}s]: {analysis}")
        
        current = end
    
    cap.release()
    
    # ========== Step 3: 综合判断 ==========
    print(f"[Tool] Step 3: 综合判断")
    results.append(f"\n📌 Step 3: 综合判断")
    
    analysis_text = "\n".join([f"[{s:.1f}s-{e:.1f}s]: {a}" for s, e, a in analyses])
    
    final_prompt = f"""根据以下分析结果，确定"{query}"的精确时间范围。

{analysis_text}

返回格式：
时间范围: [开始秒数, 结束秒数]
置信度: 高/中/低
理由: (简短说明)"""
    
    response = llm.invoke([HumanMessage(content=final_prompt)])
    final_result = response.content.strip()
    results.append(f"\n📍 最终结果:\n{final_result}")
    
    print(f"[Tool] clip_guided_moment_retrieval: 完成")
    
    return "\n".join(results)


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
        # Task 4: Moment Retrieval (原始方法)
        moment_retrieval,
        moment_retrieval_with_timeline,
        # Task 5: 渐进式时刻检索 (Hierarchical)
        hierarchical_moment_retrieval,
        # Task 6: CLIP特征搜索
        clip_moment_search,
        clip_guided_moment_retrieval,
    ]
    
    memory = MemorySaver()
    
    system_prompt = """你是一个专业的视频内容分析助手。你可以帮助用户分析视频内容，包括：

1. **查看特定时刻**: 用户问"视频第X秒展示了什么？"时，依次调用 get_second → get_frame_at_second → analyze_frame

2. **视频分段**: 用户想了解视频结构时，调用 detect_scenes 检测场景切换点

3. **视频片段总结 (Captioning)**: 用户想了解某个时间范围的内容时（如"总结30秒到60秒的内容"），使用 caption_video_segment(start_second, end_second) 工具

4. **时间线总结**: 用户想要完整视频概览时，调用 generate_timeline 生成【完整】时间线

5. **时刻检索 (Moment Retrieval)**: 有三种检索方式可选：
   - **moment_retrieval**: 基础方法，先生成完整时间线再检索（适合短视频）
   - **hierarchical_moment_retrieval**: 【推荐】渐进式检索，从粗到细逐步定位（适合长视频，更高效）
   - **clip_guided_moment_retrieval**: CLIP引导的混合检索，先用CLIP快速定位再用VLM精确分析（最高效）

6. **CLIP快速搜索**: 使用 clip_moment_search 进行纯CLIP特征匹配（速度最快，但可能不如VLM准确）

选择建议：
- 短视频（<2分钟）: 使用 moment_retrieval
- 中等视频（2-10分钟）: 使用 hierarchical_moment_retrieval
- 长视频（>10分钟）或需要快速定位: 使用 clip_guided_moment_retrieval 或 clip_moment_search

请根据用户需求和视频长度选择合适的工具。回答时请用中文。"""
    
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
    print("视频问答Agent - Milestone 3")
    print(f"Agent模型: {AGENT_MODEL}")
    print(f"视觉模型: {VL_MODEL}")
    print(f"CLIP可用: {CLIP_AVAILABLE}")
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
    print("视频问答Agent - Milestone 3 交互模式")
    print(f"Agent模型: {AGENT_MODEL}")
    print(f"视觉模型: {VL_MODEL}")
    print(f"CLIP可用: {CLIP_AVAILABLE}")
    print("\n支持的功能:")
    print("  - 视频第X秒展示了什么？")
    print("  - 检测视频场景")
    print("  - 总结30秒到60秒的视频内容 (Captioning)")
    print("  - 生成完整视频时间线")
    print("  - 找到视频中xxx的片段 (Moment Retrieval)")
    print("  - 【新】渐进式搜索xxx片段 (Hierarchical)")
    print("  - 【新】使用CLIP搜索xxx (CLIP Search)")
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