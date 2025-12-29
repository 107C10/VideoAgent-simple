"""
视频时刻定位能力测试 - Video Grounding Benchmark
基于 Charades-STA 数据集进行测试

测试任务：给定视频和文本描述，预测对应的时间范围
评估指标：IoU (Intersection over Union)

测试方法：
1. 直接格式化输出：要求模型输出固定格式 [start, end]
2. LLM重解析：使用LLM将自由格式响应解析为JSON
"""

import os
import cv2
import re
import json
import base64
import random
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ============ 配置 ============
# 视觉分析模型（多模态 VLM）
VL_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
# 解析模型（用于解析时间范围）
PARSER_MODEL = "Qwen/Qwen3-32B"

# 数据集路径（相对于项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARADES_PATH = os.path.join(PROJECT_ROOT, "datasets/Charades_v1_480_selected")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")

# 初始化模型
vl_llm = ChatOpenAI(model=VL_MODEL, temperature=0)
parser_llm = ChatOpenAI(model=PARSER_MODEL, temperature=0)

# ============ 测试数据 ============
# Charades-STA 测试集 (150个样本)
# 格式: [Video ID] [Start Time] [End Time] ## [Description]
CHARADES_VAL_DATA = """3MSZA 24.3 30.4##person turn a light on.
AMT7R 4.3 12.5##a person is putting a picture onto the wall.
YVKIV 4.4 9.2##the person puts down the bag.
VXJS4 0.0 3.4##a person walks through the doorway.
GBD1Y 26.2 31.3##person closes the door.
KVXJ9 1.0 11.0##person runs up the stairs.
E6DLK 9.3 14.4##person runs to the window to look out.
AKO6M 0.0 8.8##a person stands in the bathroom holding a glass.
F7TG5 13.2 20.8##person sits in a chair.
KOVTR 10.2 19.8##person they stand up.
FPJ9D 25.8 34.0##person sit on a chair.
5NV2Z 5.1 10.2##person takes a cup off a desk.
Y1HGC 0.2 9.8##a person walks in a doorway drinking some coffee.
9JZO2 26.1 36.0##person put on their shoes.
4J1AP 13.1 21.5##person drinking a glass of water.
V1WN7 0.0 15.9##person sitting on bed.
BL7OF 1.0 5.0##a person throws a pair of shoes under the window.
5C4EK 0.0 6.2##a person reads a book.
S591U 9.4 17.0##person sitting on top of bed closes laptop.
DLFPX 0.0 7.7##a person is standing on their stairs holding a bag.
9HG4Z 0.0 7.7##a person is closing a door.
I7AS7 0.0 5.1##a person runs through a doorway.
J48N6 18.4 32.0##the same person was laughing as he was undressing.
JVLAZ 9.3 18.0##person laying on the bed sneezing.
2RFLZ 3.0 10.4##a person picks up their phone talks on it.
N3U9S 0.0 4.8##person opens a cabinet door twice.
J9T5D 27.1 35.0##person start laughing.
1CYLM 0.0 14.2##a person sits down on a couch.
V2GC9 10.1 24.2##person putting groceries away.
LGJAR 17.6 25.2##person turns on the light.
ZHRPD 11.8 19.6##the person puts the towel onto the shelf.
43CCM 7.7 22.0##person undressing by the shelf beside the doorway.
YZ8HK 1.4 8.7##the person closes the laptop.
2P7A9 20.4 27.6##person take a picture from the top of a cabinet.
1ZWPP 2.6 7.7##a person is smiling in the mirror of their bathroom.
QUXYH 0.0 10.9##person laughs in the kitchen.
V3SOF 20.0 28.2##person opened up the refrigerator.
SY5QP 0.0 14.0##a person is standing in the room holding a sandwich.
AOK1L 30.5 36.0##person drinking from a cup.
NBSPH 26.8 32.0##person sits on the table.
HY4FS 4.9 11.6##person the open the cabinet.
YSE1G 0.6 8.9##person closed the door.
XMYXI 0.0 7.0##a person sits in a chair.
QLEN3 14.0 19.0##person open the closet.
YSKX3 13.0 18.0##person throws pillow on it.
HBWLV 9.4 14.3##person puts the shoes on the floor.
XE19C 0.9 8.5##person puts a cup in the sink.
W3SC3 8.0 15.2##the person immediately opened a window.
WKPQ3 15.8 28.7##person they talk on the phone.
FXZI5 3.0 12.4##a person is dressing in a towel.
OE2M1 12.9 19.3##another person comes in takes the box away.
ZF7YA 5.3 11.7##the person sits on a pillow on the floor.
B82GJ 17.7 22.5##the person takes a paper towel from the table.
FYYFX 2.0 11.2##person opens a window.
Q7IQI 0.0 4.5##person looks at some books open it.
HSEH1 10.0 22.6##person is fixing a light bulb.
6ZWSU 1.4 7.2##person puts the books down.
V0ZD9 0.5 16.0##person washes a cup.
AK2KG 12.5 19.0##person opens up the window.
8MQH6 0.0 8.1##a person opens a closet door.
MAUMO 0.0 9.6##person takes a drink from a glass.
EHS68 1.1 6.4##another person is throwing a pair of shoes.
G30NS 2.5 10.6##person puts it into a box.
SUJWY 17.1 29.7##the person laughs at something on the screen multiple times.
KERO6 11.4 16.9##the person puts some food onto a pan.
SV6KF 9.3 14.5##a person throws a broom at the stairs.
VCYH8 18.2 23.8##person turns off the light as they're leaving.
G852Y 3.0 10.1##one person is wrapped in a blanket sneezing.
OKXIQ 0.0 5.7##person they use the doorknob to open the cabinet.
KB7WQ 7.3 14.0##person they close the box.
27DCQ 13.5 19.2##a person awakens in bed.
UM5II 3.7 14.0##the person eats a few bite.
XT9D4 18.3 25.3##a person is putting food into the refrigerator.
1W6YY 3.5 7.9##person closes the cabinet door.
YOCRB 26.4 36.0##person sit down in a chair.
ZDV60 0.3 9.2##a person opens a closet door.
GHC5X 22.2 27.8##a person takes a glass of water.
6IOV0 0.0 13.1##person working on a laptop.
OP2SS 20.4 26.7##the person takes out their laptop.
PPY0W 0.0 7.8##a person is holding a bag.
DCV2M 25.4 32.0##person start undressing.
TEV5K 7.1 16.2##person begings to drink a glass of water.
Q3BCC 0.0 6.8##a person closes a door.
M8OYC 1.7 11.5##a person walks in holding a bag of groceries.
FBOF0 9.2 22.2##a person awakens in bed.
D1NT7 10.8 18.6##a person is putting some dishes into a box.
3UACJ 1.8 6.4##person takes a drink from a glass.
U2AO1 0.0 4.6##closes the doors.the person takes the food.
3J9L5 1.0 6.6##a person is eating.
XQVXF 4.5 11.1##person opens a fridge door.
9LWQ6 0.0 9.7##a person undresses.
2ADJI 10.0 21.9##person putting away groceries.
B1AMA 21.0 35.7##person dresses in pajamas.
LA6AA 2.2 11.8##a person was holding a blanket.
YNWMW 9.9 15.9##the person takes a broom.
LUTIO 0.0 10.4##the person lays on the floor the gets up.
Q6290 0.0 8.3##person watching the television eating an apple.
TAQ25 11.6 17.7##the person closes the laptop.
EEVD3 18.4 32.0##person begins to eat it.
IOGR7 19.7 28.3##person they stand up.
2OREK 11.5 17.6##looking out the window in a curious manner.
JCT0K 2.6 10.2##a person is opening the closet door.
H0P5D 11.0 18.1##one person was running into the table.
BFCU9 0.2 5.3##person eating it.
TUD6M 4.3 8.0##person throws their blanket inside.
A3XXB 1.9 7.1##person puts them away on a shelf in a closet.
L5YHH 0.6 5.7##a person is turns on the light in their closet.
GL2JW 14.9 21.9##person closes a cupboard door.
FAJEA 14.3 23.5##person they take some medicine.
RXELU 2.5 6.6##person turns the light on.
QCVZN 6.2 12.0##a person takes a book off a shelf.
5B9XE 18.7 29.0##person turns off the ceiling light.
LKNZD 0.0 6.5##a person is throwing clothes on the floor.
0DVVD 2.8 14.3##a person closes a window.
U8M2P 1.6 13.3##another person is sneezing on a sandwich.
VAV4C 13.8 23.5##person eating a sandwich.
6C0BK 0.3 7.8##a person is putting a towel on a towel rack.
5CN21 11.6 17.5##a person is standing in their garage holding a pillow.
OTIA2 0.0 4.4##person turns on a lightswitch.
B4ED1 23.5 32.8##a person is pouring water into a cup.
86E2E 0.0 8.8##the person is sitting on the floor.
YJ1KW 8.8 14.3##person sits down in a chair.
DJ87X 0.0 9.6##a person is looking out the window.
N0ZPI 14.4 20.8##person take a drink from the cup.
OVD84 4.1 10.8##a person turns on some lights.
NW0KT 7.9 18.8##person takes book out.
8SDK5 15.4 23.0##person sits down in a chair.
95GB4 8.8 19.8##person takes a picture with a camera.
TDAY1 0.0 16.6##person working on a laptop.
NRGQB 6.1 11.6##person throws their shoes off by kicking them.
5R8BL 12.1 22.0##person begin undressing.
BI4KK 0.9 8.4##person drinking from a glass of water.
DU416 7.9 18.0##person holding a bag.
NO1GJ 15.9 25.8##person washes clothes.
7SNIO 0.0 7.3##one person runs into the room laughing.
QOYH2 17.5 25.3##person takes a drink from a cup.
QWKVM 10.3 19.1##person closes the door.
COBS0 6.7 14.0##a smiling person takes a towel.
TJZ0P 10.2 19.0##the person is seated in a chair.
759MY 7.5 13.0##person put something on the table.
F1V30 0.0 4.2##a person holding a towel walks up to a counter.
U502L 20.4 26.9##a person awakens in their bathroom holding their phone.
OPPVW 19.2 25.0##person laugh about it.
NI15V 13.5 19.5##person holding a towel in the other hand.
6HT0J 6.8 19.5##person start reading a book.
BM3UJ 6.4 17.3##a person walks through the entryway holding a glass.
DM2XL 15.4 20.9##person opens the front door.
RQRRD 7.5 24.3##person quickly undressing.
S2FUO 0.0 8.4##a person kneeling on the floor talks on a phone.
36T5X 24.0 34.0##person starts undressing out of their outdoor clothes."""


@dataclass
class GroundingSample:
    """Grounding测试样本"""
    video_id: str
    video_path: str
    start_time: float
    end_time: float
    description: str


@dataclass
class GroundingResult:
    """Grounding测试结果"""
    video_id: str
    description: str
    gt_start: float
    gt_end: float
    pred_start: float
    pred_end: float
    iou: float
    raw_response: str
    parse_method: str  # "regex" or "llm"


def parse_val_data() -> List[GroundingSample]:
    """
    解析验证集数据
    
    Returns:
        样本列表
    """
    samples = []
    for line in CHARADES_VAL_DATA.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # 格式: [Video ID] [Start Time] [End Time]##[Description]
        parts = line.split('##')
        if len(parts) != 2:
            continue
        
        time_part = parts[0].strip()
        description = parts[1].strip()
        
        # 解析时间
        time_tokens = time_part.split()
        if len(time_tokens) != 3:
            continue
        
        video_id = time_tokens[0]
        start_time = float(time_tokens[1])
        end_time = float(time_tokens[2])
        
        # 构建视频路径
        video_path = os.path.join(CHARADES_PATH, f"{video_id}.mp4")
        
        if os.path.exists(video_path):
            samples.append(GroundingSample(
                video_id=video_id,
                video_path=video_path,
                start_time=start_time,
                end_time=end_time,
                description=description
            ))
        else:
            print(f"[Warning] 视频不存在: {video_path}")
    
    return samples


def get_video_duration(video_path: str) -> float:
    """获取视频时长"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps > 0:
        return frame_count / fps
    return 0.0


def extract_frames_uniform(video_path: str, num_frames: int = 8) -> Tuple[List[str], float]:
    """
    从视频中均匀提取帧
    
    Args:
        video_path: 视频路径
        num_frames: 提取帧数
    
    Returns:
        (base64图像列表, 视频时长)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    
    if total_frames == 0:
        cap.release()
        return [], 0.0
    
    # 均匀采样帧位置
    frame_indices = [int(i * total_frames / (num_frames + 1)) for i in range(1, num_frames + 1)]
    
    base64_images = []
    frame_times = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # 调整大小
            frame = cv2.resize(frame, (384, 256))
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            base64_image = base64.b64encode(buffer).decode('utf-8')
            base64_images.append(base64_image)
            frame_times.append(frame_idx / fps)
    
    cap.release()
    return base64_images, duration


def calculate_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """
    计算时间段的IoU
    
    Args:
        pred_start, pred_end: 预测的时间范围
        gt_start, gt_end: 真实的时间范围
    
    Returns:
        IoU值
    """
    # 确保start < end
    pred_start, pred_end = min(pred_start, pred_end), max(pred_start, pred_end)
    gt_start, gt_end = min(gt_start, gt_end), max(gt_start, gt_end)
    
    # 计算交集
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)
    
    # 计算并集
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def parse_time_range_regex(response: str) -> Tuple[Optional[float], Optional[float]]:
    """
    使用正则表达式从响应中提取时间范围
    
    支持格式:
    - [1.5, 3.2]
    - (1.5, 3.2)
    - 1.5-3.2
    - 1.5 to 3.2
    - start: 1.5, end: 3.2
    - from 1.5 seconds to 3.2 seconds
    """
    # 模式1: [start, end] 或 (start, end)
    pattern1 = r'[\[\(]\s*(\d+\.?\d*)\s*[,\s]\s*(\d+\.?\d*)\s*[\]\)]'
    match = re.search(pattern1, response)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    # 模式2: start-end 或 start - end
    pattern2 = r'(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)'
    match = re.search(pattern2, response)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    # 模式3: from X to Y 或 X to Y
    pattern3 = r'(?:from\s+)?(\d+\.?\d*)\s*(?:seconds?\s+)?to\s+(\d+\.?\d*)'
    match = re.search(pattern3, response, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    # 模式4: start: X, end: Y 或 start=X, end=Y
    pattern4 = r'start[:\s=]+(\d+\.?\d*).*?end[:\s=]+(\d+\.?\d*)'
    match = re.search(pattern4, response, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    # 模式5: X seconds ... Y seconds (提取两个时间)
    pattern5 = r'(\d+\.?\d*)\s*(?:second|sec|s).*?(\d+\.?\d*)\s*(?:second|sec|s)'
    match = re.search(pattern5, response, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    
    return None, None


def parse_time_range_llm(response: str, video_duration: float) -> Tuple[Optional[float], Optional[float]]:
    """
    使用LLM解析时间范围
    """
    parse_prompt = f"""Parse the following response and extract the time range (start and end time in seconds).

Response to parse:
"{response}"

Video duration: {video_duration:.1f} seconds

Return ONLY a JSON object in this exact format, no other text:
{{"start": <number>, "end": <number>}}

If you cannot find a clear time range, estimate based on the context.
If the response mentions the beginning, use 0 as start.
If the response mentions the end, use {video_duration:.1f} as end."""

    try:
        messages = [HumanMessage(content=parse_prompt)]
        llm_response = parser_llm.invoke(messages)
        result_text = llm_response.content.strip()
        
        # 尝试提取JSON
        json_match = re.search(r'\{[^}]+\}', result_text)
        if json_match:
            result = json.loads(json_match.group())
            return float(result.get('start', 0)), float(result.get('end', video_duration))
    except Exception as e:
        print(f"[Warning] LLM解析失败: {e}")
    
    return None, None


def test_grounding_sample(
    sample: GroundingSample,
    num_frames: int = 8,
    use_llm_parser: bool = True
) -> Optional[GroundingResult]:
    """
    测试单个Grounding样本
    
    Args:
        sample: 测试样本
        num_frames: 提取的帧数
        use_llm_parser: 是否使用LLM解析（作为后备）
    
    Returns:
        测试结果
    """
    # 提取视频帧
    base64_images, duration = extract_frames_uniform(sample.video_path, num_frames)
    if not base64_images:
        return None
    
    # 构建Prompt
    prompt = f"""You are watching a video that is {duration:.1f} seconds long. I will show you {len(base64_images)} frames evenly sampled from this video.

Your task: Find the time range in the video where this action/event occurs:
"{sample.description}"

IMPORTANT: You must respond with ONLY the time range in this exact format:
[start_time, end_time]

For example: [5.2, 12.8]

The times should be in seconds, between 0 and {duration:.1f}.
Do not include any other text or explanation."""

    # 构建消息
    content = [{"type": "text", "text": prompt}]
    for base64_img in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
        })
    
    try:
        messages = [HumanMessage(content=content)]
        response = vl_llm.invoke(messages)
        raw_response = response.content.strip()
        
        # 尝试正则解析
        pred_start, pred_end = parse_time_range_regex(raw_response)
        parse_method = "regex"
        
        # 如果正则失败，使用LLM解析
        if pred_start is None and use_llm_parser:
            pred_start, pred_end = parse_time_range_llm(raw_response, duration)
            parse_method = "llm"
        
        # 如果仍然失败，使用默认值
        if pred_start is None:
            pred_start, pred_end = 0.0, duration
            parse_method = "default"
        
        # 限制在有效范围内
        pred_start = max(0, min(pred_start, duration))
        pred_end = max(0, min(pred_end, duration))
        
        # 计算IoU
        iou = calculate_iou(pred_start, pred_end, sample.start_time, sample.end_time)
        
        return GroundingResult(
            video_id=sample.video_id,
            description=sample.description,
            gt_start=sample.start_time,
            gt_end=sample.end_time,
            pred_start=pred_start,
            pred_end=pred_end,
            iou=iou,
            raw_response=raw_response,
            parse_method=parse_method
        )
    
    except Exception as e:
        print(f"[Error] 测试失败: {e}")
        return None


def run_grounding_benchmark(
    max_samples: int = None,
    num_frames: int = 8,
    use_llm_parser: bool = True,
    verbose: bool = True
) -> Dict:
    """
    运行完整的Grounding测试
    
    Args:
        max_samples: 最大测试样本数（None表示全部）
        num_frames: 每个视频提取的帧数
        use_llm_parser: 是否使用LLM作为后备解析器
        verbose: 是否打印详细信息
    
    Returns:
        测试结果统计
    """
    print("=" * 70)
    print("视频时刻定位能力测试 - Video Grounding Benchmark")
    print(f"数据集: Charades-STA")
    print(f"VL模型: {VL_MODEL}")
    print(f"每视频帧数: {num_frames}")
    print(f"LLM解析后备: {'启用' if use_llm_parser else '禁用'}")
    print("=" * 70)
    
    # 解析测试数据
    samples = parse_val_data()
    print(f"\n加载了 {len(samples)} 个有效测试样本")
    
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]
        print(f"使用前 {max_samples} 个样本进行测试")
    
    results: List[GroundingResult] = []
    
    for i, sample in enumerate(samples):
        if verbose:
            print(f"\n[{i+1}/{len(samples)}] {sample.video_id}")
            print(f"  描述: {sample.description[:50]}...")
            print(f"  真实: [{sample.start_time:.1f}, {sample.end_time:.1f}]", end=" ")
        
        result = test_grounding_sample(sample, num_frames, use_llm_parser)
        
        if result:
            results.append(result)
            if verbose:
                print(f"-> 预测: [{result.pred_start:.1f}, {result.pred_end:.1f}]")
                print(f"  IoU: {result.iou:.3f} ({result.parse_method})")
        else:
            if verbose:
                print("-> 失败")
    
    # 统计结果
    if not results:
        print("\n没有有效的测试结果！")
        return {}
    
    ious = [r.iou for r in results]
    mean_iou = sum(ious) / len(ious)
    
    # IoU阈值统计
    iou_thresholds = [0.3, 0.5, 0.7]
    recall_at_thresholds = {}
    for thresh in iou_thresholds:
        count = sum(1 for iou in ious if iou >= thresh)
        recall_at_thresholds[f"R@{thresh}"] = count / len(ious) * 100
    
    # 解析方法统计
    parse_methods = {}
    for r in results:
        parse_methods[r.parse_method] = parse_methods.get(r.parse_method, 0) + 1
    
    # 输出结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print(f"测试样本数: {len(results)}/{len(samples)}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print("-" * 40)
    for thresh, recall in recall_at_thresholds.items():
        print(f"{thresh}: {recall:.2f}%")
    print("-" * 40)
    print("解析方法分布:")
    for method, count in parse_methods.items():
        print(f"  {method}: {count} ({count/len(results)*100:.1f}%)")
    print("=" * 70)
    
    # IoU分布统计
    print("\nIoU分布:")
    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), 
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for low, high in bins:
        count = sum(1 for iou in ious if low <= iou < high)
        bar = '█' * int(count / len(ious) * 50)
        print(f"  [{low:.1f}-{high:.1f}): {bar} {count}")
    
    # 保存详细结果
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "vl_model": VL_MODEL,
            "num_frames": num_frames,
            "use_llm_parser": use_llm_parser,
            "total_samples": len(samples),
            "tested_samples": len(results)
        },
        "metrics": {
            "mean_iou": mean_iou,
            **recall_at_thresholds
        },
        "parse_methods": parse_methods,
        "detailed_results": [
            {
                "video_id": r.video_id,
                "description": r.description,
                "gt_range": [r.gt_start, r.gt_end],
                "pred_range": [r.pred_start, r.pred_end],
                "iou": r.iou,
                "parse_method": r.parse_method,
                "raw_response": r.raw_response
            }
            for r in results
        ]
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_file = os.path.join(RESULTS_DIR, f"grounding_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {result_file}")
    
    return results_summary


def quick_test(num_samples: int = 10):
    """快速测试 - 少量样本"""
    print("=" * 70)
    print(f"快速测试模式 - {num_samples}个样本")
    print("=" * 70)
    return run_grounding_benchmark(max_samples=num_samples, verbose=True)


def full_test():
    """完整测试 - 所有150个样本"""
    print("=" * 70)
    print("完整测试模式 - 150个样本")
    print("=" * 70)
    return run_grounding_benchmark(max_samples=None, verbose=True)


# ============ 主函数 ============
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='视频时刻定位能力测试')
    parser.add_argument('--samples', type=int, default=None,
                        help='测试样本数量（默认全部150个）')
    parser.add_argument('--frames', type=int, default=8,
                        help='每个视频提取的帧数')
    parser.add_argument('--no-llm-parser', action='store_true',
                        help='禁用LLM后备解析')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试模式（10个样本）')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test(10)
    else:
        run_grounding_benchmark(
            max_samples=args.samples,
            num_frames=args.frames,
            use_llm_parser=not args.no_llm_parser,
            verbose=True
        )


if __name__ == "__main__":
    main()
