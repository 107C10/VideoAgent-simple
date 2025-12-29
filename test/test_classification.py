"""
视频分类能力测试 - Video Classification Benchmark
基于 UCF-101 和 HMDB51 数据集的10个共有类别进行测试

测试方法：
1. 选择题式测试（Multiple Choice）
2. LLM-As-Judger（自然语言判断）

测试指标：
- UCF-101 Accuracy
- HMDB51 Accuracy  
- Overall Accuracy
"""

import os
import cv2
import base64
import random
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ============ 配置 ============
# 视觉分析模型（多模态 VLM）
VL_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
# 判断模型（用于 LLM-As-Judger）
JUDGE_MODEL = "Qwen/Qwen3-32B"

# 数据集路径（相对于项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UCF101_PATH = os.path.join(PROJECT_ROOT, "datasets/UCF101/UCF-101")
HMDB51_PATH = os.path.join(PROJECT_ROOT, "datasets/hmdb51_org")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")

# 初始化模型
vl_llm = ChatOpenAI(model=VL_MODEL, temperature=0)
judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)

# ============ 共有的10个类别映射 ============
# UCF-101 和 HMDB51 共有的类别
# 格式: (UCF名称, HMDB名称, 中文描述, 英文描述)
SHARED_CLASSES = [
    ("RockClimbingIndoor", "climb", "室内攀岩/攀爬", "Rock Climbing / Climbing"),
    ("Fencing", "fencing", "击剑", "Fencing"),
    ("GolfSwing", "golf", "高尔夫挥杆", "Golf Swing"),
    ("SoccerPenalty", "kick_ball", "足球点球/踢球", "Soccer Penalty / Kicking Ball"),
    ("PullUps", "pullup", "引体向上", "Pull-ups"),
    ("Punch", "punch", "出拳/拳击", "Punching"),
    ("PushUps", "pushup", "俯卧撑", "Push-ups"),
    ("Biking", "ride_bike", "骑自行车", "Biking / Riding Bike"),
    ("HorseRiding", "ride_horse", "骑马", "Horse Riding"),
    ("Basketball", "shoot_ball", "打篮球/投篮", "Basketball / Shooting Ball"),
]

# 类别索引（用于选择题）
CLASS_LABELS = [cls[3] for cls in SHARED_CLASSES]  # 使用英文描述作为选项
CHOICE_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


@dataclass
class VideoSample:
    """视频样本数据类"""
    video_path: str
    dataset: str  # "UCF101" or "HMDB51"
    class_name: str  # 原始类别名
    class_idx: int  # 类别索引 (0-9)
    class_label: str  # 类别英文描述


@dataclass
class TestResult:
    """测试结果数据类"""
    video_path: str
    dataset: str
    ground_truth: str
    ground_truth_idx: int
    predicted: str
    predicted_idx: int
    correct: bool
    method: str  # "multiple_choice" or "llm_judger"
    raw_response: str


def get_video_samples(dataset: str, class_info: tuple, max_samples: int = 5) -> List[VideoSample]:
    """
    从指定数据集和类别获取视频样本
    
    Args:
        dataset: "UCF101" or "HMDB51"
        class_info: (UCF名称, HMDB名称, 中文描述, 英文描述)
        max_samples: 每个类别最多取多少样本
    
    Returns:
        视频样本列表
    """
    samples = []
    class_idx = SHARED_CLASSES.index(class_info)
    
    if dataset == "UCF101":
        class_path = os.path.join(UCF101_PATH, class_info[0])
    else:  # HMDB51
        class_path = os.path.join(HMDB51_PATH, class_info[1])
    
    if not os.path.exists(class_path):
        print(f"[Warning] 路径不存在: {class_path}")
        return samples
    
    # 获取所有视频文件
    video_files = [f for f in os.listdir(class_path) if f.endswith('.avi')]
    
    # 随机采样
    if len(video_files) > max_samples:
        video_files = random.sample(video_files, max_samples)
    
    for video_file in video_files:
        video_path = os.path.join(class_path, video_file)
        samples.append(VideoSample(
            video_path=video_path,
            dataset=dataset,
            class_name=class_info[0] if dataset == "UCF101" else class_info[1],
            class_idx=class_idx,
            class_label=class_info[3]
        ))
    
    return samples


def extract_frames(video_path: str, num_frames: int = 5) -> List[str]:
    """
    从视频中均匀提取帧并编码为base64
    
    Args:
        video_path: 视频文件路径
        num_frames: 要提取的帧数
    
    Returns:
        base64编码的图像列表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] 无法打开视频: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []
    
    # 计算均匀分布的帧位置
    frame_indices = [int(i * total_frames / (num_frames + 1)) for i in range(1, num_frames + 1)]
    
    base64_images = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # 调整大小以减少token消耗
            frame = cv2.resize(frame, (384, 256))
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            base64_image = base64.b64encode(buffer).decode('utf-8')
            base64_images.append(base64_image)
    
    cap.release()
    return base64_images


def build_multiple_choice_prompt() -> str:
    """构建选择题的选项文本"""
    choices = []
    for i, label in enumerate(CLASS_LABELS):
        choices.append(f"{CHOICE_LETTERS[i]}. {label}")
    return "\n".join(choices)


def test_multiple_choice(sample: VideoSample, num_frames: int = 5) -> Optional[TestResult]:
    """
    使用选择题方式测试视频分类
    
    Args:
        sample: 视频样本
        num_frames: 提取的帧数
    
    Returns:
        测试结果
    """
    # 提取帧
    base64_images = extract_frames(sample.video_path, num_frames)
    if not base64_images:
        return None
    
    # 构建选项
    choices_text = build_multiple_choice_prompt()
    
    # 构建消息内容
    content = [
        {"type": "text", "text": f"""What action is shown in this video? Select ONE choice from the following options:

{choices_text}

Please answer with ONLY a single capital letter (A, B, C, D, E, F, G, H, I, or J). Do not include any explanation or additional text."""}
    ]
    
    # 添加图像
    for base64_img in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
        })
    
    try:
        messages = [HumanMessage(content=content)]
        response = vl_llm.invoke(messages)
        raw_response = response.content.strip()
        
        # 解析响应 - 提取第一个有效字母
        predicted_letter = None
        for char in raw_response.upper():
            if char in CHOICE_LETTERS:
                predicted_letter = char
                break
        
        if predicted_letter is None:
            print(f"[Warning] 无法解析响应: {raw_response}")
            return None
        
        predicted_idx = CHOICE_LETTERS.index(predicted_letter)
        predicted_label = CLASS_LABELS[predicted_idx]
        
        return TestResult(
            video_path=sample.video_path,
            dataset=sample.dataset,
            ground_truth=sample.class_label,
            ground_truth_idx=sample.class_idx,
            predicted=predicted_label,
            predicted_idx=predicted_idx,
            correct=(predicted_idx == sample.class_idx),
            method="multiple_choice",
            raw_response=raw_response
        )
    
    except Exception as e:
        print(f"[Error] 测试失败: {e}")
        return None


def test_llm_judger(sample: VideoSample, num_frames: int = 5) -> Optional[TestResult]:
    """
    使用 LLM-As-Judger 方式测试视频分类
    
    Step 1: VLM 描述视频内容
    Step 2: Judge LLM 判断描述是否匹配真实标签
    
    Args:
        sample: 视频样本
        num_frames: 提取的帧数
    
    Returns:
        测试结果
    """
    # 提取帧
    base64_images = extract_frames(sample.video_path, num_frames)
    if not base64_images:
        return None
    
    # Step 1: VLM 描述视频内容
    content = [
        {"type": "text", "text": "What action or activity is shown in this video? Please describe briefly in one sentence."}
    ]
    
    for base64_img in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
        })
    
    try:
        messages = [HumanMessage(content=content)]
        vlm_response = vl_llm.invoke(messages)
        vlm_description = vlm_response.content.strip()
        
        # Step 2: Judge LLM 判断
        # 尝试匹配所有类别，找到最匹配的
        judge_prompt = f"""You are an action classification judge. Given a video description and a list of action categories, determine which category best matches the description.

Video Description: "{vlm_description}"

Action Categories:
{build_multiple_choice_prompt()}

Which category (A-J) best matches this video description? Answer with ONLY a single capital letter. Do not explain."""

        judge_messages = [HumanMessage(content=judge_prompt)]
        judge_response = judge_llm.invoke(judge_messages)
        judge_answer = judge_response.content.strip()
        
        # 解析 Judge 响应
        predicted_letter = None
        for char in judge_answer.upper():
            if char in CHOICE_LETTERS:
                predicted_letter = char
                break
        
        if predicted_letter is None:
            print(f"[Warning] Judge无法解析响应: {judge_answer}")
            return None
        
        predicted_idx = CHOICE_LETTERS.index(predicted_letter)
        predicted_label = CLASS_LABELS[predicted_idx]
        
        return TestResult(
            video_path=sample.video_path,
            dataset=sample.dataset,
            ground_truth=sample.class_label,
            ground_truth_idx=sample.class_idx,
            predicted=predicted_label,
            predicted_idx=predicted_idx,
            correct=(predicted_idx == sample.class_idx),
            method="llm_judger",
            raw_response=f"VLM: {vlm_description}\nJudge: {judge_answer}"
        )
    
    except Exception as e:
        print(f"[Error] LLM Judger测试失败: {e}")
        return None


def run_benchmark(
    samples_per_class: int = 3,
    method: str = "multiple_choice",
    num_frames: int = 5,
    verbose: bool = True
) -> Dict:
    """
    运行完整的分类测试基准
    
    Args:
        samples_per_class: 每个类别每个数据集采样数量
        method: 测试方法 ("multiple_choice" or "llm_judger")
        num_frames: 每个视频提取的帧数
        verbose: 是否打印详细信息
    
    Returns:
        测试结果统计
    """
    print("=" * 70)
    print(f"视频分类能力测试 - Video Classification Benchmark")
    print(f"测试方法: {method}")
    print(f"每类样本数: {samples_per_class}")
    print(f"每视频帧数: {num_frames}")
    print(f"VL模型: {VL_MODEL}")
    if method == "llm_judger":
        print(f"Judge模型: {JUDGE_MODEL}")
    print("=" * 70)
    
    all_results: List[TestResult] = []
    
    # 遍历所有类别
    for class_info in SHARED_CLASSES:
        class_label = class_info[3]
        print(f"\n[{class_label}]")
        
        # UCF101 样本
        ucf_samples = get_video_samples("UCF101", class_info, samples_per_class)
        print(f"  UCF101: {len(ucf_samples)} 个样本")
        
        for i, sample in enumerate(ucf_samples):
            if verbose:
                print(f"    测试 UCF-{i+1}: {os.path.basename(sample.video_path)[:40]}...", end=" ")
            
            if method == "multiple_choice":
                result = test_multiple_choice(sample, num_frames)
            else:
                result = test_llm_judger(sample, num_frames)
            
            if result:
                all_results.append(result)
                status = "✓" if result.correct else "✗"
                if verbose:
                    print(f"{status} (预测: {result.predicted})")
            else:
                if verbose:
                    print("跳过")
        
        # HMDB51 样本
        hmdb_samples = get_video_samples("HMDB51", class_info, samples_per_class)
        print(f"  HMDB51: {len(hmdb_samples)} 个样本")
        
        for i, sample in enumerate(hmdb_samples):
            if verbose:
                print(f"    测试 HMDB-{i+1}: {os.path.basename(sample.video_path)[:40]}...", end=" ")
            
            if method == "multiple_choice":
                result = test_multiple_choice(sample, num_frames)
            else:
                result = test_llm_judger(sample, num_frames)
            
            if result:
                all_results.append(result)
                status = "✓" if result.correct else "✗"
                if verbose:
                    print(f"{status} (预测: {result.predicted})")
            else:
                if verbose:
                    print("跳过")
    
    # 统计结果
    ucf_results = [r for r in all_results if r.dataset == "UCF101"]
    hmdb_results = [r for r in all_results if r.dataset == "HMDB51"]
    
    ucf_correct = sum(1 for r in ucf_results if r.correct)
    hmdb_correct = sum(1 for r in hmdb_results if r.correct)
    total_correct = ucf_correct + hmdb_correct
    
    ucf_total = len(ucf_results)
    hmdb_total = len(hmdb_results)
    total = len(all_results)
    
    ucf_acc = ucf_correct / ucf_total * 100 if ucf_total > 0 else 0
    hmdb_acc = hmdb_correct / hmdb_total * 100 if hmdb_total > 0 else 0
    total_acc = total_correct / total * 100 if total > 0 else 0
    
    # 输出结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print(f"UCF-101 Accuracy: {ucf_correct}/{ucf_total} = {ucf_acc:.2f}%")
    print(f"HMDB51 Accuracy:  {hmdb_correct}/{hmdb_total} = {hmdb_acc:.2f}%")
    print(f"Overall Accuracy: {total_correct}/{total} = {total_acc:.2f}%")
    print("=" * 70)
    
    # 按类别统计
    print("\n各类别准确率:")
    print("-" * 50)
    for class_info in SHARED_CLASSES:
        class_label = class_info[3]
        class_idx = SHARED_CLASSES.index(class_info)
        class_results = [r for r in all_results if r.ground_truth_idx == class_idx]
        class_correct = sum(1 for r in class_results if r.correct)
        class_total = len(class_results)
        class_acc = class_correct / class_total * 100 if class_total > 0 else 0
        print(f"  {class_label:30s}: {class_correct}/{class_total} = {class_acc:.2f}%")
    
    # 保存详细结果
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "samples_per_class": samples_per_class,
        "num_frames": num_frames,
        "vl_model": VL_MODEL,
        "judge_model": JUDGE_MODEL if method == "llm_judger" else None,
        "accuracy": {
            "ucf101": {"correct": ucf_correct, "total": ucf_total, "accuracy": ucf_acc},
            "hmdb51": {"correct": hmdb_correct, "total": hmdb_total, "accuracy": hmdb_acc},
            "overall": {"correct": total_correct, "total": total, "accuracy": total_acc}
        },
        "detailed_results": [
            {
                "video_path": r.video_path,
                "dataset": r.dataset,
                "ground_truth": r.ground_truth,
                "predicted": r.predicted,
                "correct": r.correct,
                "raw_response": r.raw_response
            }
            for r in all_results
        ]
    }
    
    # 保存到文件
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_file = os.path.join(RESULTS_DIR, f"classification_results_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {result_file}")
    
    return results_summary


def quick_test(method: str = "multiple_choice"):
    """
    快速测试 - 每类仅1个样本
    """
    print("=" * 70)
    print("快速测试模式 - 每类1个样本")
    print("=" * 70)
    return run_benchmark(samples_per_class=1, method=method, verbose=True)


def full_test(method: str = "multiple_choice", samples_per_class: int = 5):
    """
    完整测试
    """
    print("=" * 70)
    print(f"完整测试模式 - 每类{samples_per_class}个样本")
    print("=" * 70)
    return run_benchmark(samples_per_class=samples_per_class, method=method, verbose=True)


# ============ 主函数 ============
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='视频分类能力测试')
    parser.add_argument('--method', type=str, default='multiple_choice',
                        choices=['multiple_choice', 'llm_judger'],
                        help='测试方法: multiple_choice 或 llm_judger')
    parser.add_argument('--samples', type=int, default=3,
                        help='每个类别每个数据集的样本数量')
    parser.add_argument('--frames', type=int, default=5,
                        help='每个视频提取的帧数')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试模式（每类1个样本）')
    
    args = parser.parse_args()
    
    # 设置随机种子以保证可复现
    random.seed(42)
    
    if args.quick:
        quick_test(method=args.method)
    else:
        run_benchmark(
            samples_per_class=args.samples,
            method=args.method,
            num_frames=args.frames,
            verbose=True
        )


if __name__ == "__main__":
    main()
