"""
VideoAgent 理解能力测试套件
===========================

包含两个测试任务：
1. 视频分类能力测试 (Classification)
   - 数据集: UCF-101 + HMDB51 (10个共有类别)
   - 方法: Multiple Choice / LLM-As-Judger
   
2. 视频时刻定位能力测试 (Grounding)
   - 数据集: Charades-STA (150个样本)
   - 指标: Mean IoU, R@0.3, R@0.5, R@0.7

运行方式：
    python run_benchmark.py --task all
    python run_benchmark.py --task classification --method multiple_choice
    python run_benchmark.py --task grounding --samples 50
"""

import argparse
import sys
import os
from datetime import datetime

# 确保能导入同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_classification_test(args):
    """运行分类测试"""
    from test_classification import run_benchmark as run_cls_benchmark
    
    return run_cls_benchmark(
        samples_per_class=args.samples_per_class,
        method=args.method,
        num_frames=args.frames,
        verbose=True
    )


def run_grounding_test(args):
    """运行Grounding测试"""
    from test_grounding import run_grounding_benchmark
    
    return run_grounding_benchmark(
        max_samples=args.samples,
        num_frames=args.frames,
        use_llm_parser=not args.no_llm_parser,
        verbose=True
    )


def main():
    parser = argparse.ArgumentParser(
        description='VideoAgent 理解能力测试套件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行所有测试
  python run_benchmark.py --task all

  # 仅运行分类测试 (选择题方法)
  python run_benchmark.py --task classification --method multiple_choice --samples-per-class 3

  # 仅运行分类测试 (LLM判断方法)
  python run_benchmark.py --task classification --method llm_judger --samples-per-class 3

  # 仅运行Grounding测试 (快速)
  python run_benchmark.py --task grounding --samples 20

  # 完整Grounding测试
  python run_benchmark.py --task grounding
        """
    )
    
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'classification', 'grounding'],
                        help='测试任务类型')
    
    # 分类测试参数
    parser.add_argument('--method', type=str, default='multiple_choice',
                        choices=['multiple_choice', 'llm_judger'],
                        help='分类测试方法')
    parser.add_argument('--samples-per-class', type=int, default=3,
                        help='分类测试每个类别的样本数')
    
    # Grounding测试参数
    parser.add_argument('--samples', type=int, default=None,
                        help='Grounding测试样本数（默认全部150个）')
    parser.add_argument('--no-llm-parser', action='store_true',
                        help='禁用LLM后备解析器')
    
    # 通用参数
    parser.add_argument('--frames', type=int, default=5,
                        help='每个视频提取的帧数')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("VideoAgent 理解能力测试套件")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    results = {}
    
    if args.task in ['all', 'classification']:
        print("\n" + "=" * 70)
        print("【任务1】视频分类能力测试")
        print("=" * 70)
        try:
            results['classification'] = run_classification_test(args)
        except Exception as e:
            print(f"[Error] 分类测试失败: {e}")
            results['classification'] = None
    
    if args.task in ['all', 'grounding']:
        print("\n" + "=" * 70)
        print("【任务2】视频时刻定位能力测试")
        print("=" * 70)
        try:
            results['grounding'] = run_grounding_test(args)
        except Exception as e:
            print(f"[Error] Grounding测试失败: {e}")
            results['grounding'] = None
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试完成汇总")
    print("=" * 70)
    
    if 'classification' in results and results['classification']:
        cls_result = results['classification']
        acc = cls_result.get('accuracy', {})
        print(f"\n【分类测试】")
        print(f"  UCF-101 Accuracy: {acc.get('ucf101', {}).get('accuracy', 0):.2f}%")
        print(f"  HMDB51 Accuracy:  {acc.get('hmdb51', {}).get('accuracy', 0):.2f}%")
        print(f"  Overall Accuracy: {acc.get('overall', {}).get('accuracy', 0):.2f}%")
    
    if 'grounding' in results and results['grounding']:
        grnd_result = results['grounding']
        metrics = grnd_result.get('metrics', {})
        print(f"\n【Grounding测试】")
        print(f"  Mean IoU: {metrics.get('mean_iou', 0):.4f}")
        print(f"  R@0.3:    {metrics.get('R@0.3', 0):.2f}%")
        print(f"  R@0.5:    {metrics.get('R@0.5', 0):.2f}%")
        print(f"  R@0.7:    {metrics.get('R@0.7', 0):.2f}%")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
