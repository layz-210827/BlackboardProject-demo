# -*- coding: utf-8 -*-
"""
Appendix A: Image-based Blackboard Dust Residue Quantitative Analysis Program
Designed for the National Youth Scientific Exploration Modeling Capability Competition.

Optimized version with batch processing, parallel execution, and improved error handling.
"""
import cv2
import numpy as np
import os
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BlackboardAnalyzer:
    """黑板粉笔灰残留分析器 - 优化版本"""
    
    def __init__(self, roi_ratio: Tuple[float, float, float, float] = (0.2, 0.8, 0.2, 0.8)):
        """
        初始化分析器
        
        Args:
            roi_ratio: (y_start%, y_end%, x_start%, x_end%) 感兴趣区域比例
                      默认为 (0.2, 0.8, 0.2, 0.8) 表示 20%-80% 的区域
        """
        self.roi_ratio = roi_ratio
        self._cache = {}  # 简单的内存缓存
        logger.info(f"初始化分析器，ROI比例: {roi_ratio}")
    
    def calculate_residue_index(self, image_path: str) -> Optional[float]:
        """
        量化黑板表面亮度（灰度值）。
        亮度越高，粉笔灰残留越多。
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            残留指数 (float) 或 None (如果处理失败)
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                logger.error(f"文件不存在: {image_path}")
                return None
            
            # 使用缓存避免重复处理
            if image_path in self._cache:
                logger.debug(f"使用缓存: {image_path}")
                return self._cache[image_path]
            
            # 1. 加载实验图像
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"无法读取文件: {image_path}")
                return None
            
            logger.debug(f"加载图像: {image_path}, 尺寸: {img.shape}")
            
            # 2. 转换为灰度以消除环境色彩干扰
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 3. 裁剪ROI(感兴趣区域)以排除黑板边框
            # 改进：直接在视图上计算，避免复制
            h, w = gray.shape
            y_start, y_end, x_start, x_end = self.roi_ratio
            y1, y2 = int(h * y_start), int(h * y_end)
            x1, x2 = int(w * x_start), int(w * x_end)
            
            # 4. 计算平均灰度值 - 直接在视图上计算，无复制
            brightness_val = np.mean(gray[y1:y2, x1:x2])
            result = round(float(brightness_val), 2)
            
            # 缓存结果
            self._cache[image_path] = result
            logger.debug(f"处理完成: {image_path} -> {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"处理图像时出错 {image_path}: {str(e)}")
            return None
    
    def batch_process(self, image_paths: List[str], use_parallel: bool = True, 
                     max_workers: int = 4) -> Dict[str, Optional[float]]:
        """
        批量处理多张图像
        
        Args:
            image_paths: 图像文件路径列表
            use_parallel: 是否使用并行处理
            max_workers: 并行处理的最大工作进程数
            
        Returns:
            {图像路径: 残留指数} 的字典
        """
        results = {}
        
        if not use_parallel or len(image_paths) == 1:
            logger.info(f"顺序处理 {len(image_paths)} 张图像")
            for path in image_paths:
                results[path] = self.calculate_residue_index(path)
        else:
            logger.info(f"并行处理 {len(image_paths)} 张图像，工作进程数: {max_workers}")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.calculate_residue_index, path): path 
                          for path in image_paths}
                
                completed = 0
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        results[path] = future.result()
                        completed += 1
                    except Exception as e:
                        logger.error(f"并行处理失败 {path}: {str(e)}")
                        results[path] = None
                    
                    if completed % 10 == 0:
                        logger.info(f"进度: {completed}/{len(image_paths)}")
        
        return results
    
    def process_directory(self, directory: str, extensions: Tuple[str, ...] = ('.jpg', '.png', '.jpeg'),
                         use_parallel: bool = True) -> Dict[str, Optional[float]]:
        """
        处理目录中的所有图像
        
        Args:
            directory: 目录路径
            extensions: 支持的文件扩展名
            use_parallel: 是否使用并行处理
            
        Returns:
            {图像路径: 残留指数} 的字典
        """
        if not os.path.isdir(directory):
            logger.error(f"目录不存在: {directory}")
            return {}
        
        # 查找所有匹配的图像文件
        image_paths = []
        for ext in extensions:
            image_paths.extend(
                str(p) for p in Path(directory).glob(f"*{ext}") 
                if p.is_file()
            )
        
        logger.info(f"在 {directory} 中找到 {len(image_paths)} 张图像")
        
        if not image_paths:
            logger.warning(f"未在 {directory} 中找到任何图像")
            return {}
        
        return self.batch_process(image_paths, use_parallel=use_parallel)
    
    def export_results(self, results: Dict[str, Optional[float]], output_file: str = "results.csv"):
        """
        导出分析结果到CSV文件
        
        Args:
            results: 分析结果字典
            output_file: 输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("图像文件,残留指数\n")
                for path, value in sorted(results.items()):
                    if value is not None:
                        f.write(f"{path},{value}\n")
                    else:
                        f.write(f"{path},ERROR\n")
            
            logger.info(f"结果已导出到: {output_file}")
        except Exception as e:
            logger.error(f"导出结果时出错: {str(e)}")


def main():
    """主程序入口"""
    print("=== 黑板粉笔灰残留量化分析系统 ===\n")
    
    analyzer = BlackboardAnalyzer()
    
    # 示例 1: 处理单个图像
    print("示例 1: 处理单个图像")
    print("-" * 50)
    sample_image = "dry_wipe_0.jpg"
    result = analyzer.calculate_residue_index(sample_image)
    if result is not None:
        print(f"图像: {sample_image}")
        print(f"残留指数: {result}\n")
    else:
        print(f"无法处理图像: {sample_image}\n")
    
    # 示例 2: 批量处理多张图像（顺序）
    print("示例 2: 批量处理多张图像")
    print("-" * 50)
    samples = ["dry_wipe_0.jpg", "wet_wipe_0.45.jpg", "wet_wipe_0.70.jpg"]
    results = analyzer.batch_process(samples, use_parallel=False)
    for path, value in results.items():
        status = f"残留指数: {value}" if value is not None else "处理失败"
        print(f"{path}: {status}")
    print()
    
    # 示例 3: 处理整个目录（并行处理）
    print("示例 3: 处理整个目录（并行处理）")
    print("-" * 50)
    current_dir = os.getcwd()
    print(f"处理目录: {current_dir}")
    
    # 注意：实际使用时，确保目录中包含图像文件
    dir_results = analyzer.process_directory(current_dir, use_parallel=True, max_workers=4)
    
    if dir_results:
        print(f"\n总处理: {len(dir_results)} 张图像")
        successful = sum(1 for v in dir_results.values() if v is not None)
        print(f"成功: {successful}, 失败: {len(dir_results) - successful}")
        
        # 导出结果
        analyzer.export_results(dir_results, "analysis_results.csv")
        
        # 显示统计信息
        values = [v for v in dir_results.values() if v is not None]
        if values:
            print(f"\n统计信息:")
            print(f"  平均残留指数: {np.mean(values):.2f}")
            print(f"  最小值: {np.min(values):.2f}")
            print(f"  最大值: {np.max(values):.2f}")
            print(f"  标准差: {np.std(values):.2f}")
    else:
        print("未找到任何图像进行处理")


if __name__ == "__main__":
    main()
