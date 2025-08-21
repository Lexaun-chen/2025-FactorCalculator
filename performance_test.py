"""
Calculator 性能测试脚本
测试3D和4D数据的算子性能，并进行对比分析
"""

import sys
import time
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append('.')
from Data_Provider.Data_Provider import Calculator

def test_3d_performance():
    """测试3D数据算子性能"""
    print("=== 3D数据性能测试 ===")
    
    # 创建更大的3D测试数据
    large_3d = np.random.randn(252, 5000, 10)  # 1年数据，5000只股票，10个特征
    large_calc_3d = Calculator(large_3d, feature_names=[f'feature_{i}' for i in range(10)])

    # 测试不同算子的性能
    algorithms_3d = [
        ('abs_val', 'feature_0', {}),
        ('rank', 'feature_1', {}),
        ('ts_mean', 'feature_9', {'window': 20}),
        ('ts_std', 'feature_5', {'window': 20}),
        ('ts_cov', ['feature_3', 'feature_6'], {'window': 5}),
        ('add', ['feature_7', 'feature_1'], {}),
        ('ts_lag', 'feature_2', {'periods': 5}),
        ('ts_delta', 'feature_4', {'periods': 3}),
        ('multiply', ['feature_8', 'feature_9'], {}),
        ('zscore', 'feature_0', {})
    ]

    print(f"测试数据形状: {large_3d.shape}")
    print("3D算子性能测试结果:")
    print("-" * 70)

    results_3d = []
    for algo_name, features, kwargs in algorithms_3d:
        start_time = time.time()
        
        try:
            if isinstance(features, list) and len(features) == 2:
                result = large_calc_3d.apply_binary_operator(algo_name, features[0], features[1], **kwargs)
            else:
                result = large_calc_3d.apply_unary_operator(algo_name, features, **kwargs)
            
            end_time = time.time()
            execution_time = end_time - start_time
            results_3d.append((algo_name, execution_time, result.shape))
            print(f"{algo_name:15} | {execution_time:.4f}s | 结果形状: {result.shape}")
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"{algo_name:15} | {execution_time:.4f}s | 错误: {str(e)[:40]}...")
    
    return results_3d, large_3d.shape

def test_4d_performance():
    """测试4D数据算子性能"""
    print("\n=== 4D数据性能测试 ===")
    
    # 创建更大的4D测试数据
    large_4d = np.random.randn(60, 1000, 12, 8)  # 5年月度数据，1000只股票，12个财务区间，8个特征
    feature_names_4d = ['PE', 'ROE', 'ROA', 'PB', 'EPS', 'Revenue', 'NetIncome', 'TotalAssets']
    large_calc_4d = Calculator(large_4d, feature_names=feature_names_4d)

    # 测试不同4D算子的性能
    algorithms_4d = [
        ('custom_abs', 'PE', {}),
        ('cross_sectional_rank', 'ROE', {}),
        ('cross_sectional_normalize', 'ROA', {}),
        ('rolling_mean', 'PB', {'window': 6, 'axis': -1}),
        ('rolling_std', 'EPS', {'window': 8, 'axis': -1}),
        ('rolling_corr', ['Revenue', 'NetIncome'], {'window': 5, 'axis': -1}),
        ('cross_sectional_neutralize', ['PE', 'ROE'], {}),
        ('divide', ['PE', 'PB'], {}),
        ('rolling_sum', 'TotalAssets', {'window': 4, 'axis': -1}),
        ('delay', 'PE', {'window': 2, 'axis': -1}),
        ('add', ['PE', 'ROE'], {}),
        ('multiply', ['ROA', 'PB'], {}),
        ('cross_sectional_correlation', ['EPS', 'Revenue'], {}),
        ('rolling_covariance', ['NetIncome', 'TotalAssets'], {'window': 6, 'axis': -1})
    ]

    print(f"测试数据形状: {large_4d.shape} (月数, 股票数, 财务区间数, 特征数)")
    print("4D算子性能测试结果:")
    print("-" * 70)

    results_4d = []
    for algo_name, features, kwargs in algorithms_4d:
        start_time = time.time()
        
        try:
            if isinstance(features, list) and len(features) == 2:
                result = large_calc_4d.apply_binary_operator(algo_name, features[0], features[1], **kwargs)
            else:
                result = large_calc_4d.apply_unary_operator(algo_name, features, **kwargs)
            
            end_time = time.time()
            execution_time = end_time - start_time
            results_4d.append((algo_name, execution_time, result.shape))
            print(f"{algo_name:25} | {execution_time:.4f}s | 结果形状: {result.shape}")
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"{algo_name:25} | {execution_time:.4f}s | 错误: {str(e)[:30]}...")
    
    return results_4d, large_4d.shape

def compare_3d_4d_performance():
    """比较3D和4D算子性能"""
    print("\n=== 3D vs 4D 性能对比 ===")
    
    # 创建对应的测试数据
    large_3d = np.random.randn(252, 1000, 8)  # 1年数据，1000只股票，8个特征
    large_4d = np.random.randn(60, 1000, 12, 8)  # 5年月度数据，1000只股票，12个财务区间，8个特征
    
    feature_names = ['PE', 'ROE', 'ROA', 'PB', 'EPS', 'Revenue', 'NetIncome', 'TotalAssets']
    large_calc_3d = Calculator(large_3d, feature_names=feature_names)
    large_calc_4d = Calculator(large_4d, feature_names=feature_names)

    # 选择几个共同的算子进行对比
    common_tests = [
        ('abs_val', 'custom_abs', 'PE'),
        ('rank', 'cross_sectional_rank', 'ROE'),
        ('add', 'add', ['PE', 'ROE']),
        ('multiply', 'multiply', ['ROA', 'PB']),
        ('divide', 'divide', ['EPS', 'Revenue'])
    ]

    print("算子类型              | 3D时间    | 4D时间    | 性能比率  | 数据量比率")
    print("-" * 75)

    data_ratio = large_4d.size / large_3d.size
    
    for test_3d, test_4d, features in common_tests:
        # 3D测试
        start_time = time.time()
        try:
            if isinstance(features, list):
                result_3d = large_calc_3d.apply_binary_operator(test_3d, features[0], features[1])
            else:
                result_3d = large_calc_3d.apply_unary_operator(test_3d, features)
            time_3d = time.time() - start_time
        except Exception as e:
            time_3d = -1
            print(f"{test_3d:20} | 错误     | -        | -        | -")
            continue
        
        # 4D测试
        start_time = time.time()
        try:
            if isinstance(features, list):
                result_4d = large_calc_4d.apply_binary_operator(test_4d, features[0], features[1])
            else:
                result_4d = large_calc_4d.apply_unary_operator(test_4d, features)
            time_4d = time.time() - start_time
        except Exception as e:
            time_4d = -1
            print(f"{test_3d:20} | {time_3d:.4f}s  | 错误     | -        | -")
            continue
        
        # 性能比率
        perf_ratio = time_4d / time_3d if time_3d > 0 else float('inf')
        
        print(f"{test_3d:20} | {time_3d:.4f}s  | {time_4d:.4f}s  | {perf_ratio:.2f}x     | {data_ratio:.1f}x")

    print(f"\n数据规模对比:")
    print(f"3D数据: {large_3d.shape} = {large_3d.size:,} 个元素")
    print(f"4D数据: {large_4d.shape} = {large_4d.size:,} 个元素")
    print(f"4D数据量是3D的 {data_ratio:.1f} 倍")

def analyze_performance_by_category():
    """按算子类别分析性能"""
    print("\n=== 算子类别性能分析 ===")
    
    # 创建测试数据
    test_3d = np.random.randn(100, 1000, 5)
    test_4d = np.random.randn(50, 1000, 12, 5)
    
    calc_3d = Calculator(test_3d, feature_names=[f'f{i}' for i in range(5)])
    calc_4d = Calculator(test_4d, feature_names=[f'f{i}' for i in range(5)])
    
    # 按类别组织算子
    categories = {
        "基础运算": {
            "3d": [('abs_val', 'f0'), ('add', ['f0', 'f1']), ('multiply', ['f2', 'f3'])],
            "4d": [('custom_abs', 'f0'), ('add', ['f0', 'f1']), ('multiply', ['f2', 'f3'])]
        },
        "截面算子": {
            "3d": [('rank', 'f0'), ('zscore', 'f1')],
            "4d": [('cross_sectional_rank', 'f0'), ('cross_sectional_normalize', 'f1')]
        },
        "时序算子": {
            "3d": [('ts_mean', 'f0'), ('ts_std', 'f1'), ('ts_lag', 'f2')],
            "4d": [('rolling_mean', 'f0'), ('rolling_std', 'f1'), ('delay', 'f2')]
        }
    }
    
    for category, tests in categories.items():
        print(f"\n{category}:")
        print("  维度 | 算子名称           | 执行时间")
        print("  " + "-" * 40)
        
        for dim, test_list in tests.items():
            calc = calc_3d if dim == "3d" else calc_4d
            
            for test_item in test_list:
                if len(test_item) == 2:
                    algo_name, features = test_item
                    kwargs = {}
                else:
                    algo_name, features, kwargs = test_item
                
                start_time = time.time()
                try:
                    if isinstance(features, list):
                        calc.apply_binary_operator(algo_name, features[0], features[1], **kwargs)
                    else:
                        calc.apply_unary_operator(algo_name, features, **kwargs)
                    exec_time = time.time() - start_time
                    print(f"  {dim.upper():2} | {algo_name:18} | {exec_time:.4f}s")
                except Exception as e:
                    exec_time = time.time() - start_time
                    print(f"  {dim.upper():2} | {algo_name:18} | 错误")

def main():
    """主函数"""
    print("Calculator 性能测试")
    print("=" * 50)
    
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 运行各项测试
    test_3d_performance()
    test_4d_performance()
    compare_3d_4d_performance()
    analyze_performance_by_category()
    
    print("\n" + "=" * 50)
    print("性能测试完成")

if __name__ == "__main__":
    main()
