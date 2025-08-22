#!/usr/bin/env python3
"""
测试Formula_Reader中的表达式解析器功能

测试用例包括：
1. 基本表达式解析
2. 函数调用格式解析
3. 嵌套表达式解析
4. 算子调用和计算
5. 错误处理
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Data_Provider.Formula_Reader import FormulaParser, create_formula_parser
from Data_Provider.Data_Provider import Calculator

def create_test_data():
    """创建测试数据"""
    # 创建3D测试数据 (D=10天, N=5股票, M=4特征)
    np.random.seed(42)
    
    D, N, M = 10, 5, 4
    data_3d = np.random.randn(D, N, M)
    
    # 添加一些NaN值来测试处理
    data_3d[0, 0, 0] = np.nan
    data_3d[5, 2, 1] = np.nan
    
    feature_names = ['close', 'open', 'volume', 'high']
    
    return data_3d, feature_names

def test_basic_parsing():
    """测试基本解析功能"""
    print("=== 测试基本解析功能 ===")
    
    parser = create_formula_parser(data_dimension="3D")
    
    # 测试用例
    test_expressions = [
        "close",
        "close + open", 
        "rank(close)",
        "corr(close, open)",
        "rank(corr(close, open))",
        "add(close, open)",
        "rank(add(corr(close, open), abs_val(close)))"
    ]
    
    for expr in test_expressions:
        print(f"\n表达式: {expr}")
        try:
            parser.debug_parse(expr)
        except Exception as e:
            print(f"解析失败: {e}")

def test_with_calculator():
    """测试与Calculator结合的功能"""
    print("\n=== 测试与Calculator结合的功能 ===")
    
    # 创建测试数据
    data_3d, feature_names = create_test_data()
    
    # 创建Calculator
    calculator = Calculator(data_3d, feature_names=feature_names)
    
    # 创建解析器
    parser = create_formula_parser(calculator=calculator, data_dimension="3D")
    
    # 测试简单表达式
    test_expressions = [
        "close",
        "rank(close)", 
        "add(close, open)",
        "corr(close, open)",
    ]
    
    for expr in test_expressions:
        print(f"\n计算表达式: {expr}")
        try:
            result = parser.evaluate(expr)
            print(f"结果形状: {result.shape}")
            print(f"结果类型: {result.dtype}")
            print(f"NaN数量: {np.sum(np.isnan(result))}")
            print(f"有效值范围: [{np.nanmin(result):.4f}, {np.nanmax(result):.4f}]")
        except Exception as e:
            print(f"计算失败: {e}")

def test_complex_expressions():
    """测试复杂表达式"""
    print("\n=== 测试复杂表达式 ===")
    
    # 创建测试数据
    data_3d, feature_names = create_test_data()
    calculator = Calculator(data_3d, feature_names=feature_names)
    parser = create_formula_parser(calculator=calculator, data_dimension="3D")
    
    # 测试你提到的表达式
    complex_expressions = [
        "rank(add(corr(close, open), abs_val(close)))",
        "ts_mean(rank(close), 5)",
        "zscore(add(close, open))",
        "multiply(rank(close), rank(open))"
    ]
    
    for expr in complex_expressions:
        print(f"\n复杂表达式: {expr}")
        try:
            # 先调试解析
            parser.debug_parse(expr)
            
            # 然后计算
            result = parser.evaluate(expr)
            print(f"计算成功! 结果形状: {result.shape}")
            print(f"有效值数量: {np.sum(~np.isnan(result))}")
            
        except Exception as e:
            print(f"处理失败: {e}")

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    data_3d, feature_names = create_test_data()
    calculator = Calculator(data_3d, feature_names=feature_names)
    parser = create_formula_parser(calculator=calculator, data_dimension="3D")
    
    # 错误表达式测试
    error_expressions = [
        "",  # 空表达式
        "unknown_feature",  # 未知特征
        "unknown_operator(close)",  # 未知算子
        "add(close)",  # 参数数量不匹配
        "rank(close, open, volume)",  # 参数过多
    ]
    
    for expr in error_expressions:
        print(f"\n错误表达式: '{expr}'")
        try:
            result = parser.evaluate(expr)
            print(f"意外成功: {result.shape}")
        except Exception as e:
            print(f"预期错误: {e}")

def test_arithmetic_operators():
    """测试算术运算符"""
    print("\n=== 测试算术运算符 ===")
    
    data_3d, feature_names = create_test_data()
    calculator = Calculator(data_3d, feature_names=feature_names)
    parser = create_formula_parser(calculator=calculator, data_dimension="3D")
    
    # 算术运算符测试
    arithmetic_expressions = [
        "close + open",
        "close - open", 
        "close * open",
        "close / open",
        "(close + open) * volume",
        "rank(close + open)"
    ]
    
    for expr in arithmetic_expressions:
        print(f"\n算术表达式: {expr}")
        try:
            parser.debug_parse(expr)
            result = parser.evaluate(expr)
            print(f"计算成功! 结果形状: {result.shape}")
        except Exception as e:
            print(f"计算失败: {e}")

def test_feature_access():
    """测试特征访问方式"""
    print("\n=== 测试特征访问方式 ===")
    
    data_3d, feature_names = create_test_data()
    calculator = Calculator(data_3d, feature_names=feature_names)
    parser = create_formula_parser(calculator=calculator, data_dimension="3D")
    
    # 测试不同的特征访问方式
    access_tests = [
        ("close", "特征名访问"),
        ("x1", "x索引访问"),
        # ("0", "数字索引访问")  # 这个可能需要特殊处理
    ]
    
    for expr, desc in access_tests:
        print(f"\n{desc}: {expr}")
        try:
            result = parser.evaluate(expr)
            print(f"成功! 结果形状: {result.shape}")
            print(f"前5个值: {result.flatten()[:5]}")
        except Exception as e:
            print(f"失败: {e}")

def main():
    """主测试函数"""
    print("开始测试Formula_Reader表达式解析器...")
    
    try:
        test_basic_parsing()
        test_with_calculator()
        test_complex_expressions()
        test_arithmetic_operators()
        test_feature_access()
        test_error_handling()
        
        print("\n=== 所有测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
