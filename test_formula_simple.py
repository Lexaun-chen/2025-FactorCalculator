#!/usr/bin/env python3
"""
简化的Formula_Reader测试脚本，避免复杂依赖
"""

import sys
import numpy as np
import inspect
import re
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 直接导入算子模块
import operators_3D

# 复制FormulaParser的核心部分进行测试
class SimpleFormulaParser:
    """简化的表达式解析器用于测试"""
    
    def __init__(self):
        # 3D合法算子名列表
        ARITHMETIC_OP = ["add", "subtract", "multiply", "divide", "power", "mean2"]
        MATH_OP = ["log", "sqrt", "abs_val", "neg", "sigmoid", "hardsigmoid", "leakyrelu", "gelu", "sign", "power2", "power3", "curt", "inv"]
        TIME_SERIES_OP = ["ts_lag", "ts_diff", "ts_pct_change", "ts_mean", "ts_std", "ts_ewm", "ts_max", "ts_min", "ts_argmin", "ts_argmax", 
                          "ts_max_to_min", "ts_sum", "ts_max_mean", "ts_cov", "ts_corr", "ts_rankcorr", "ts_to_wm", "ts_rank", "ts_median"]
        CROSS_SECTION_OP = ["rank", "rank_div", "rank_sub", "rank_mul", "zscore", "min_max_scale", "umr", "regress_residual", "sortreverse"]
        CONDITIONAL_OP = ["if_then_else", "series_max"]
        CONSTANT_OP = ["return_const_1", "return_const_5", "return_const_10", "return_const_20"]
        
        self.valid_operators = (ARITHMETIC_OP + MATH_OP + TIME_SERIES_OP + 
                              CROSS_SECTION_OP + CONDITIONAL_OP + CONSTANT_OP)
        self.operators_module = operators_3D
        self.operator_info_cache = {}
        
        # 算术运算符映射
        self.arithmetic_mapping = {
            '+': 'add',
            '-': 'subtract', 
            '*': 'multiply',
            '/': 'divide',
            'add': 'add',
            'subtract': 'subtract',
            'multiply': 'multiply', 
            'divide': 'divide'
        }
        
        # 模拟特征数据
        self.test_data = self.create_test_data()
    
    def create_test_data(self):
        """创建测试数据"""
        np.random.seed(42)
        D, N = 10, 5  # 10天，5只股票
        
        data = {
            'close': np.random.randn(D, N) * 10 + 100,
            'open': np.random.randn(D, N) * 10 + 100,
            'volume': np.random.randint(1000, 10000, (D, N)).astype(float),
            'high': np.random.randn(D, N) * 10 + 105
        }
        
        # 添加一些NaN值
        data['close'][0, 0] = np.nan
        data['volume'][5, 2] = np.nan
        
        return data
    
    def tokenize(self, expression):
        """词法分析"""
        expression = re.sub(r'\s+', ' ', expression.strip())
        token_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*|\d+\.?\d*|[+\-*/(),])'
        tokens = re.findall(token_pattern, expression)
        return [token for token in tokens if token.strip()]
    
    def is_operator(self, token):
        """判断是否为算子"""
        return (token in self.valid_operators or 
                token in self.arithmetic_mapping)
    
    def is_feature(self, token):
        """判断是否为特征名"""
        return token in self.test_data
    
    def is_number(self, token):
        """判断是否为数字"""
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    def get_operator_info(self, operator_name):
        """获取算子信息"""
        if operator_name in self.operator_info_cache:
            return self.operator_info_cache[operator_name]
        
        actual_op_name = self.arithmetic_mapping.get(operator_name, operator_name)
        
        if not hasattr(self.operators_module, actual_op_name):
            raise ValueError(f"算子 '{actual_op_name}' 不存在")
        
        operator_func = getattr(self.operators_module, actual_op_name)
        sig = inspect.signature(operator_func)
        
        required_params = [p for p in sig.parameters.values() 
                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) 
                          and p.default == p.empty]
        
        param_count = len(required_params)
        
        info = {
            'name': actual_op_name,
            'param_count': param_count,
            'function': operator_func,
        }
        
        self.operator_info_cache[operator_name] = info
        return info
    
    def parse_function_call(self, tokens, start_idx):
        """解析函数调用"""
        if start_idx >= len(tokens):
            return [], start_idx
        
        func_name = tokens[start_idx]
        
        if (start_idx + 1 < len(tokens) and tokens[start_idx + 1] == '('):
            paren_count = 0
            end_idx = start_idx + 1
            
            for i in range(start_idx + 1, len(tokens)):
                if tokens[i] == '(':
                    paren_count += 1
                elif tokens[i] == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        end_idx = i
                        break
            
            args_tokens = tokens[start_idx + 2:end_idx]
            parsed_args = self.parse_arguments(args_tokens)
            result = parsed_args + [func_name]
            
            return result, end_idx + 1
        else:
            return [func_name], start_idx + 1
    
    def parse_arguments(self, args_tokens):
        """解析函数参数，处理逗号分隔的参数列表"""
        if not args_tokens:
            return []
        
        # 分割参数（按逗号分隔，但要考虑嵌套括号）
        arguments = []
        current_arg = []
        paren_count = 0
        
        for token in args_tokens:
            if token == ',' and paren_count == 0:
                # 遇到顶层逗号，结束当前参数
                if current_arg:
                    # 递归解析当前参数
                    parsed_arg = self.parse_expression_tokens(current_arg)
                    arguments.extend(parsed_arg)
                    current_arg = []
            else:
                if token == '(':
                    paren_count += 1
                elif token == ')':
                    paren_count -= 1
                current_arg.append(token)
        
        # 处理最后一个参数
        if current_arg:
            parsed_arg = self.parse_expression_tokens(current_arg)
            arguments.extend(parsed_arg)
        
        return arguments
    
    def parse_expression_tokens(self, tokens):
        """解析表达式token"""
        if not tokens:
            return []
        
        parsed_tokens = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if self.is_operator(token):
                if i + 1 < len(tokens) and tokens[i + 1] == '(':
                    func_tokens, next_i = self.parse_function_call(tokens, i)
                    parsed_tokens.extend(func_tokens)
                    i = next_i
                else:
                    parsed_tokens.append(token)
                    i += 1
            else:
                parsed_tokens.append(token)
                i += 1
        
        return parsed_tokens
    
    def parse_expression(self, expression):
        """解析表达式"""
        tokens = self.tokenize(expression)
        
        if not tokens:
            raise ValueError("表达式为空")
        
        if '(' in tokens:
            parsed_tokens = self.parse_expression_tokens(tokens)
            return parsed_tokens
        else:
            return tokens
    
    def evaluate_postfix(self, postfix_tokens):
        """计算后缀表达式"""
        if not postfix_tokens:
            raise ValueError("后缀表达式为空")
        
        stack = []
        
        for token in postfix_tokens:
            if self.is_feature(token):
                feature_data = self.test_data[token]
                stack.append(feature_data)
            
            elif self.is_number(token):
                constant_value = float(token)
                if stack and hasattr(stack[-1], 'shape'):
                    constant_array = np.full(stack[-1].shape, constant_value)
                else:
                    constant_array = constant_value
                stack.append(constant_array)
            
            elif self.is_operator(token):
                op_info = self.get_operator_info(token)
                operator_func = op_info['function']
                param_count = op_info['param_count']
                
                if len(stack) < param_count:
                    raise ValueError(f"算子 '{token}' 需要 {param_count} 个参数，但栈中只有 {len(stack)} 个")
                
                args = []
                for _ in range(param_count):
                    args.append(stack.pop())
                args.reverse()
                
                try:
                    result = operator_func(*args)
                    stack.append(result)
                except Exception as e:
                    raise ValueError(f"算子 '{token}' 计算失败: {e}")
            
            else:
                raise ValueError(f"未知的token: {token}")
        
        if len(stack) != 1:
            raise ValueError(f"表达式计算错误，最终栈中应该只有一个结果，但有 {len(stack)} 个")
        
        return stack[0]
    
    def evaluate(self, expression):
        """计算表达式"""
        postfix_tokens = self.parse_expression(expression)
        result = self.evaluate_postfix(postfix_tokens)
        return result
    
    def debug_parse(self, expression):
        """调试解析过程"""
        print(f"=== 调试解析表达式: {expression} ===")
        
        tokens = self.tokenize(expression)
        print(f"词法分析结果: {tokens}")
        
        postfix_tokens = self.parse_expression(expression)
        print(f"后缀表达式: {postfix_tokens}")
        
        print("\nToken分析:")
        for token in postfix_tokens:
            if self.is_feature(token):
                print(f"  {token}: 特征")
            elif self.is_number(token):
                print(f"  {token}: 数字")
            elif self.is_operator(token):
                op_info = self.get_operator_info(token)
                print(f"  {token}: 算子 (参数数量: {op_info['param_count']})")
            else:
                print(f"  {token}: 未知")
        
        print("=" * 50)

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")
    
    parser = SimpleFormulaParser()
    
    # 测试数据访问
    print("\n测试数据:")
    for feature, data in parser.test_data.items():
        print(f"{feature}: 形状={data.shape}, NaN数量={np.sum(np.isnan(data))}")
    
    # 测试简单表达式
    simple_expressions = [
        "close",
        "rank(close)",
        "add(close, open)",
    ]
    
    for expr in simple_expressions:
        print(f"\n表达式: {expr}")
        try:
            parser.debug_parse(expr)
            result = parser.evaluate(expr)
            print(f"计算成功! 结果形状: {result.shape}")
            print(f"NaN数量: {np.sum(np.isnan(result))}")
            print(f"有效值范围: [{np.nanmin(result):.4f}, {np.nanmax(result):.4f}]")
        except Exception as e:
            print(f"失败: {e}")

def test_complex_expressions():
    """测试复杂表达式"""
    print("\n=== 测试复杂表达式 ===")
    
    parser = SimpleFormulaParser()
    
    # 修正的表达式（使用正确的算子名称）
    complex_expressions = [
        "rank(add(abs_val(close), open))",
        "ts_corr(close, open, 5)",  # 使用ts_corr而不是corr
        "ts_mean(close, 5)",
        "add(rank(close), rank(open))",  # 你提到的表达式的简化版本
    ]
    
    for expr in complex_expressions:
        print(f"\n复杂表达式: {expr}")
        try:
            parser.debug_parse(expr)
            result = parser.evaluate(expr)
            print(f"计算成功! 结果形状: {result.shape}")
            print(f"有效值数量: {np.sum(~np.isnan(result))}")
        except Exception as e:
            print(f"处理失败: {e}")

def test_operator_info():
    """测试算子信息获取"""
    print("\n=== 测试算子信息 ===")
    
    parser = SimpleFormulaParser()
    
    test_operators = ['rank', 'add', 'corr', 'abs_val', 'ts_mean']
    
    for op in test_operators:
        try:
            info = parser.get_operator_info(op)
            print(f"{op}: 参数数量={info['param_count']}, 函数名={info['name']}")
        except Exception as e:
            print(f"{op}: 错误 - {e}")

def main():
    """主测试函数"""
    print("开始测试简化版Formula_Reader...")
    
    try:
        test_operator_info()
        test_basic_functionality()
        test_complex_expressions()
        
        print("\n=== 所有测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
