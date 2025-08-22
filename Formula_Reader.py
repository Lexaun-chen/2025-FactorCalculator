import os
import sys
import inspect
import re
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy import create_engine 


current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import *
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


# 3D合法算子名列表

# 算术算子
ARITHMETIC_OP = ["add", "subtract", "multiply", "divide", "power", "mean2"]
# 数学函数算子
MATH_OP = ["log", "sqrt", "abs_val", "neg", "sigmoid", "hardsigmoid", "leakyrelu", "gelu", "sign", "power2", "power3", "curt", "inv"]
# 时序算子
TIME_SERIES_OP = ["ts_lag", "ts_diff", "ts_pct_change", "ts_mean", "ts_std", "ts_ewm", "ts_max", "ts_min", "ts_argmin", "ts_argmax", 
                  "ts_max_to_min", "ts_sum", "ts_max_mean", "ts_cov", "ts_corr", "ts_rankcorr", "ts_to_wm", "ts_rank", "ts_median"]
# 截面算子 (排名、标准化等)
CROSS_SECTION_OP = ["rank", "rank_div", "rank_sub", "rank_mul", "zscore", "min_max_scale", "umr", "regress_residual", "sortreverse"]
# 条件算子
CONDITIONAL_OP = ["if_then_else", "series_max"]
# 常数算子
CONSTANT_OP = ["return_const_1", "return_const_5", "return_const_10", "return_const_20"]



# 给出4D合法算子名列表

CROSS_OP = ["cs_norm", "cs_rank"]
UNARY_OP = ["custom_abs", "sign", "log", "sqrt", "square"]
SM_BINARY_OP = ["add", "multiply", "custom_min", "custom_max"]
ASM_BINARY_OP = ["minus", "signedpower", "divide", "isover", "cs_neut"]
ROLLING_UNARY_OP = ["delay", "delta", "pctchange", "rolling_sum", "rolling_rank", "rolling_mean", "rolling_std", "rolling_max", "rolling_min",
                "rolling_weighted_mean_linear_decay"]
SM_ROLLING_BINARY_OP = ["rolling_covariance", "rolling_corr"]
ASM_ROLLING_BINARY_OP = []


class FormulaParser:
    """
    表达式解析器，支持将逆波兰表达式转换为中波兰表达式并计算结果
    
    支持的表达式格式：
    - 逆波兰表达式：rank(corr(close, open) add abs_val(close))
    - 中波兰表达式：rank + corr close open abs_val close
    
    功能：
    1. 表达式词法分析和语法解析
    2. 逆波兰到中波兰表达式转换
    3. 根据表达式调用数据和算子
    4. 递归计算并返回结果
    """
    
    def __init__(self, calculator=None, data_dimension="3D"):
        """
        初始化表达式解析器
        
        Parameters:
            calculator: Calculator实例，用于数据和算子调用
            data_dimension: 数据维度，"3D"或"4D"
        """
        self.calculator = calculator
        self.data_dimension = data_dimension
        
        # 根据数据维度选择合法算子列表
        if data_dimension == "3D":
            self.valid_operators = (ARITHMETIC_OP + MATH_OP + TIME_SERIES_OP + 
                                  CROSS_SECTION_OP + CONDITIONAL_OP + CONSTANT_OP)
            # 导入3D算子模块
            import operators_3D
            self.operators_module = operators_3D
        elif data_dimension == "4D":
            self.valid_operators = (CROSS_OP + UNARY_OP + SM_BINARY_OP + 
                                  ASM_BINARY_OP + ROLLING_UNARY_OP + 
                                  SM_ROLLING_BINARY_OP + ASM_ROLLING_BINARY_OP)
            # 导入4D算子模块
            import operators_4D
            self.operators_module = operators_4D
        else:
            raise ValueError(f"不支持的数据维度: {data_dimension}")
        
        # 算子参数信息缓存
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
    
    def tokenize(self, expression):
        """
        词法分析：将表达式分解为token列表
        
        Parameters:
            expression: 输入表达式字符串
            
        Returns:
            list: token列表
        """
        # 移除多余空格并标准化
        expression = re.sub(r'\s+', ' ', expression.strip())
        
        # 定义token模式
        token_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*|\d+\.?\d*|[+\-*/()])'
        
        # 提取所有token
        tokens = re.findall(token_pattern, expression)
        
        # 过滤空token
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def is_operator(self, token):
        """判断token是否为算子"""
        return (token in self.valid_operators or 
                token in self.arithmetic_mapping)
    
    def is_feature(self, token):
        """判断token是否为特征名"""
        if self.calculator is None:
            # 如果没有calculator，假设非算子且非数字的标识符都是特征名
            return (not self.is_operator(token) and 
                    not token.isdigit() and 
                    not token in ['(', ')'])
        
        # 检查是否为有效的特征标识符
        try:
            self.calculator.get_feature_index(token)
            return True
        except (ValueError, KeyError):
            return False
    
    def is_number(self, token):
        """判断token是否为数字"""
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    def get_operator_info(self, operator_name):
        """
        获取算子信息（参数数量、是否为一元/二元算子等）
        
        Parameters:
            operator_name: 算子名称
            
        Returns:
            dict: 算子信息
        """
        if operator_name in self.operator_info_cache:
            return self.operator_info_cache[operator_name]
        
        # 处理算术运算符映射
        actual_op_name = self.arithmetic_mapping.get(operator_name, operator_name)
        
        if not hasattr(self.operators_module, actual_op_name):
            raise ValueError(f"算子 '{actual_op_name}' 不存在")
        
        operator_func = getattr(self.operators_module, actual_op_name)
        
        # 使用inspect获取函数签名
        sig = inspect.signature(operator_func)
        
        # 计算所有位置参数数量（包括有默认值的参数）
        all_params = [p for p in sig.parameters.values() 
                     if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        
        # 必需参数数量（没有默认值的参数）
        required_params = [p for p in all_params if p.default == p.empty]
        
        # 使用所有参数数量作为param_count，这样可以处理带默认值的参数
        param_count = len(all_params)
        min_param_count = len(required_params)
        
        info = {
            'name': actual_op_name,
            'original_name': operator_name,
            'param_count': param_count,
            'is_unary': param_count == 1,
            'is_binary': param_count == 2,
            'function': operator_func,
            'signature': sig
        }
        
        self.operator_info_cache[operator_name] = info
        return info
    
    def infix_to_postfix(self, tokens):
        """
        将中缀表达式转换为后缀表达式（逆波兰表达式）
        使用调度场算法(Shunting Yard Algorithm)
        
        Parameters:
            tokens: token列表
            
        Returns:
            list: 后缀表达式token列表
        """
        output = []
        operator_stack = []
        
        # 算子优先级定义
        precedence = {
            '+': 1, '-': 1, 'add': 1, 'subtract': 1,
            '*': 2, '/': 2, 'multiply': 2, 'divide': 2,
        }
        
        # 其他算子默认优先级为3
        for op in self.valid_operators:
            if op not in precedence:
                precedence[op] = 3
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if self.is_feature(token) or self.is_number(token):
                output.append(token)
            
            elif self.is_operator(token):
                # 处理算子优先级
                while (operator_stack and 
                       operator_stack[-1] != '(' and
                       operator_stack[-1] in precedence and
                       precedence.get(operator_stack[-1], 3) >= precedence.get(token, 3)):
                    output.append(operator_stack.pop())
                operator_stack.append(token)
            
            elif token == '(':
                operator_stack.append(token)
            
            elif token == ')':
                # 弹出直到遇到左括号
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                if operator_stack and operator_stack[-1] == '(':
                    operator_stack.pop()  # 移除左括号
            
            i += 1
        
        # 弹出剩余算子
        while operator_stack:
            if operator_stack[-1] in ['(', ')']:
                operator_stack.pop()
            else:
                output.append(operator_stack.pop())
        
        return output
    
    def parse_function_call(self, tokens, start_idx):
        """
        解析函数调用格式：func_name(arg1, arg2, ...)
        
        Parameters:
            tokens: token列表
            start_idx: 函数名的起始索引
            
        Returns:
            tuple: (解析后的token列表, 下一个处理位置)
        """
        if start_idx >= len(tokens):
            return [], start_idx
        
        func_name = tokens[start_idx]
        
        # 检查是否为函数调用格式
        if (start_idx + 1 < len(tokens) and tokens[start_idx + 1] == '('):
            # 找到匹配的右括号
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
            
            # 提取括号内的参数
            args_tokens = tokens[start_idx + 2:end_idx]
            
            # 解析参数，处理逗号分隔
            parsed_args = self.parse_arguments(args_tokens)
            
            # 构建后缀表达式：参数 + 函数名
            result = parsed_args + [func_name]
            
            return result, end_idx + 1
        else:
            # 不是函数调用，直接返回token
            return [func_name], start_idx + 1
    
    def parse_arguments(self, args_tokens):
        """
        解析函数参数，处理逗号分隔的参数列表
        
        Parameters:
            args_tokens: 参数token列表
            
        Returns:
            list: 解析后的参数token列表
        """
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
        """
        解析表达式token，处理函数调用和运算符
        
        Parameters:
            tokens: token列表
            
        Returns:
            list: 解析后的后缀表达式token列表
        """
        if not tokens:
            return []
        
        parsed_tokens = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if self.is_operator(token):
                # 检查是否为函数调用格式
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
        """
        解析表达式，支持多种格式
        
        Parameters:
            expression: 表达式字符串
            
        Returns:
            list: 后缀表达式token列表
        """
        # 词法分析
        tokens = self.tokenize(expression)
        
        if not tokens:
            raise ValueError("表达式为空")
        
        # 检查表达式格式并解析
        if '(' in tokens:
            # 包含括号，可能是函数调用格式或中缀表达式
            parsed_tokens = self.parse_expression_tokens(tokens)
            
            # 如果解析后仍包含中缀运算符，转换为后缀
            if any(op in parsed_tokens for op in ['+', '-', '*', '/', 'add', 'subtract', 'multiply', 'divide']):
                return self.infix_to_postfix(parsed_tokens)
            else:
                return parsed_tokens
        else:
            # 不包含括号，可能是简单的中缀表达式
            if any(op in tokens for op in ['+', '-', '*', '/', 'add', 'subtract', 'multiply', 'divide']):
                return self.infix_to_postfix(tokens)
            else:
                # 假设已经是后缀表达式或简单表达式
                return tokens
    
    def evaluate_postfix(self, postfix_tokens):
        """
        计算后缀表达式（逆波兰表达式）
        
        Parameters:
            postfix_tokens: 后缀表达式token列表
            
        Returns:
            numpy.ndarray: 计算结果
        """
        if not postfix_tokens:
            raise ValueError("后缀表达式为空")
        
        stack = []
        
        for token in postfix_tokens:
            if self.is_feature(token):
                # 特征数据
                if self.calculator is None:
                    raise ValueError("需要Calculator实例来获取特征数据")
                
                feature_data = self.calculator.get_feature(token)
                stack.append(feature_data)
            
            elif self.is_number(token):
                # 数字常量
                constant_value = float(token)
                # 创建与特征数据相同形状的常量数组
                if stack and hasattr(stack[-1], 'shape'):
                    constant_array = np.full(stack[-1].shape, constant_value)
                else:
                    # 如果栈为空，创建标量
                    constant_array = constant_value
                stack.append(constant_array)
            
            elif self.is_operator(token):
                # 算子操作
                op_info = self.get_operator_info(token)
                operator_func = op_info['function']
                param_count = op_info['param_count']
                
                if len(stack) < param_count:
                    raise ValueError(f"算子 '{token}' 需要 {param_count} 个参数，但栈中只有 {len(stack)} 个")
                
                # 从栈中弹出参数（注意顺序）
                args = []
                for _ in range(param_count):
                    args.append(stack.pop())
                
                # 参数顺序需要反转（因为栈是后进先出）
                args.reverse()
                
                # 调用算子函数
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
        """
        计算表达式
        
        Parameters:
            expression: 表达式字符串
            
        Returns:
            numpy.ndarray: 计算结果
        """
        # 解析表达式为后缀形式
        postfix_tokens = self.parse_expression(expression)
        
        # 计算后缀表达式
        result = self.evaluate_postfix(postfix_tokens)
        
        return result
    
    def to_infix_notation(self, postfix_tokens):
        """
        将后缀表达式转换为中缀表达式（用于调试和显示）
        
        Parameters:
            postfix_tokens: 后缀表达式token列表
            
        Returns:
            str: 中缀表达式字符串
        """
        stack = []
        
        for token in postfix_tokens:
            if self.is_feature(token) or self.is_number(token):
                stack.append(token)
            elif self.is_operator(token):
                op_info = self.get_operator_info(token)
                param_count = op_info['param_count']
                
                if len(stack) < param_count:
                    raise ValueError(f"算子 '{token}' 需要 {param_count} 个参数")
                
                if param_count == 1:
                    # 一元算子
                    arg = stack.pop()
                    expr = f"{token}({arg})"
                elif param_count == 2:
                    # 二元算子
                    arg2 = stack.pop()
                    arg1 = stack.pop()
                    
                    # 对于基本算术运算，使用中缀形式
                    if token in ['+', '-', '*', '/', 'add', 'subtract', 'multiply', 'divide']:
                        op_symbol = token if token in ['+', '-', '*', '/'] else token
                        expr = f"({arg1} {op_symbol} {arg2})"
                    else:
                        expr = f"{token}({arg1}, {arg2})"
                else:
                    # 多元算子
                    args = []
                    for _ in range(param_count):
                        args.append(stack.pop())
                    args.reverse()
                    expr = f"{token}({', '.join(args)})"
                
                stack.append(expr)
        
        if len(stack) != 1:
            raise ValueError("表达式转换错误")
        
        return stack[0]
    
    def debug_parse(self, expression):
        """
        调试解析过程，显示详细信息
        
        Parameters:
            expression: 表达式字符串
        """
        print(f"=== 调试解析表达式: {expression} ===")
        
        # 词法分析
        tokens = self.tokenize(expression)
        print(f"词法分析结果: {tokens}")
        
        # 解析为后缀表达式
        postfix_tokens = self.parse_expression(expression)
        print(f"后缀表达式: {postfix_tokens}")
        
        # 转换为中缀表达式（用于验证）
        try:
            infix_expr = self.to_infix_notation(postfix_tokens)
            print(f"中缀表达式: {infix_expr}")
        except Exception as e:
            print(f"中缀转换失败: {e}")
        
        # 分析token类型
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


def create_formula_parser(calculator=None, data_dimension="3D"):
    """
    创建表达式解析器的便捷函数
    
    Parameters:
        calculator: Calculator实例
        data_dimension: 数据维度，"3D"或"4D"
        
    Returns:
        FormulaParser: 表达式解析器实例
    """
    return FormulaParser(calculator=calculator, data_dimension=data_dimension)
