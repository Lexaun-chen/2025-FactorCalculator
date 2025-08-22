import os
import random
import itertools
import numpy as np
import pandas as pd
from operators import * 
from collections import defaultdict
from sklearn.cluster import SpectralClustering

def invformula(f):
    if not '(' in f:
        return RNode(name=f)
    op, _, content = f.partition('(')
    content = content[:-1]
    if op in UNARY_OP or op in CROSS_OP:
        return Node(op, parents=[invformula(content)])
    elif op in ROLLING_UNARY_OP:
        content, _, window = content.rpartition(',')
        return Node(op, parents=[invformula(content)], window=int(window[1:]))
    elif op in SM_BINARY_OP or op in ASM_BINARY_OP:
        bracket_count = 0
        for i, s in enumerate(content):
            if s == '(':
                bracket_count += 1
            elif s == ')':
                bracket_count -= 1
            elif s == ',':
                if bracket_count == 0:
                    f1, f2 = content[:i], content[i+2:]
                    break
        return Node(op, parents=[invformula(f1), invformula(f2)])
    elif op in SM_ROLLING_BINARY_OP or op in ASM_ROLLING_BINARY_OP:
        content, _, window = content.rpartition(',')
        bracket_count = 0
        for i, s in enumerate(content):
            if s == '(':
                bracket_count += 1
            elif s == ')':
                bracket_count -= 1
            elif s == ',':
                if bracket_count == 0:
                    f1, f2 = content[:i], content[i+2:]
                    break
        return Node(op, window=int(window[1:]), parents=[invformula(f1), invformula(f2)])

class RNode:
    def __init__(self, name):
        self.name = name

    def calculate(self, data_dir):
        data = np.load(os.path.join(data_dir, '{}.npy'.format(self.name)))
        return data

    def formula(self):
        return self.name

class Node:
    def __init__(self, operator, window=None, parents=None):
        self.window = window
        self.operator = operator
        self.parents = parents

    def calculate(self, data_dir):
        if self.operator in UNARY_OP + CROSS_OP:
            pdata = self.parents[0].calculate(data_dir)
            axis = 0 if pdata.ndim == 2 else -1
            if self.operator == "custom_abs":
                value = custom_abs(pdata)
            elif self.operator == "sign":
                value = sign(pdata)
            elif self.operator == "log":
                value = log(pdata)
            elif self.operator == "sqrt":
                value = sqrt(pdata)
            elif self.operator == "square":
                value = square(pdata)
            elif self.operator == "cs_norm":
                value = cross_sectional_normalize(pdata)
            elif self.operator == "cs_rank":
                value = cross_sectional_rank(pdata)
        elif self.operator in ROLLING_UNARY_OP:
            pdata = self.parents[0].calculate(data_dir)
            axis = 0 if pdata.ndim == 2 else -1
            if self.operator == "delay":
                value = delay(pdata, self.window, axis=axis)
            elif self.operator == "delta":
                value = delta(pdata, self.window, axis=axis)     
            elif self.operator == "pctchange":
                value = pctchange(pdata, self.window, axis=axis)       
            elif self.operator == "rolling_sum":
                value = rolling_sum(pdata, self.window, axis=axis)
            elif self.operator == "rolling_rank":
                value = rolling_rank(pdata, self.window, axis=axis)
            elif self.operator == "rolling_mean":
                value = rolling_mean(pdata, self.window, axis=axis)
            elif self.operator == "rolling_std":
                value = rolling_std(pdata, self.window, axis=axis)
            elif self.operator == "rolling_max":
                value = rolling_max(pdata, self.window, axis=axis)
            elif self.operator == "rolling_min":
                value = rolling_min(pdata, self.window, axis=axis)
            elif self.operator == "rolling_argmax":
                value = rolling_argmax(pdata, self.window, axis=axis)
            elif self.operator == "rolling_argmin":
                value = rolling_argmin(pdata, self.window, axis=axis)
            elif self.operator == "rolling_cumulative_product":
                value = rolling_cumulative_product(pdata, self.window)
            elif self.operator == "rolling_weighted_mean_linear_decay":
                value = rolling_weighted_mean_linear_decay(pdata, self.window, axis=axis)
        elif self.operator in SM_BINARY_OP + ASM_BINARY_OP:
            fdata, mdata = self.parents[0].calculate(data_dir), self.parents[1].calculate(data_dir)
            axis = 0 if fdata.ndim == 2 else -1
            if self.operator == "add":
                value = add(fdata, mdata)
            elif self.operator == "minus":
                value = minus(fdata, mdata)
            elif self.operator == "multiply":
                value = multiply(fdata, mdata)
            elif self.operator == "divide":
                value = divide(fdata, mdata)
            elif self.operator == "isover":
                value = isover(fdata, mdata)
            elif self.operator == "custom_min":
                value = custom_min(fdata, mdata)
            elif self.operator == "custom_max":
                value = custom_max(fdata, mdata)
            elif self.operator == "signedpower":
                value = signedpower(fdata, mdata)
            elif self.operator == "cs_neut":
                value = cross_sectional_neutralize(fdata, mdata)
        elif self.operator in SM_ROLLING_BINARY_OP + ASM_ROLLING_BINARY_OP:
            fdata, mdata = self.parents[0].calculate(data_dir), self.parents[1].calculate(data_dir)
            axis = 0 if fdata.ndim == 2 else -1
            if self.operator == "rolling_covariance":
                value = rolling_covariance(fdata, mdata, self.window, axis=axis)
            elif self.operator == "rolling_corr":
                value = rolling_corr(fdata, mdata, self.window, axis=axis)
        return value

    def formula(self):
        if self.operator in SM_BINARY_OP + ASM_BINARY_OP:
            result = self.operator + '(' + self.parents[0].formula() + ', ' + self.parents[1].formula() + ')'
        elif self.operator in ROLLING_UNARY_OP:
            result = self.operator + '(' + self.parents[0].formula() + ', ' + str(self.window) + ')'
        elif self.operator in UNARY_OP + CROSS_OP:
            result = self.operator + '(' + self.parents[0].formula() + ')'
        elif self.operator in SM_ROLLING_BINARY_OP + ASM_ROLLING_BINARY_OP:
            result = self.operator + '(' + self.parents[0].formula() + ', ' + self.parents[1].formula() + ', ' + str(self.window) + ')'
        else:
            raise NameError
        return result

if __name__ == '__main__':
    example = 'rolling_corr(rolling_mean(close, 5), multiply(b, c), 10)'
    node = invformula(example)
    ## 通过invformula函数和formula方法可以实现因子表达式（字符串）和一个Node实例间的转换，可以在Node的calculate方法中自定义计算规则 ##
    print(node.formula())
    value = node.calculate(data_dir='')
    ## 这一行是跑不通的，需要Data_Provider提供数据存储地址才行 ##
    