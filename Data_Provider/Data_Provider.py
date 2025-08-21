
# 标准库导入
import os
import sys
import inspect
from pathlib import Path
from multiprocessing import Pool

# 第三方库导入
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy import create_engine 

# 本地模块导入 - 使用绝对导入
# 添加当前目录到Python路径
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from read import *
from save import *
from process import *
from download import *

# 父级模块导入 - 使用绝对导入
# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入项目模块
from utils import *

# 加载环境变量 - 指定.env文件路径
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Data_Provider:
    def __init__(self):
        raise NotImplementedError
    
    def download(self):
        raise NotImplementedError
    
    def get_data(self):
        raise NotImplementedError    

class ReturnDaily_Provider(Data_Provider):
    def __init__(self):
        if not os.path.exists("./ReturnDaily"):
            os.makedirs("./ReturnDaily")  

    def download(self, start, end):
        trading_day = get_trading_day(start, end)
        conn = "mysql://{}%40public%23Thetis:{}@192.168.55.161:2883/".format(os.getenv("OB_USER"), os.getenv("OB_PASSWORD"))
        for date in tqdm(trading_day): 
            sql_query = "select * from SmartQuant.ReturnDaily WHERE DataDate = '{}'".format(date)
            data = cx.read_sql(conn, sql_query)
            data.to_parquet('./ReturnDaily/{}.parquet'.format(date), index=False)

class Fundamental_Provider(Data_Provider):
    def __init__(self, start="2006-12-01", end="2025-07-31", workers=32):
        self.workers = workers
        self.start, self.end = start, end
        self.secucode = pd.Series(get_secucode(self.start, self.end))
        if not os.path.exists("./Fundamental"):
            os.makedirs("./Fundamental")
            os.makedirs("./Fundamental/raw")
            os.makedirs("./Fundamental/Lib")
    
    def download(self, start="2006-12-01", end="2025-07-31", table_list=['Fundamental_Item1002','Fundamental_Item3152']):
        for name in table_list:
            if not os.path.exists("./Fundamental/raw/{}".format(name)):
                os.makedirs("./Fundamental/raw/{}".format(name))
        with Pool(self.workers) as p:
            for date in tqdm(get_month_first_trading_day(start, end)):
                tasks = [(date, name) for name in table_list]
                p.starmap(download_Fundamental, tasks)
    
    def process(self, start="2006-12-01", end="2025-07-31", table_list=['Fundamental_Item1002','Fundamental_Item3152']):
        with Pool(self.workers) as p:
            for date in tqdm(get_month_first_trading_day(start, end)):
                tasks = [(date, name, self.secucode) for name in table_list]
                p.starmap(process_Fundamental, tasks)

    def get_data(self, start, end, feature_names):
        data_list = []
        dates = get_month_first_trading_day(start, end)
        with Pool(self.workers) as p:
            for date in tqdm(dates, desc="loading data from disk"):
                tasks = [(date, name) for name in feature_names]
                data = p.starmap(read_Fundamental, tasks)
                data_list.append(np.asarray(data).transpose(1, 2, 0))  # (M, N, 12) ->（N, 12, M)
        data = np.asarray(data_list) #（D, N, 12, M)
        return data



class Minute_Provider(Data_Provider):
    def __init__(self, start="2016-06-20", end="2025-07-31", workers=12):
        self.workers = workers
        self.start, self.end = start, end
        self.secucode = pd.Series(get_secucode(self.start, self.end))
        
        # 定义分钟数据的特征列表
        self.feature_list = ['early_ret', 'tail_ret', 'max_ret', 'min_ret', 'mean_ret', 'intra_vol', 'intra_skew']
        
        if not os.path.exists("./Minute"):
            os.makedirs("./Minute")
            os.makedirs("./Minute/raw")
            os.makedirs("./Minute/Lib")
    
    def download(self, start="2016-06-20", end="2025-07-31", mode='five_minute'):
        """
        下载分钟数据
        
        Parameters:
            start: 开始日期
            end: 结束日期  
            mode: 数据模式，默认'five_minute'
        """
        
        # 下载数据
        dates = get_trading_day(start, end)
        for date in tqdm(dates, desc="下载分钟数据"):
            try:
                download_Minute(date, mode)
            except Exception as e:
                print(f"下载日期 {date} 失败: {e}")
                continue
    
    def process(self, start="2016-06-20", end="2025-07-31"):
        """
        处理分钟数据
        """
        dates = get_trading_day(start, end)
        # 对每个日期，处理所有特征
        with Pool(self.workers) as p:
            for date in tqdm(dates, desc="处理分钟数据"):
                try:
                    process_Minute(date, None, self.secucode)
                except Exception as e:
                    print(f"处理日期 {date} 失败: {e}")
                    continue

    def get_data(self, start, end, feature_names = None):
        """
        获取处理后的分钟数据
        
        Parameters:
            start: 开始日期
            end: 结束日期
            feature_names: 要获取的特征名列表
            
        Returns:
            data: numpy数组，形状为(D, N, M)
            feature_map: 特征映射字典
        """
        
        if feature_names is None:
            feature_names = self.feature_list
        else:
            # 验证特征名是否有效
            invalid_features = [f for f in feature_names if f not in self.feature_list]
            if invalid_features:
                raise ValueError(f"无效的特征名: {invalid_features}. 可用特征: {self.feature_list}")
        
        feature_map = {f'x{i+1}':name for i,name in enumerate(feature_names)}
        data_list = []
        dates = get_trading_day(start, end)
        
        with Pool(self.workers) as p:
            for date in tqdm(dates, desc="loading data from disk"):
                try:
                    tasks = [(date, name) for name in feature_names]
                    daily_data = p.starmap(read_Minute, tasks) #每个特征 (N,)
                    data_list.append(np.column_stack(daily_data))   # (N, M)
                except Exception as e:
                    print(f"读取日期 {date} 数据失败: {e}")
                    # 创建空数据填充
                    empty_data = np.full((len(self.secucode), len(feature_names)), np.nan)
                    data_list.append(empty_data)
                    
        data = np.stack(data_list) # (D, N, M)
        return data, feature_map
    
    def get_available_features(self):
        """返回可用的特征列表"""
        return self.feature_list.copy()


class Calculator:
    """
    传入数据形状：
    - 3D数据 (minute数据): (D, N, M) - D=天数, N=股票数, M=特征数
    - 4D数据 (fundamental数据): (D, N, T, M) - D=天数, N=股票数, T=财务区间数(12), M=特征数
    
    算子调用:
    - 3D数据: 每个特征传入算子时为(D, N)的2D数组
    - 4D数据: 每个特征传入算子时为(D, N, T)的3D数组
    """
    
    def __init__(self, data, feature_names=None, feature_map=None):
        """
        Parameters:
            data: numpy数组，3D或4D
            feature_names: 特征名称列表 ['Close', 'Volume', ...]
            feature_map: 特征映射字典 {x1: 'Close', x2: 'Volume', ...}，与feature_names二选一
        """
        self.data = data
        self.ndim = data.ndim
        
        if self.ndim not in [3, 4]:
            raise ValueError(f"数据必须是3D或4D，当前是{self.ndim}D")
        
        # 处理特征映射
        if feature_map is not None:
            self.feature_map = feature_map
            # 确保feature_map的键是按顺序的x1, x2, x3...
            sorted_keys = sorted(feature_map.keys(), key=lambda x: int(x[1:]))
            self.feature_names = [feature_map[key] for key in sorted_keys]
        elif feature_names is not None:
            self.feature_names = feature_names
            self.feature_map = {f'x{i+1}': name for i, name in enumerate(feature_names)}
        
        else:
            num_features = data.shape[-1]
            self.feature_names = [f'feature_{i+1}' for i in range(num_features)]
            self.feature_map = {f'x{i+1}': name for i, name in enumerate(self.feature_names)}
        
        # 反向映射 - 确保索引对应正确
        self.name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        
        # 为x索引创建正确的映射
        if feature_map is not None:
            # 使用排序后的键来确保正确的索引映射
            sorted_keys = sorted(feature_map.keys(), key=lambda x: int(x[1:]))
            self.x_to_idx = {key: i for i, key in enumerate(sorted_keys)}
        else:
            self.x_to_idx = {f'x{i+1}': i for i in range(len(self.feature_names))}
        
        # 导入对应的算子模块
        if self.ndim == 3:
            import operators_3D
            self.operators = operators_3D
        else:  # 4D
            import operators_4D
            self.operators = operators_4D
    
    
    def get_feature_index(self, identifier):
        """
        获取特征索引
        
        Parameters:
            identifier: 特征标识符，三种任选一:
                - 特征名 (如 'Close')
                - x索引 (如 'x1') 
                - 数字索引 (如 0)
        
        Returns:
            int: 特征索引
        """
        if isinstance(identifier, (int, np.integer)):
            return identifier
        elif isinstance(identifier, str):
            if identifier.startswith('x'):
                return self.x_to_idx[identifier]
            else:
                return self.name_to_idx[identifier]
        else:
            raise ValueError(f"无效的标识符类型: {type(identifier)}")
    
    
    def get_feature(self, identifier):
        """
        获取特定特征的数据
        
        Parameters:
            identifier: 特征标识符
        
        Returns:
            numpy.ndarray: 特征数据
                - 3D数据返回 (D, N) - 适合传入3D算子
                - 4D数据返回 (D, N, T) - 适合传入4D算子
        """
        idx = self.get_feature_index(identifier)
        if self.ndim == 3:
            return self.data[:, :, idx]  # (D, N)
        else:  # 4D
            return self.data[:, :, :, idx]  # (D, N, T)
    
    def get_features(self, identifiers):
        """
        获取多个特征的数据
        
        Parameters:
            identifiers: 特征标识符列表
        
        Returns:
            list: 特征数据列表，每个元素是对应特征的数据
        """
        return [self.get_feature(ident) for ident in identifiers]
    
    
    def apply_operator(self, operator_name, features=None, **kwargs):
        """
        应用算子到指定特征
        
        Parameters:
            operator_name: 算子名称 (字符串)
            features: None(所有特征) 或 特征标识符列表
            **kwargs: 算子参数
        
        Returns:
            numpy.ndarray: 计算结果
        """
        # 获取算子函数
        if hasattr(self.operators, operator_name):
            operator_func = getattr(self.operators, operator_name)
        else:
            raise ValueError(f"算子 '{operator_name}' 不存在")
        
        if features is None:
            # 应用到所有特征
            return self._apply_to_all_features(operator_func, **kwargs)
        else:
            # 应用到指定特征
            if not isinstance(features, (list, tuple)):
                features = [features]
            
            feature_data_list = self.get_features(features)
            
            # 根据算子参数数量决定调用方式
            import inspect
            sig = inspect.signature(operator_func)
            # 只计算没有默认值的必需参数
            param_count = len([p for p in sig.parameters.values() 
                             if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) 
                             and p.default == p.empty])
            
            if param_count == 1:
                # 单参数算子
                if len(feature_data_list) == 1:
                    # 单个特征，直接返回结果
                    result = operator_func(feature_data_list[0], **kwargs)
                    return result
                else:
                    # 多个特征，分别应用算子
                    results = []
                    for feature_data in feature_data_list:
                        result = operator_func(feature_data, **kwargs)
                        results.append(result)
                    return self._stack_results(results)
            
            elif param_count == 2:
                # 双参数算子
                if len(feature_data_list) != 2:
                    raise ValueError(f"算子 '{operator_name}' 需要2个特征，但提供了{len(feature_data_list)}个")
                result = operator_func(feature_data_list[0], feature_data_list[1], **kwargs)
                return result
            
            elif param_count >= 3:
                # 多参数算子，尝试直接调用
                try:
                    result = operator_func(*feature_data_list, **kwargs)
                    return result
                except Exception as e:
                    raise ValueError(f"算子 '{operator_name}' 调用失败: {e}")
            
            else:
                raise ValueError(f"不支持的算子参数数量: {param_count}")
    
    def _apply_to_all_features(self, operator_func, **kwargs):
        """应用算子到所有特征"""
        results = []
        for i in range(len(self.feature_names)):
            feature_data = self.get_feature(i)
            result = operator_func(feature_data, **kwargs)
            results.append(result)
        return self._stack_results(results)
    
    def _stack_results(self, results):
        """将结果列表堆叠成正确的形状"""
        if self.ndim == 3:
            # 3D数据: 结果应该是 (D, N, len(results))
            return np.stack(results, axis=2)
        else:  # 4D
            # 4D数据: 结果应该是 (D, N, T, len(results))
            return np.stack(results, axis=3)
    
    def apply_unary_operator(self, operator_name, feature, **kwargs):
        """
        应用单参数算子到单个特征
        
        Parameters:
            operator_name: 算子名称
            feature: 特征标识符
            **kwargs: 算子参数
        
        Returns:
            numpy.ndarray: 计算结果，形状与输入特征相同
        """
        return self.apply_operator(operator_name, features=[feature], **kwargs)
    
    def apply_binary_operator(self, operator_name, feature1, feature2, **kwargs):
        """
        应用双参数算子到两个特征
        
        Parameters:
            operator_name: 算子名称
            feature1: 第一个特征标识符
            feature2: 第二个特征标识符
            **kwargs: 算子参数
        
        Returns:
            numpy.ndarray: 计算结果
        """
        return self.apply_operator(operator_name, features=[feature1, feature2], **kwargs)
    
    def list_operators(self):
        """列出可用的算子"""
        operators = []
        for name in dir(self.operators):
            if not name.startswith('_') and callable(getattr(self.operators, name)):
                operators.append(name)
        return operators
    
    @property
    def shape(self):
        """数据形状"""
        return self.data.shape
    
    @property
    def num_features(self):
        """特征数量"""
        return len(self.feature_names)
    
    def info(self):
        """数据信息"""
        print(f"数据形状: {self.shape}")
        print(f"数据维度: {self.ndim}D")
        print(f"特征数量: {self.num_features}")
        print(f"特征名称: {self.feature_names}")
        print(f"特征映射: {self.feature_map}")
        print(f"可用算子数量: {len(self.list_operators())}")
    
    def debug_feature_mapping(self):
        """调试特征映射"""
        print("=== 特征映射调试信息 ===")
        print(f"feature_names: {self.feature_names}")
        print(f"feature_map: {self.feature_map}")
        print(f"name_to_idx: {self.name_to_idx}")
        print(f"x_to_idx: {self.x_to_idx}")
        print("\n=== 测试不同访问方式 ===")
        
        # 测试第一个特征的不同访问方式
        if len(self.feature_names) > 0:
            first_feature_name = self.feature_names[0]
            first_x_key = list(self.feature_map.keys())[0]
            
            print(f"第一个特征名: {first_feature_name}")
            print(f"对应的x键: {first_x_key}")
            
            # 通过特征名访问
            idx_by_name = self.get_feature_index(first_feature_name)
            print(f"通过特征名 '{first_feature_name}' 获取的索引: {idx_by_name}")
            
            # 通过x键访问
            idx_by_x = self.get_feature_index(first_x_key)
            print(f"通过x键 '{first_x_key}' 获取的索引: {idx_by_x}")
            
            # 通过数字索引访问
            idx_by_num = self.get_feature_index(0)
            print(f"通过数字索引 0 获取的索引: {idx_by_num}")
            
            print(f"三种方式索引是否一致: {idx_by_name == idx_by_x == idx_by_num}")
            
            # 测试实际数据访问
            print("\n=== 测试数据访问 ===")
            data_by_name = self.get_feature(first_feature_name)
            data_by_x = self.get_feature(first_x_key)
            data_by_idx = self.get_feature(0)
            
            print(f"数据形状一致: {data_by_name.shape == data_by_x.shape == data_by_idx.shape}")
            print(f"数据类型一致: {data_by_name.dtype == data_by_x.dtype == data_by_idx.dtype}")
            
            # NaN值分析
            nan_count_name = np.sum(np.isnan(data_by_name))
            nan_count_x = np.sum(np.isnan(data_by_x))
            nan_count_idx = np.sum(np.isnan(data_by_idx))
            total_elements = data_by_name.size
            
            print(f"NaN值数量 (name): {nan_count_name}/{total_elements} ({100*nan_count_name/total_elements:.1f}%)")
            print(f"NaN值数量 (x): {nan_count_x}/{total_elements} ({100*nan_count_x/total_elements:.1f}%)")
            print(f"NaN值数量 (idx): {nan_count_idx}/{total_elements} ({100*nan_count_idx/total_elements:.1f}%)")
            
            # 对于有效值的差异分析
            valid_mask = ~np.isnan(data_by_name)
            if np.any(valid_mask):
                valid_diff_name_x = np.max(np.abs(data_by_name[valid_mask] - data_by_x[valid_mask]))
                valid_diff_name_idx = np.max(np.abs(data_by_name[valid_mask] - data_by_idx[valid_mask]))
                valid_diff_x_idx = np.max(np.abs(data_by_x[valid_mask] - data_by_idx[valid_mask]))
                print(f"有效值最大差异 (name vs x): {valid_diff_name_x}")
                print(f"有效值最大差异 (name vs idx): {valid_diff_name_idx}")
                print(f"有效值最大差异 (x vs idx): {valid_diff_x_idx}")
            else:
                print("所有值都是NaN，无法计算有效值差异")
            
            # 使用新的features_equal方法
            print(f"特征相等 (name vs x): {self.features_equal(first_feature_name, first_x_key)}")
            print(f"特征相等 (name vs idx): {self.features_equal(first_feature_name, 0)}")
            print(f"特征相等 (x vs idx): {self.features_equal(first_x_key, 0)}")
    
    def features_equal(self, identifier1, identifier2, rtol=1e-15, atol=1e-15):
        """
        比较两个特征是否相等，正确处理NaN值
        
        Parameters:
            identifier1: 第一个特征标识符
            identifier2: 第二个特征标识符
            rtol: 相对容差
            atol: 绝对容差
            
        Returns:
            bool: 特征是否相等
        """
        data1 = self.get_feature(identifier1)
        data2 = self.get_feature(identifier2)
        
        # 检查形状是否相同
        if data1.shape != data2.shape:
            return False
        
        # 检查NaN位置是否相同
        nan_mask1 = np.isnan(data1)
        nan_mask2 = np.isnan(data2)
        if not np.array_equal(nan_mask1, nan_mask2):
            return False
        
        # 对于非NaN值，使用allclose比较
        valid_mask = ~(nan_mask1 | nan_mask2)
        if np.any(valid_mask):
            return np.allclose(data1[valid_mask], data2[valid_mask], rtol=rtol, atol=atol)
        else:
            # 如果所有值都是NaN，则认为相等
            return True
