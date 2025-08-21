# og_ops 框架总结

## 1. 框架整体架构

### 1.1 核心组件
```
og_ops/
├── Data_Provider/           # 数据提供模块
│   ├── Data_Provider.py    # 主要数据提供类
│   ├── download.py         # 数据下载功能
│   ├── process.py          # 数据处理功能
│   ├── read.py            # 数据读取功能
│   └── save.py            # 数据保存功能
├── operators_3D.py         # 3D数据算子库
├── operators_4D.py         # 4D数据算子库
├── utils.py               # 工具函数
├── .env                   # 环境变量配置
└── Demo.ipynb            # 使用示例
```

### 1.2 设计理念
- **数据分离**: 3D分钟数据和4D基本面数据分别处理
- **算子分离**: 针对不同维度数据提供专门的算子库
- **统一接口**: Calculator类提供统一的算子调用接口
- **并行处理**: 支持多进程并行下载和处理数据
- **形状兼容**: 自动处理不同维度数据的形状变换

## 2. 3D分钟数据处理流程

### 2.1 数据流程概览
```
原始分钟数据 → 下载 → 特征计算 → 处理 → 存储 → 读取 → Calculator
```

### 2.2 详细处理步骤

#### Step 1: 数据下载 (download_Minute)
- **输入**: 日期, 模式('five_minute')
- **数据源**: `/data/cephfs/minute/five_minute/{date}.parquet`
- **功能**: 从原始分钟数据计算7个日内特征
- **输出**: `./Minute/raw/{date}.parquet`

**特征计算 (cal_miniute_data)**:
```python
# 输入: 原始分钟数据 (security_code, trading_day, start_time, minute_return)
# 计算7个特征:
early_ret = 开盘前6分钟累计收益     # np.prod(x.head(6) + 1) - 1
tail_ret = 收盘前6分钟累计收益      # np.prod(x.tail(6) + 1) - 1  
max_ret = 日内最大收益率           # max()
min_ret = 日内最小收益率           # min()
mean_ret = 日内平均收益率          # mean()
intra_vol = 日内波动率            # std()
intra_skew = 日内偏度             # skew()
```

#### Step 2: 数据处理 (process_Minute)
- **输入**: `./Minute/raw/{date}.parquet`
- **功能**: 
  - 获取交易日股票池
  - 填充完整股票池 (对齐secucode)
  - 按SecuCode排序
  - 分特征保存
- **输出**: `./Minute/Lib/{feature_name}/{date}.parquet`

**形状变化**:
```
原始数据: (任意股票数, 7特征) 
→ 处理后: (N股票, 7特征) # N为完整股票池大小
→ 分特征存储: 每个特征单独文件 (N,)
```

#### Step 3: 数据读取 (read_Minute)
- **输入**: 日期, 特征名
- **功能**: 读取单个特征的单日数据
- **输出**: numpy数组 (N,) # N为股票数

#### Step 4: 批量获取 (Minute_Provider.get_data)
- **输入**: 日期范围, 特征名列表
- **功能**: 
  - 并行读取多日多特征数据
  - 堆叠成3D数组
  - 创建特征映射
- **输出**: 
  - `data_3d`: (D, N, M) # D=天数, N=股票数, M=特征数
  - `feature_map`: {x1: 'early_ret', x2: 'tail_ret', ...}

### 2.3 最终输入Calculator的形状
```python
# 3D分钟数据
data_3d.shape = (D, N, M)  # (天数, 股票数, 特征数)
# 例如: (22, 5106, 7) = (22天, 5106只股票, 7个特征)

# Calculator中每个特征的形状
feature_data.shape = (D, N)  # (天数, 股票数)
# 例如: (22, 5106) = (22天, 5106只股票)
```

## 3. 4D基本面数据处理流程

### 3.1 数据流程概览
```
基本面数据库 → 下载 → 处理重构 → 存储 → 读取 → Calculator
```

### 3.2 详细处理步骤

#### Step 1: 数据下载 (download_Fundamental)
- **输入**: 日期, 表名(如'Fundamental_Item1353')
- **数据源**: OceanBase数据库 `Fundamental.{table_name}`
- **功能**: 下载指定日期的基本面数据
- **输出**: `./Fundamental/raw/{table_name}/{date}.parquet`

#### Step 2: 数据处理 (process_Fundamental)
- **输入**: `./Fundamental/raw/{table_name}/{date}.parquet`
- **核心处理逻辑**:
  ```python
  # 处理12个财务区间 (EndDateRank 1-12)
  for i in range(12):
      subset = data.loc[data['EndDateRank']==i+1]
      # 对齐股票池
      subset_ = pd.DataFrame(np.full((len(secucode), data.shape[1]), np.nan))
      cond = secucode.isin(subset['SecuCode'].values)
      subset_.loc[cond] = subset.values
      new_data.append(subset_)
  ```
- **输出**: `./Fundamental/Lib/{feature_name}/{date}.parquet`

**形状变化**:
```
原始数据: (变长, 多列) # 不同EndDateRank的记录数不同
→ 重构后: (N*12, 多列) # N股票 × 12财务区间
→ 存储: 每个特征单独文件
```

#### Step 3: 数据读取 (read_Fundamental)
- **输入**: 日期, 特征名
- **功能**: 读取并重塑为(N, 12)形状
- **输出**: numpy数组 (N, 12) # N股票 × 12财务区间

#### Step 4: 批量获取 (Fundamental_Provider.get_data)
- **输入**: 日期范围, 特征名列表
- **功能**:
  - 并行读取多月多特征数据
  - 转置和堆叠: `(M, N, 12) → (N, 12, M) → (D, N, 12, M)`
- **输出**: `data_4d`: (D, N, T, M) # D=月数, N=股票数, T=财务区间数, M=特征数

### 3.3 最终输入Calculator的形状
```python
# 4D基本面数据
data_4d.shape = (D, N, T, M)  # (月数, 股票数, 财务区间数, 特征数)
# 例如: (12, 100, 12, 5) = (12个月, 100只股票, 12个财务区间, 5个特征)

# Calculator中每个特征的形状
feature_data.shape = (D, N, T)  # (月数, 股票数, 财务区间数)
# 例如: (12, 100, 12) = (12个月, 100只股票, 12个财务区间)
```

## 4. Calculator兼容性设计

### 4.1 统一接口设计
Calculator类通过以下机制实现对3D和4D数据的兼容:

#### 4.1.1 自动维度检测
```python
def __init__(self, data, feature_names=None, feature_map=None):
    self.ndim = data.ndim  # 自动检测数据维度
    if self.ndim == 3:
        import operators_3D
        self.operators = operators_3D
    else:  # 4D
        import operators_4D  
        self.operators = operators_4D
```

#### 4.1.2 特征提取适配
```python
def get_feature(self, identifier):
    idx = self.get_feature_index(identifier)
    if self.ndim == 3:
        return self.data[:, :, idx]      # 返回 (D, N)
    else:  # 4D
        return self.data[:, :, :, idx]   # 返回 (D, N, T)
```

#### 4.1.3 结果堆叠适配
```python
def _stack_results(self, results):
    if self.ndim == 3:
        return np.stack(results, axis=2)  # (D, N, len(results))
    else:  # 4D
        return np.stack(results, axis=3)  # (D, N, T, len(results))
```

### 4.2 特征访问统一化
支持三种访问方式，自动映射到正确索引:
```python
# 方式1: 特征名
calc.get_feature('early_ret')    # 3D数据
calc.get_feature('PE')           # 4D数据

# 方式2: x索引  
calc.get_feature('x1')           # 对应第一个特征

# 方式3: 数字索引
calc.get_feature(0)              # 对应第一个特征
```

### 4.3 算子调用统一化
```python
# 单参数算子
result = calc.apply_unary_operator('abs_val', 'feature_name')

# 双参数算子  
result = calc.apply_binary_operator('add', 'feature1', 'feature2')

# 批量应用
result = calc.apply_operator('rank')  # 应用到所有特征
```

## 5. Calculator使用指南

### 5.1 初始化
```python
# 3D数据
calc_3d = Calculator(data_3d, feature_map=feature_map)

# 4D数据  
calc_4d = Calculator(data_4d, feature_names=['PE', 'ROE', 'ROA', 'PB', 'EPS'])
```

### 5.2 基本操作
```python
# 查看信息
calc.info()
calc.list_operators()

# 特征访问
feature_data = calc.get_feature('feature_name')
multi_features = calc.get_features(['feat1', 'feat2'])

# 特征比较
is_equal = calc.features_equal('early_ret', 'x1')
```

### 5.3 算子应用

#### 5.3.1 算子调用逻辑
Calculator通过自动检测算子函数的参数数量来决定调用方式：

```python
def apply_operator(self, operator_name, features=None, **kwargs):
    # 获取算子函数
    operator_func = getattr(self.operators, operator_name)
    
    # 使用inspect检查函数签名
    import inspect
    sig = inspect.signature(operator_func)
    
    # 计算没有默认值的必需参数数量
    param_count = len([p for p in sig.parameters.values() 
                     if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) 
                     and p.default == p.empty])
    
    if param_count == 1:
        # 单参数算子：abs_val, rank, ts_mean等
        result = operator_func(feature_data, **kwargs)
        
    elif param_count == 2:
        # 双参数算子：add, ts_corr, rolling_corr等
        if len(feature_data_list) != 2:
            raise ValueError(f"算子需要2个特征，但提供了{len(feature_data_list)}个")
        result = operator_func(feature_data_list[0], feature_data_list[1], **kwargs)
        
    elif param_count >= 3:
        # 多参数算子：直接传递所有特征
        result = operator_func(*feature_data_list, **kwargs)
```

**调用分类说明：**
- **单参数算子**: 只需要一个输入特征的算子（如数学函数、时序算子、截面算子）
- **双参数算子**: 需要两个输入特征的算子（如算术运算、相关性计算）
- **多参数算子**: 需要三个或更多输入特征的算子（如复杂组合算子）

#### 5.3.2 3D算子示例
```python
# 数学函数
abs_result = calc_3d.apply_unary_operator('abs_val', 'early_ret')
log_result = calc_3d.apply_unary_operator('log', 'tail_ret')

# 时序算子
ma_result = calc_3d.apply_unary_operator('ts_mean', 'early_ret', window=5)
lag_result = calc_3d.apply_unary_operator('ts_lag', 'early_ret', periods=1)

# 截面算子
rank_result = calc_3d.apply_unary_operator('rank', 'early_ret')
zscore_result = calc_3d.apply_unary_operator('zscore', 'early_ret')

# 双参数算子
add_result = calc_3d.apply_binary_operator('add', 'early_ret', 'tail_ret')
corr_result = calc_3d.apply_binary_operator('ts_corr', 'early_ret', 'tail_ret', window=10)
```

#### 5.3.2 4D算子示例
```python
# 基本算子
abs_result = calc_4d.apply_unary_operator('custom_abs', 'PE')
norm_result = calc_4d.apply_unary_operator('cross_sectional_normalize', 'PE')

# 滚动算子
rolling_mean = calc_4d.apply_unary_operator('rolling_mean', 'PE', window=6, axis=-1)
rolling_corr = calc_4d.apply_binary_operator('rolling_corr', 'PE', 'ROE', window=5, axis=-1)

# 截面算子
rank_result = calc_4d.apply_unary_operator('cross_sectional_rank', 'PE')
neut_result = calc_4d.apply_binary_operator('cross_sectional_neutralize', 'PE', 'ROE')
```

### 5.4 高级用法

#### 5.4.1 算子链式调用
```python
# 计算移动平均
ma1 = calc_3d.apply_unary_operator('ts_mean', 'early_ret', window=3)
ma2 = calc_3d.apply_unary_operator('ts_mean', 'tail_ret', window=3)

# 创建新Calculator处理结果
combined_data = np.stack([ma1, ma2], axis=2)
combined_calc = Calculator(combined_data, feature_names=['ma1', 'ma2'])

# 计算比值并排名
ratio = combined_calc.apply_binary_operator('divide', 'ma1', 'ma2')
ratio_calc = Calculator(ratio.reshape(*ratio.shape, 1), feature_names=['ratio'])
final_factor = ratio_calc.apply_unary_operator('rank', 'ratio')
```

#### 5.4.2 性能测试
```python
# 创建测试数据
large_3d = np.random.randn(252, 5000, 10)
large_calc = Calculator(large_3d, feature_names=[f'feature_{i}' for i in range(10)])

# 测试算子性能
algorithms = [
    ('abs_val', 'feature_0', {}),
    ('rank', 'feature_1', {}),
    ('ts_mean', 'feature_2', {'window': 20}),
    # ... 更多算子
]

for algo_name, features, kwargs in algorithms:
    start_time = time.time()
    result = large_calc.apply_unary_operator(algo_name, features, **kwargs)
    end_time = time.time()
    print(f"{algo_name}: {end_time - start_time:.4f}s")
```

## 6. 关键技术特性

### 6.1 数据对齐机制
- **股票池对齐**: 所有数据都对齐到统一的secucode序列
- **时间对齐**: 3D数据按交易日对齐，4D数据按月度对齐
- **缺失值处理**: 统一使用np.nan填充缺失值

### 6.2 并行处理
- **下载并行**: 使用multiprocessing.Pool并行下载数据
- **处理并行**: 并行处理不同日期的数据
- **读取并行**: 并行读取多个特征的数据

### 6.3 内存优化
- **按需加载**: 只加载指定日期范围和特征的数据
- **分文件存储**: 每个特征单独存储，减少I/O开销
- **数据类型优化**: 使用适当的数据类型减少内存占用

### 6.4 错误处理
- **数据验证**: 验证特征名、算子名的有效性
- **异常捕获**: 处理数据读取、算子调用中的异常
- **调试支持**: 提供debug_feature_mapping等调试工具

## 7. 扩展性设计

### 7.1 新算子添加
- 在operators_3D.py或operators_4D.py中添加新函数
- Calculator会自动识别并提供调用接口

### 7.2 新数据类型支持
- 继承Data_Provider基类
- 实现download、process、get_data方法
- 遵循统一的数据形状约定

### 7.3 新维度支持
- 扩展Calculator的维度检测逻辑
- 添加对应的算子库
- 实现相应的特征提取和结果堆叠逻辑

## 8. 总结

og_ops框架是一个专为金融数据处理设计的高效、灵活的计算框架，具有以下核心优势:

1. **统一接口**: Calculator类提供统一的算子调用接口，屏蔽底层数据维度差异
2. **高性能**: 支持并行处理和内存优化，适合大规模数据计算
3. **易扩展**: 模块化设计，易于添加新算子和新数据类型
4. **类型安全**: 自动处理数据形状变换和类型转换
5. **调试友好**: 提供丰富的调试和验证工具

该框架成功解决了3D分钟数据和4D基本面数据的统一处理问题，为量化研究提供了强大的技术支撑。
