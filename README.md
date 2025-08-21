# 2025-FactorCalculator
因子自动挖掘框架 3D数据 or 4D数据
## DataProvider

熟悉openfe,gp数据处理的方式并检查正确性，进行整合，针对不同来源数据，构建一个标准化的数据处理模块


以returndaily每日股票作为基准股票池
处理成3D or 4D

## Caculator

熟悉openfe,gp算子，检查定义正确性，比较性能差异，统一算子定义


检查定义和实现是否一致
检查极端情况的处理是否正确(0值，空值等)
选择性能最优的方式
在每个算子函数下方，标明算子定义和参数定义，示例如下：

<img width="445" height="237" alt="image" src="https://github.com/user-attachments/assets/3042b19a-14f8-4d49-8358-c88fdd00b1cd" />


gp已有算子定义


<img width="1071" height="692" alt="image" src="https://github.com/user-attachments/assets/75456a59-1a2b-4d31-a688-09ccf2db174d" />

<img width="1087" height="748" alt="image" src="https://github.com/user-attachments/assets/2419cd16-6a54-493d-91a3-85abae0e17bb" />

<img width="916" height="347" alt="image" src="https://github.com/user-attachments/assets/aafeaf65-0ae3-4587-88c1-b70529c79370" />

扩充算子

扩充列表
<img width="1120" height="463" alt="image" src="https://github.com/user-attachments/assets/4920995a-ec6a-4fb6-bb39-928b14d1d0c8" />

<img width="1120" height="463" alt="image" src="https://github.com/user-attachments/assets/81ceba91-92de-4830-9d8b-e425a6ac8920" />
<img width="1121" height="851" alt="image" src="https://github.com/user-attachments/assets/1ae2b5b4-b3e2-448b-b905-52dd24791eed" />
<img width="1078" height="135" alt="image" src="https://github.com/user-attachments/assets/bc757f45-cb35-46e0-a1ba-dc224cc31e24" />




## 公式转化

利用逆波兰表达式解析公式表达式并调用定义好的函数进行计算回填

# 取数逻辑
----
## OpenFE
      
### utils

```get_secucode(start, end)```:从```SmartQuant.ReturnDaily```中获取SecuCode List

```get_month_first_trading_day(start, end)```: 从 ```SmartQuant.CalenderDay_TradingDay```中获取每月首日数据列表

```get_trading_day(start, end, margins=0)``` :获取[start , end + margin] 的交易日列表

### download
```download_Fundamental(date, name, mode='sqlserver')```
- 从指定的数据库（SQL Server 或 OceanBase）下载财务基本面数据，并保存为本地 Parquet 文件。
- data.to_parquet(os.path.join("./Fundamental/raw", name, "{}.parquet".format(date)), index=False)

Q:load_dotenv()是什么？？？

### process
```process_Fundamental(date, name, secucode)```：

Q： ```feature_name = name[12:]```是什么？？？   是不是没有对['CumLatest', 'Quarterly', 'TTM']的不同数据进行处理？

- 从 ./Fundamental/raw/{指标名称}/{日期}.parquet 读取原始财务数据。

            数据包含列：CumLatest（累计值）、Quarterly（季度值）、TTM（滚动年度值）、DataDate（数据日期）、InnerCode（内部代码）等。

- 映射证券代码：读取 InnerCode 到 SecuCode 的映射关系，将财务数据的 InnerCode 转换为标准的 SecuCode（证券代码）

            删除无法映射到有效证券代码的行。

- 按财务期间处理数据（循环12次）

            subset = data.loc[data['EndDateRank']==i+1]    
            创建一个与 secucode 长度相同的空 DataFrame（用 NaN 填充）    
            如果证券代码在当前月份有数据，则填充；否则保持 NaN
  
- 拼接数据：排序后最终存储为结构化格式，传递给```save_Fundamental(new_data, date, feature_name)```

### save
```save_Fundamental(data, date, feature_name)```
- 对每列财务指标（累计值、季度值、滚动年度值）分别处理
-  提取关键列并重命名
  
            df = data[['DataDate', 'SecuCode', 'EndDateRank', col]]
   
            df = df.rename({col: '{}_{}'.format(feature_name, col)}, axis=1) 需要一点example
   
            将处理后的 DataFrame 保存为 Parquet 文件，路径格式为：
   
            df.to_parquet("./Fundamental/Lib/{}_{}/{}.parquet".format(feature_name, col, date), index=False)
```text
./Fundamental/Lib/
    ├── ROE_CumLatest/
    │   └── 20230812.parquet
    ├── ROE_Quarterly/
    │   └── 20230812.parquet
    └── ROE_TTM/
        └── 20230812.parquet
```

### read
```read_Fundamental(date, name)```:
         
            read_parquet("./Fundamental/Lib/{}/{}.parquet".format(name, date))

            data[name].values.reshape(-1, 12): 时间序列数据按固定窗口切分 (N,) -> (N//12, 12,)
            
            每行代表一个证券、每列代表一个月份
            
### DataProvider:

过抽象基类 Data_Provider 和具体子类 (ReturnDaily_Provider 和 Fundamental_Provider) 来统一管理不同数据源的下载处理获取

- 1. 抽象基类 Data_Provider
定义数据提供类的统一接口（download, get_data），强制子类必须实现这些方法。
直接实例化会报错 (NotImplementedError)，确保只能通过子类使用。

- 2. 子类 ReturnDaily_Provider（日频收益率数据）
根据交易日列表，从 OceanBase 逐日数据库下载 ReturnDaily 表（日频收益率数据），并保存为 Parquet 文件。

- 3. 子类 Fundamental_Provider（财务基本面数据）
        - "./Fundamental/raw" 存储原始数据；"./Fundamental/Lib" 存储处理后的数据
        - 为每个财务表创建目录 ：```./Fundamental/raw/{}".format(name)``` 并行下载数据
        - ```process```: 并行处理数据 process_Fundamental 负责数据清洗、对齐证券代码等操作。
        - ```get_data```:从磁盘加载数据

           data = p.starmap(read_Fundamental, tasks)  # 并行读取

           输入：tasks 是一个任务列表，每个任务为 (date, feature_name); 输出：data 是一个列表，包含每个特征对应的 DataFrame，如data =   [df_roe, df_revenue] ， 每个df形状为 (证券数量, 时间维度=1)    

          data_list.append(np.asarray(data).transpose(1, 2, 0))  # 调整维度
          np.asarray(data)：将列表 data 转为 NumPy 数组，默认形状为 (特征数, 证券数, 时间维度)
         .transpose(1, 2, 0)：调整轴顺序，新形状为 (证券数, 时间维度, 特征数)

-----

## GP_OPS

- 从```smartquant.returndaily```中获取交易日股票池```universe```
- 为每个交易日日期生成分钟数据 Parquet 文件路径 保存在一个array中
- 进行多进程因子计算
  - 将2023年的所有交易日均分给30个进程并行处理（将array分成30个进程 并构造参数列表）
  -  ```cal_miniute_data```: 输入一组日期对应的分钟数据文件路径，for循环遍历日期文件，得到每个股票在每日的7个因子值，将这些日期的数据拼接成大表。
- 并行执行并合并所有进程结果
- 结果合并与存储: 将结果表与```universe```拼接，以```universe```为基准，保存```left```存为parquet文件


## 通用框架 OG_OPS总结

```
og_ops/
├── Data_Provider/           # 数据提供模块
│   ├── Data_Provider.py    # 主要数据提供类
│   ├── download.py         # 数据下载
│   ├── process.py          # 数据处理
│   ├── read.py            # 数据读取
│   └── save.py            # 数据保存
├── operators_3D.py         # 3D数据算子库
├── operators_4D.py         # 4D数据算子库
├── utils.py               # 工具函数
└── .env                   # 环境变量配置

Demo.ipynb            # 使用示例
```

### 3D分钟数据处理流程

```
原始分钟数据 → 下载 → （自定义）特征计算 → 处理 → 存储 → 读取 → Calculator
```

##### Step 1: 数据下载 (download_Minute)
- 输入: 日期, 模式('five_minute')
- 数据源: `/data/cephfs/minute/five_minute/{date}.parquet`
- 功能: 从原始分钟数据计算7个日内特征
- 输出: `./Minute/raw/{date}.parquet`

```特征计算 (cal_miniute_data)```:
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

##### Step 2: 数据处理 (process_Minute)
- 输入: `./Minute/raw/{date}.parquet`
- 功能: 
  - 获取交易日股票池
  - 填充完整股票池 (对齐secucode)
  - 按SecuCode排序
  - 分特征保存
- 输出: `./Minute/Lib/{feature_name}/{date}.parquet`

形状变化:
```
原始数据: (n股票, M 特征) → 处理后: (N股票, M 特征) # N为完整股票池大小 → 分特征存储: 每个特征单独文件 (N,)
```

##### Step 3: 数据读取 (read_Minute)
- 输入: 日期, 特征名
- 功能: 读取单个特征的单日数据
- 输出: numpy数组 (N,) 

##### Step 4: 批量获取 (Minute_Provider.get_data)
- 输入: 日期范围, 特征名列表
- 功能: 
  - 并行读取多日多特征数据
  - 堆叠成3D数组
  - 创建特征映射
- 输出: 
  - `data_3d`: (D, N, M) # D=天数, N=股票数, M=特征数
  - `feature_map`: {x1: 'early_ret', x2: 'tail_ret', ...}

#### 最终输入Calculator的形状
```python
# 3D分钟数据
data_3d.shape = (D, N, M)  # (天数, 股票数, 特征数)
# 例如: (22, 5106, 7) = (22天, 5106只股票, 7个特征)

# Calculator中每个特征的形状
feature_data.shape = (D, N)  # (天数, 股票数)
# 例如: (22, 5106) = (22天, 5106只股票)
```

### 4D基本面数据处理流程

```
基本面数据库 → 下载 → 处理重构 → 存储 → 读取 → Calculator
```

#### Step 1: 数据下载 (download_Fundamental)
- 输入: 日期, 表名(如'Fundamental_Item1353')
- 数据源: OceanBase数据库 `Fundamental.{table_name}`
- 功能: 下载指定日期的基本面数据
- 输出: `./Fundamental/raw/{table_name}/{date}.parquet`

#### Step 2: 数据处理 (process_Fundamental 与 save_Fundamental)
- 输入: `./Fundamental/raw/{table_name}/{date}.parquet`
- 核心处理逻辑:
```python
  # 处理12个财务区间 (EndDateRank 1-12) 填齐缺失数据
  for i in range(12):
      subset = data.loc[data['EndDateRank']==i+1]
      # 对齐股票池
      subset_ = pd.DataFrame(np.full((len(secucode), data.shape[1]), np.nan))
      cond = secucode.isin(subset['SecuCode'].values)
      subset_.loc[cond] = subset.values
      new_data.append(subset_)

  # 按证券代码和EndDateRank排序
 new_data = new_data.sort_values(["SecuCode", "EndDateRank"], ascending=[True, False])
```
- 输出: `./Fundamental/Lib/{feature_name}/{date}.parquet # feature_name 如 CumLatest_PE`

形状变化:
```
原始数据: (变长, 多列) # 不同EndDateRank的记录数不同 → 重构后: (N*12, 多列) # N股票 × 12财务区间 → 存储: 每个特征不同单独文件
```

#### Step 3: 数据读取 (read_Fundamental)
- 输入: 日期, 特征名
- 功能: 读取该特征每日数据并重塑为(N, 12)形状
- 输出: numpy数组 (N, 12) # N股票 × 12财务区间

#### Step 4: 批量获取 (Fundamental_Provider.get_data)
- 输入: 日期范围, 特征名列表
- 功能:
  - 并行读取多月多特征数据
  - 转置和堆叠: `单日 (M特征, N股票, 12) → 单日(N, 12, M) → 按日期堆叠 (D, N, 12, M)`
- 输出 : `data_4d`: (D, N, T, M) # D=日期数, N=股票数, T=财务区间数, M=特征数

#### 3.3 最终输入Calculator的形状
```python
# 4D基本面数据
data_4d.shape = (D, N, T, M)  # (月数, 股票数, 财务区间数, 特征数)
# 例如: (12, 100, 12, 5) = (12个月, 100只股票, 12个财务区间, 5个特征)

# Calculator中每个特征的形状
feature_data.shape = (D, N, T)  # (日期, 股票数, 财务区间数)
# 例如: (12, 100, 12) = (12个月, 100只股票, 12个财务区间)
```


#### Calculator兼容性调用

##### 统一接口设计
Calculator类通过以下机制实现对3D和4D数据的兼容:

- 1. 自动维度检测
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

-- 特征提取适配
```python
def get_feature(self, identifier):
    idx = self.get_feature_index(identifier)
    if self.ndim == 3:
        return self.data[:, :, idx]      # 返回 (D, N)
    else:  # 4D
        return self.data[:, :, :, idx]   # 返回 (D, N, T)
```

-- 结果堆叠适配
```python
def _stack_results(self, results):
    if self.ndim == 3:
        return np.stack(results, axis=2)  # (D, N, len(results))
    else:  # 4D
        return np.stack(results, axis=3)  # (D, N, T, len(results))
```

- 特征访问      
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

- 3. 算子调用统一化
```python
# 单参数算子
result = calc.apply_unary_operator('abs_val', 'feature_name')

# 双参数算子  
result = calc.apply_binary_operator('add', 'feature1', 'feature2')

# 批量应用
result = calc.apply_operator('rank')  # 应用到所有特征
```

#### Calculator使用指南

1. 初始化

```python
# 3D数据
calc_3d = Calculator(data_3d, feature_map=feature_map)

# 4D数据  
calc_4d = Calculator(data_4d, feature_names=['PE', 'ROE', 'ROA', 'PB', 'EPS'])
```

2. 基本操作
```python
# 查看信息
calc.info()
calc.list_operators()

# 特征访问
feature_data = calc.get_feature('feature_name')
multi_features = calc.get_features(['feat1', 'feat2'])
```

3. 算子调用逻辑     

Calculator通过自动检测算子函数的参数数量来决定调用方式    

**调用分类说明：**
- 单参数算子: 只需要一个输入特征的算子（如数学函数、时序算子、截面算子）
- 双参数算子: 需要两个输入特征的算子（如算术运算、相关性计算）
- 多参数算子: 需要三个或更多输入特征的算子（如复杂组合算子）

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
        # 多参数算子：尝试直接传递所有特征
        result = operator_func(*feature_data_list, **kwargs)
```

3D算子示例
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

4D算子示例
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

#### numba   Parallel=True vs Parallel=False 性能对比 

- 测试数据形状: (504, 5000, 30)
- 测试参数: window=20, num=5

1. 单进程
```
ts_max_mean 对比 :
  Parallel=True:  0.2320s          Parallel=False: 3.8243s     

ts_corr 对比:
  Parallel=True:  0.0642s          Parallel=False: 1.5068s
```

2. 多进程（4进程）
```
两任务总时长对比 :
  Parallel=True:  3.8593s         Parallel=False: 3.8507s  
```


#### numba 并行嵌套速度测试

- 测试算子: ts_max_mean, ts_corr, ts_to_wm, ts_rankcorr
- 数据规模: 小数据集（100，1000，5）、中数据集（252，3000，7）、大数据集（504，5000，30）
- 并行配置: 单进程、4进程、8进程、12进程； numba parallel=True   
- 测试任务: 每种算子×数据规模组合，共12个任务


```python
# 多进程嵌套测试架构
ProcessPoolExecutor(max_workers=N) 
├──  子进程1: numba算子(parallel=True)
├──  子进程2: numba算子(parallel=True)  
├──  ...
└──  子进程N: numba算子(parallel=True)
```

```
算子性能Benchmark (单进程):
├── ts_max_mean:  小(0.0051s) → 中(0.0317s) → 大(0.0848s)
├── ts_corr:      小(0.0291s) → 中(0.2501s) → 大(0.8693s)  
├── ts_to_wm:     小(0.0019s) → 中(0.0093s) → 大(0.0239s)
└── ts_rankcorr:  小(0.0099s) → 中(0.0745s) → 大(0.2514s)

**4进程环境**:
├── ts_max_mean:  小(0.0178s) → 中(0.1788s) → 大(0.3301s)
├── ts_corr:      小(0.0428s) → 中(0.2965s) → 大(0.8719s)  
├── ts_to_wm:     小(0.0112s) → 中(0.0128s) → 大(0.0540s)
└── ts_rankcorr:  小(0.0382s) → 中(0.1302s) → 大(0.4499s)

**8进程环境**:
├── ts_max_mean:  小(0.0176s) → 中(0.1297s) → 大(0.5081s)
├── ts_corr:      小(0.0299s) → 中(0.2526s) → 大(0.8723s)  
├── ts_to_wm:     小(0.0142s) → 中(0.0211s) → 大(0.0821s)
└── ts_rankcorr:  小(0.0262s) → 中(0.2720s) → 大(1.3685s)

**12进程环境**:
├── ts_max_mean:  小(0.0166s) → 中(0.1183s) → 大(0.3162s)
├── ts_corr:      小(0.0531s) → 中(0.2526s) → 大(0.8792s)  
├── ts_to_wm:     小(0.0198s) → 中(0.0245s) → 大(0.0513s)
└── ts_rankcorr:  小(0.0299s) → 中(0.3753s) → 大(0.4745s)
```

结论： `numba parallel=True` 在多进程环境下**没有退化为单线程**，每个进程的numba都尝试使用多线程，导致了线程过度订阅

8进程 × 每进程4-8线程 = 32-64线程 >> 16CPU核心   ->  严重的内存线程竞争和上下文切换开销     

需要主动控制线程数

```
ts_max_mean 大数据集   # 循环遍历 计算滚动窗口内最大的num个值的平均值    
单进程: 0.0848s
4进程环境: 0.3301s
8进程环境: 0.5081s 
12进程环境: 0.3162s 

ts_rankcorr大数据集
单进程: 0.2514s
4进程环境: 0.4499s
8进程环境: 1.3685s 
12进程环境: 0.4745s 

ts_corr大数据集 
单进程: 0.8693s
4进程环境: 0.8719s
8进程环境: 0.8723s 
12进程环境: 0.8792s 
```
