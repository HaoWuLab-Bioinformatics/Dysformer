import pandas as pd
from scipy import stats

# 读取Excel文件
file_path = '工作簿1.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(file_path, header=None)  # 假设Excel文件没有列名，读取时指定header=None

# 提取你需要的行（假设你要的是第1到第3行，列0到29）
row1 = df.iloc[24, 0:30]  # 第一行（索引0）
row2 = df.iloc[25, 0:30]  # 第二行（索引1）
row3 = df.iloc[26, 0:30]  # 第三行（索引2）

# 进行t检验，计算p-value
t_stat_1_3, p_value_1_3 = stats.ttest_ind(row1, row3)  # 第一行和第三行的t检验
t_stat_2_3, p_value_2_3 = stats.ttest_ind(row2, row3)  # 第二行和第三行的t检验

# 输出结果
print(f"第一行和第三行的p-value: {p_value_1_3}")
print(f"第二行和第三行的p-value: {p_value_2_3}")

# 判断是否显著（通常显著性水平为0.05）
if p_value_1_3 < 0.05:
    print("第一行和第三行之间存在显著差异")
else:
    print("第一行和第三行之间不存在显著差异")

if p_value_2_3 < 0.05:
    print("第二行和第三行之间存在显著差异")
else:
    print("第二行和第三行之间不存在显著差异")
