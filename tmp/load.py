import glob

import torch

folder_path = './'

# 使用glob模块查找所有匹配'tmp*.pth'的文件
files = glob.glob(f'{folder_path}/tmp*')
pdata_list = []
# 打印找到的文件列表
for file in files:
    t = torch.load(file)
    pdata_list.append(t['pdata'])
res = torch.stack(pdata_list)
res = res.view((res.shape[0] * res.shape[1], -1))
torch.save(res, 'pdata.pth')