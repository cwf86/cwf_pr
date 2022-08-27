import torch

points = torch.tensor([[4.,1.],[5.,3.],[2.,1.]])

#索引
#print(points)
#print(points[0,1])
#print(points[0])

#以下的分片操作是pytorch张量实现的
print(points[1:])
print(points[1:, :])
print(points[1:, 0])
print(points[None])#增加一个维度

#指定张量类型
double_points = torch.ones(10,2,dtype=torch.double)#torch.double等同于torch.float64
print(double_points.dtype)

short_points = double_points.short()#torch.short等同于torch.int16
print(short_points.dtype)

#保存张量(序列化)
#torch.save(double_points, r'D:\\SVN_WORK\\cwf_pytorch\\cwf_points.t')




