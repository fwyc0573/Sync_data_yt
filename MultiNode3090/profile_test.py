import json
import torch
import torchvision
from torch.autograd import Variable
import torch.optim as optim
import time
import argparse
import torch.distributed as dist
import deepspeed.comm as ds_dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd.profiler_util import (_format_time, EventList, FunctionEvent, FunctionEventAvg)
import pandas as pd
import os


# densenet121, vgg19, resnet50
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--repeat", default=20, type=int)
parser.add_argument('--model', type=str, default='densenet121',
                    help='model to benchmark')
parser.add_argument('--bucket_cap_mb', type=int, default=25,
                    help='ddp bucket_cap_mb') # DDP中梯度桶的大小
parser.add_argument("--batchsize", default=32, type=int)
FLAGS = parser.parse_args()


# local_rank = FLAGS.local_rank
local_rank = int(os.environ['LOCAL_RANK'])

bucket_cap_mb = FLAGS.bucket_cap_mb
# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
ds_dist.init_distributed(dist_backend='nccl')
# dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端


from torchvision import models
# 加载了指定的模型（例如ResNet50），创建了随机数据作为输入，以及设置了用于训练的优化器（这里使用了随机梯度下降SGD）
model = getattr(models, FLAGS.model)().cuda()
example = torch.rand(FLAGS.batchsize, 3, 224, 224).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01)

module = DDP(model, device_ids=[local_rank], output_device=local_rank)
# 使用 torch.autograd.profiler 捕获和分析模型的性能数据。分析期间收集的信息包括 CUDA 调用的时间和其他详细事件数据。
from torch.autograd.profiler_util import (_format_time, EventList, FunctionEvent, FunctionEventAvg)
import torch.autograd.profiler as torch_profiler


results = {}
for i in range(2):
    for j in range(5):
        y = module(example)
        y.backward(y)
    with torch_profiler.profile(use_cuda=True) as prof:
        y = module(example)
        y.backward(y)

    if i == 0:
        continue

    #print(prof.table(top_level_events_only=True))#, sort_by="self_cuda_time_total"))
    event_list = prof.function_events
    count = 0
    self_cuda_time_total = 0
    for e in event_list:
        if e.self_cuda_time_total != 0:
            # key = e.name + str(count)
            # if key not in results:
            #     results[key] = []
            # results[key].append(e.self_cuda_time_total)
            # results[key].append(str(e))
            
            # 对同一operator计算总累计时间
            key = e.name
            if key not in results:
                results[key] = e.self_cuda_time_total
            else:
                results[key] += e.self_cuda_time_total
            self_cuda_time_total += e.self_cuda_time_total
            # print(e.name, e.self_cuda_time_total)
            count += 1

    if 'average_step_time' not in results:
        results['whole_step_time'] = str(self_cuda_time_total / 1000) + " ms"

    # results['average_step_time'].append(self_cuda_time_total / 1000)
    # results['average_step_time'].append("")
    # self_cuda_time_total = (sum([e.self_cuda_time_total for e in event_list])) / 1000
    print(local_rank, self_cuda_time_total / 1000)
    print(f"results = {results}")


data_for_df = [{"operator": op, "time_us": time} for op, time in results.items()]
df = pd.DataFrame(data_for_df)

filename = f"{FLAGS.model}_b{FLAGS.batchsize}_log{local_rank}.csv"
df.to_csv(filename, index=False)
print(f"表格{filename}完成存储...")

# df = pd.DataFrame(results)
# df.to_csv(str(FLAGS.model) + '_b' + str(FLAGS.batchsize) + '_log' + str(local_rank) + '.csv')

# json.dump(results, open(str(FLAGS.batchsize) +'_event' + str(local_rank) + '.json', 'w'), indent=4)


# 单节点启动命令
# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="172.17.0.3" --master_port=1234 profile_test.py

