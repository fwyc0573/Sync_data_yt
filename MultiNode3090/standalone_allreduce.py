# 使用PyTorch的性能分析器 (torch.profiler) 来测量和记录在分布式环境中执行all_reduce操作的性能。
import os
import time
import torch
import argparse
import deepspeed.comm as ds_dist
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import schedule
from torch.profiler import profile, record_function, ProfilerActivity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standalone Allreduce')
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank. ")
    parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=1120)
    parser.add_argument("--profile", type=int, default=0, help="Whether to enable torch profiler.")
    # --p 和 --q 参数定义了all_reduce操作中传输的一个形状为 (P, Q) 的张量，其中 P 和 Q 分别是两个维度的大小。
    parser.add_argument(
        '--p',
        type=int,
        default=50000,
        required=False,
        help='The P dim of Tensor(P, Q) to transfer in NCCL AllReduce.',
    )
    parser.add_argument(
        '--q',
        type=int,
        default=14000,
        required=False,
        help='The Q dim of Tensor(P, Q) to transfer in NCCL AllReduce.',
    )
    # 执行指定次数（--num_steps）的all_reduce操作
    parser.add_argument(
        '--num_steps',
        type=int,
        default=10,
        required=False,
        help='The iteration number.',
    )

    args = parser.parse_args()
    P = args.p
    Q = args.q

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.manual_seed(args.random_seed)
    torch.cuda.set_device(local_rank)
    ds_dist.init_distributed(dist_backend=args.backend)
    # dist.init_process_group(backend=args.backend)

    # 创建一个形状为 [P, Q] 的随机张量 message，并将其放在 GPU 上。
    shape = [P, Q]
    message = torch.randn(*shape).cuda()
    
    profile_schedule = torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=2,
        repeat=1
    )
    worker_name="allreduce_P{}_Q{}".format(P,Q)
    # 开始一个性能分析器上下文管理器，指定跟踪的输出目录和其他配置。
    with torch.profiler.profile(
            schedule=profile_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("/mnt", worker_name),
            record_shapes= True,
            with_flops =True,
            with_stack=True
        ) as profiler:

        start = time.perf_counter()
        for _ in range(args.num_steps):
            # 执行一个分布式的 all_reduce 操作。这个操作对分布式系统中所有节点的 message 张量进行求和，
            # 并将结果放回到每个节点的原始 message 张量中。这是一个异步操作。
            dist.all_reduce(message, op=dist.ReduceOp.SUM, async_op=True)
            profiler.step()
        # 同步所有CUDA操作。在测量耗时时，这确保了所有前面的操作都已完成。
        torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed = end - start

    result = round(elapsed*1000/args.num_steps, 2)
    filename = f"allreduce_rank{dist.get_rank()}_{result}ms.txt"
    with open(filename, 'w') as file:
        file.write(str(result))

    # 获取当前进程的排名，print每个GPU上 all_reduce 操作的平均耗时（以毫秒为单位）。
    print("gpu {} avg. elapsed time: {} ms".format(dist.get_rank(), result))
    print(f"File {filename} has been created with the result: {result}")


