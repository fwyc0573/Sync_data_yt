# 3090测试

	## 1.1 "resnet50" --batchsize="32"
deepspeed --hostfile hostfile  profile_test.py --model="resnet50" --batchsize="32"
	
	## 1.2 "resnet50" --batchsize="128"
deepspeed --hostfile hostfile  profile_test.py --model="resnet50" --batchsize="128"
	
	## 2.1 "vgg19" --batchsize="32"
deepspeed --hostfile hostfile  profile_test.py --model="vgg19" --batchsize="32"
	
	## 2.2 "vgg19" --batchsize="128"
deepspeed --hostfile hostfile  profile_test.py --model="vgg19" --batchsize="128"
	
	## 3.1 "densenet121" --batchsize="32"
deepspeed --hostfile hostfile  profile_test.py --model="densenet121" --batchsize="32"
	
	## 3.2 "densenet121" --batchsize="128"
deepspeed --hostfile hostfile  profile_test.py --model="densenet121" --batchsize="128"
	

	# 4. all-reduce
deepspeed --hostfile hostfile standalone_allreduce.py