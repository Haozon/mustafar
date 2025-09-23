# model e.g.: meta-llama/Llama-2-7b-hf

#gpuid=$1
# k_sparsity=$1
# v_sparsity=$2
# group_size=32
# model=$3
# mode=$4
e=0

# baseline: k_sparsity=0, v_sparsity=0 (no sparsity)
k_sparsity=0
v_sparsity=0
group_size=32
# model=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/LLMs/LLMs_HF/llama-2-7b
model="/home/zh/model/Meta-Llama-3-8B-Instruct"
mode='mustafar'

CUDA_VISIBLE_DEVICES=0 python ./pred_long_bench.py --model_name_or_path $model \
    --cache_dir ./cached_models \
    --k_sparsity $k_sparsity \
    --v_sparsity $v_sparsity \
    --group_size $group_size \
    --residual_length $group_size \
    --mode $mode \
    --e ${e}

# 注释掉额外的测试，只运行baseline
# k_sparsity=0.7
# v_sparsity=0.7
# CUDA_VISIBLE_DEVICES=0 python ./pred_long_bench.py --model_name_or_path $model \
#     --cache_dir ./cached_models \
#     --k_sparsity $k_sparsity \
#   --v_sparsity $v_sparsity \
#     --group_size $group_size \
#     --residual_length $group_size \
#     --mode $mode \
#     --e ${e} 

# wait  # 等待所有后台任务完成
