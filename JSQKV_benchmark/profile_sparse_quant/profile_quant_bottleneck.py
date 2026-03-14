#!/usr/bin/env python3
"""
快速诊断量化性能瓶颈
"""
import torch
import os
import sys
import time
from transformers import AutoTokenizer, LlamaConfig

# 添加项目根目录到路径（向上两级到 mustafar/）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = '/home/zh/model/Meta-Llama-3-8B-Instruct'
BATCH_SIZE = 8
INPUT_LENGTH = 4096
OUTPUT_LENGTH = 100  # 短一点，快速测试

print("="*70)
print("🔍 量化性能瓶颈诊断")
print("="*70)

# 加载模型
print("\n📦 Loading quantized model...")
config = LlamaConfig.from_pretrained(MODEL_PATH)
config.k_sparsity = 0.5
config.v_sparsity = 0.5
config.group_size = 32
config.residual_length = 256
config.use_flash = True
config.quant_bits = 2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llama_mustafar_quant_kernel import LlamaForCausalLM_MUSTAFAR_QUANT
model = LlamaForCausalLM_MUSTAFAR_QUANT.from_pretrained(
    MODEL_PATH,
    config=config,
    torch_dtype=torch.float16,
    device_map='auto'
)
model.eval()

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 准备输入
context = []
for _ in range(BATCH_SIZE):
    string = 'apple bear' * (INPUT_LENGTH // 2)
    context.append(string[:-1])

inputs = tokenizer(context, return_tensors="pt").to('cuda')

print(f"✅ Model loaded")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Input length: {inputs['input_ids'].shape[1]}")
print(f"   Output length: {OUTPUT_LENGTH}")

# Warmup
print("\n⏳ Warmup...")
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=10, eos_token_id=None)
torch.cuda.synchronize()

# ==================== Profile with PyTorch Profiler ====================
print("\n" + "="*70)
print("🔥 Running PyTorch Profiler...")
print("="*70)

from torch.profiler import profile, ProfilerActivity, record_function

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=OUTPUT_LENGTH, eos_token_id=None)

torch.cuda.synchronize()

# 打印 Top 20 最耗时的操作
print("\n📊 Top 20 CUDA operations by time:")
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20,
    max_name_column_width=60
))

# 保存详细 trace
trace_file = os.path.join(current_dir, "quant_profile_trace.json")
os.makedirs(os.path.dirname(trace_file), exist_ok=True)
try:
    prof.export_chrome_trace(trace_file)
    print(f"\n💾 Detailed trace saved to: {trace_file}")
    print(f"   Open in Chrome: chrome://tracing")
except Exception as e:
    print(f"\n⚠️  Failed to save trace: {e}")

# ==================== 统计 kernel 调用次数 ====================
print("\n" + "="*70)
print("📈 Kernel Call Statistics")
print("="*70)

# 从 profiler 中提取关键信息
events = prof.key_averages()

# 查找量化相关的操作
quant_ops = []
dequant_ops = []
matmul_ops = []
cat_ops = []

for evt in events:
    name = evt.key
    # 使用 self_cuda_time_total 而不是 cuda_time_total
    cuda_time = evt.self_cuda_time_total / 1000 if hasattr(evt, 'self_cuda_time_total') else 0  # ms
    
    if 'quant' in name.lower():
        quant_ops.append((name, evt.count, cuda_time))
    if 'matmul' in name.lower() or 'mm' in name.lower() or 'gemm' in name.lower():
        matmul_ops.append((name, evt.count, cuda_time))
    if 'cat' in name.lower() or 'concat' in name.lower():
        cat_ops.append((name, evt.count, cuda_time))

print("\n🔧 Quantization Operations:")
if quant_ops:
    for name, count, time_ms in sorted(quant_ops, key=lambda x: x[2], reverse=True)[:5]:
        print(f"   {name[:50]:50s} | Calls: {count:4d} | Time: {time_ms:8.2f} ms")
else:
    print("   (No explicit quant ops found - may be in custom kernels)")

print("\n🔧 Matrix Multiplication Operations:")
if matmul_ops:
    for name, count, time_ms in sorted(matmul_ops, key=lambda x: x[2], reverse=True)[:5]:
        print(f"   {name[:50]:50s} | Calls: {count:4d} | Time: {time_ms:8.2f} ms")

print("\n🔧 Concatenation Operations:")
if cat_ops:
    for name, count, time_ms in sorted(cat_ops, key=lambda x: x[2], reverse=True)[:5]:
        print(f"   {name[:50]:50s} | Calls: {count:4d} | Time: {time_ms:8.2f} ms")

# ==================== 简单性能对比 ====================
print("\n" + "="*70)
print("⚡ Quick Performance Test")
print("="*70)

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=OUTPUT_LENGTH, eos_token_id=None)

torch.cuda.synchronize()
elapsed = time.time() - start

total_tokens = BATCH_SIZE * OUTPUT_LENGTH
throughput = total_tokens / elapsed
tpot = elapsed * 1000 / OUTPUT_LENGTH

print(f"\n📊 Results:")
print(f"   Total time: {elapsed*1000:.2f} ms")
print(f"   Throughput: {throughput:.2f} tokens/sec")
print(f"   TPOT: {tpot:.2f} ms")
print(f"   Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print("\n" + "="*70)
print("✅ Profiling completed!")
print("="*70)
print("\n💡 Next steps:")
print("   1. Check the top operations in the table above")
print("   2. Open the trace file in Chrome to visualize timeline")
print("   3. Look for:")
print("      - Custom CUDA kernels (SparseMatMul_Quant)")
print("      - Frequent small operations (quantization)")
print("      - Memory operations (cat, copy)")
