# TileLang 常见错误及解决方案

## 1. ValueError: Expected N inputs, got M with X inputs and Y outputs

### 问题描述
```python
ValueError: Expected 3 inputs, got 4 with 3 inputs and 1 outputs
```

### 原因
这个错误是由于 `@tilelang.jit(out_idx=...)` 参数的使用不当导致的。

### 两种使用模式

#### 模式 1: 自动创建输出 (推荐)
```python
@tilelang.jit(out_idx=[-1])  # -1 表示最后一个参数是输出
def matmul(...):
    @T.prim_func
    def gemm(A: T.Tensor, B: T.Tensor, C: T.Tensor):
        ...
    return gemm

# 使用时，输出会自动创建并返回
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = kernel(a, b)  # ✓ 正确：只传入 2 个输入
```

#### 模式 2: 手动传入输出
```python
@tilelang.jit()  # 不指定 out_idx
# 或
@tilelang.jit(out_idx=None)
def matmul(...):
    @T.prim_func
    def gemm(A: T.Tensor, B: T.Tensor, C: T.Tensor):
        ...
    return gemm

# 使用时，需要手动创建并传入输出张量
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = torch.empty(M, N, device="cuda", dtype=torch.float16)
kernel(a, b, c)  # ✓ 正确：传入 3 个参数
```

### out_idx 参数说明

| out_idx 值 | 含义 | 调用方式 |
|-----------|------|---------|
| `None` (默认) | 所有参数都需要手动传入 | `kernel(a, b, c)` |
| `[-1]` | 最后一个参数是输出，自动创建 | `c = kernel(a, b)` |
| `[-1, -2]` | 最后两个参数是输出 | `c, d = kernel(a, b)` |
| `[2]` | 第 3 个参数 (索引2) 是输出 | `c = kernel(a, b)` |

---

## 2. 找不到 .so 文件

### 问题描述
```python
ImportError: cannot import name '_ffi_api' from 'tilelang'
OSError: libtilelang.so: cannot open shared object file
```

### 解决方案
```bash
# 方法 1: 设置环境变量
export PYTHONPATH=/root/wkspace/tilelang:$PYTHONPATH
export LD_LIBRARY_PATH=/root/wkspace/tilelang/build:$LD_LIBRARY_PATH

# 方法 2: 使用 editable install
pip install -e /root/wkspace/tilelang

# 方法 3: 检查构建是否成功
ls -la /root/wkspace/tilelang/build/libtilelang*.so
```

---

## 3. CUDA out of memory

### 问题描述
```python
RuntimeError: CUDA out of memory
```

### 解决方案
```python
# 1. 减小 batch size 或矩阵维度
M, N, K = 512, 512, 512  # 而不是 1024

# 2. 减小 block size
block_M, block_N = 64, 64  # 而不是 128

# 3. 减少 pipeline stages
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):  # 而不是 3

# 4. 清理 GPU 缓存
import torch
torch.cuda.empty_cache()

# 5. 检查内存使用
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

---

## 4. 编译错误：nvcc not found

### 问题描述
```
CMake Error: nvcc not found
```

### 解决方案
```bash
# 1. 检查 CUDA 是否安装
nvcc --version

# 2. 设置 CUDA 路径
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3. 或者在 CMake 中指定
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# 4. 如果没有 GPU，使用 CPU 后端
echo "set(USE_CUDA OFF)" >> build/config.cmake
echo "set(USE_LLVM ON)" >> build/config.cmake
```

---

## 5. 类型不匹配错误

### 问题描述
```python
RuntimeError: Expected tensor of type float16 but got float32
```

### 解决方案
```python
# 确保输入数据类型与 kernel 定义一致
kernel = matmul(..., dtype="float16")  # kernel 定义

# 输入张量也要匹配
a = torch.randn(M, K, device="cuda", dtype=torch.float16)  # ✓ 正确
# a = torch.randn(M, K, device="cuda")  # ✗ 错误：默认是 float32
```

---

## 6. 维度不匹配错误

### 问题描述
```python
RuntimeError: Shape mismatch
```

### 解决方案
```python
# 方法 1: 静态维度 - 必须完全匹配
kernel = matmul(1024, 1024, 1024, ...)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)  # ✓
# a = torch.randn(512, 512, device="cuda", dtype=torch.float16)  # ✗

# 方法 2: 动态维度 - 使用 symbolic
M = T.symbolic("m")
N = T.symbolic("n")
K = T.symbolic("k")
kernel = matmul(M, N, K, 128, 128, 32)

# 现在可以接受任意大小
a = torch.randn(512, 512, device="cuda", dtype=torch.float16)  # ✓
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)  # ✓
```

---

## 7. 编译缓存问题

### 问题描述
修改代码后，运行结果没有变化（使用了旧的缓存）

### 解决方案
```bash
# 清理编译缓存
rm -rf ~/.tilelang/cache

# 或者禁用缓存
export TILELANG_DISABLE_CACHE=1
python your_script.py
```

---

## 8. 调试时看不到详细错误

### 问题描述
错误信息不够详细，难以定位问题

### 解决方案
```bash
# 启用详细日志
export TVM_LOG_DEBUG=1
export TVM_BACKTRACE=1
export CUDA_LAUNCH_BLOCKING=1  # CUDA 同步模式

python your_script.py
```

在 Python 代码中：
```python
# 编译时启用 verbose
kernel = tilelang.jit(
    target="cuda",
    verbose=True  # 打印所有编译 Pass 信息
)(my_function)
```

---

## 9. 性能不如预期

### 问题描述
生成的 kernel 性能远低于预期

### 排查步骤

#### 1. 检查是否使用了优化
```python
# 启用 swizzle 优化
with T.Kernel(...) as (bx, by):
    T.use_swizzle(panel_size=10, enable=True)
    ...

# 使用 pipeline
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    ...
```

#### 2. 调整 block size
```python
# 尝试不同的 block size
# 对于 A100/H100，常用 128x128 或 256x128
block_M, block_N = 128, 128

# 对于较小的矩阵，可以减小
block_M, block_N = 64, 64
```

#### 3. 使用 profiler 分析
```python
profiler = kernel.get_profiler()
latency = profiler.do_bench(backend="cupti")  # 获取详细信息
```

#### 4. 与 baseline 对比
```python
# 与 PyTorch cuBLAS 对比
ref_c = a @ b
# 对比性能和正确性
```

#### 5. 使用 Nsight Compute 分析
```bash
ncu -o profile --set full python script.py
ncu-ui profile.ncu-rep
# 查看：
# - 内存带宽利用率
# - 计算单元利用率
# - Bank conflicts
# - Occupancy
```

---

## 10. 测试失败

### 问题描述
```python
AssertionError: Tensor mismatch
```

### 解决方案
```python
# 1. 调整容差
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)  # float16
# torch.testing.assert_close(c, ref_c, rtol=1e-5, atol=1e-5)  # float32

# 2. 检查数值精度
print(f"Max diff: {(c - ref_c).abs().max()}")
print(f"Mean diff: {(c - ref_c).abs().mean()}")

# 3. 使用更宽松的比较
assert torch.allclose(c, ref_c, rtol=1e-1, atol=1e-1)

# 4. 可视化差异
import matplotlib.pyplot as plt
plt.imshow((c - ref_c).abs().cpu().numpy())
plt.colorbar()
plt.savefig("diff.png")
```

---

## 快速诊断检查表

遇到问题时，按顺序检查：

- [ ] Python 版本和依赖是否正确: `python --version`, `pip list | grep torch`
- [ ] CUDA 是否可用: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] TileLang 是否正确安装: `python -c "import tilelang; print(tilelang.__version__)"`
- [ ] 环境变量是否设置: `echo $PYTHONPATH`, `echo $LD_LIBRARY_PATH`
- [ ] 构建是否成功: `ls -la build/libtilelang*.so`
- [ ] 输入数据类型是否匹配
- [ ] 输入数据维度是否匹配
- [ ] `out_idx` 参数使用是否正确
- [ ] GPU 内存是否足够: `nvidia-smi`
- [ ] 是否启用了详细日志: `export TVM_LOG_DEBUG=1`

---

## 获取帮助

如果以上方法都无法解决问题：

1. **查看日志**：保存完整的错误日志
   ```bash
   python script.py 2>&1 | tee error.log
   ```

2. **最小复现示例**：创建最简单的能复现问题的代码

3. **提供环境信息**：
   ```bash
   python --version
   nvcc --version
   pip list | grep -E "torch|tilelang"
   nvidia-smi
   ```

4. **提交 Issue**：
   - GitHub: https://github.com/tile-ai/tilelang/issues
   - Discord: https://discord.gg/TUrHyJnKPG

5. **查看示例代码**：
   - `examples/` 目录包含大量经过验证的示例
   - 参考类似的实现

---

**更多详细调试信息请查看 DEBUGGING_GUIDE.md**
