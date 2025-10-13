# TileLang 联调指南

## 项目架构概览

TileLang 是由 **Python 前端 + C++ 后端** 构成的编译器框架，其架构如下：

```
Python 前端 (tilelang/)
    ↓ (通过 FFI)
TVM IR (中间表示)
    ↓ (Codegen)
C++ 后端 (src/ + 3rdparty/)
    ↓
CUDA/CUTLASS/HIP 代码生成
```

### 核心组件

1. **Python 前端** (`tilelang/`):
   - `language/`: DSL 语法定义
   - `jit/`: JIT 编译器入口
   - `primitives/`: 算子原语 (gemm, copy, etc.)
   - `_ffi_api.py`: FFI 调用接口

2. **C++ 后端** (`src/`):
   - `ir.cc`: IR 节点定义
   - `transform/`: IR 变换 Pass
   - `target/`: 目标代码生成
   - `tl_templates/`: CUTLASS/CuTe 模板

3. **第三方依赖** (`3rdparty/`):
   - `tvm/`: 编译器基础设施
   - `cutlass/`: NVIDIA CUTLASS 库
   - `composable_kernel/`: AMD CK 库

---

## 构建与安装

### 1. 从源码构建 (推荐用于开发)

```bash
# 1. 克隆仓库 (包含子模块)
git clone --recursive https://github.com/tile-ai/tilelang
cd tilelang

# 2. 配置构建选项
mkdir -p build
cd build

# 复制配置文件
cp ../3rdparty/tvm/cmake/config.cmake .

# 编辑 config.cmake 启用所需后端
cat >> config.cmake << EOF
set(USE_CUDA ON)       # NVIDIA GPU
# set(USE_ROCM ON)     # AMD GPU
set(USE_LLVM ON)       # CPU backend
set(CMAKE_BUILD_TYPE RelWithDebInfo)  # 包含调试信息的优化构建
EOF

# 3. 构建
cmake ..
make -j$(nproc)

# 4. 安装 Python 包 (开发模式)
cd ..
pip install -e . -v

# 5. 检查 torch 与 cuda 版本是否匹配
python -c "import torch; print(torch.version.cuda)"                                       
# 12.8

# 重装至 12.6
pip uninstall torch -y && pip install torch --extra-index-url https://download.pytorch.org/whl/cu126
```

**关键参数说明：**
- `RelWithDebInfo`: 既优化又保留调试符号，最适合联调
- `-e`: editable mode，代码修改后无需重新安装
- `-v`: 显示详细构建日志

**clangd 配置**

```bash
ln -fs build/compile_commands.json compile_commands.json
```

### 2. 验证安装

```bash
python -c "import tilelang; print(tilelang.__version__)"
python examples/gemm/example_gemm.py
```

---

## 调试配置

### 方法 1: VSCode Python 调试

项目已包含 `.vscode/launch.json`，可直接使用：

```json
{
    "name": "Python: Debug Current .py",
    "type": "debugpy",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal",
    "justMyCode": false,  // 改为 false 可进入库代码
    "env": {
        "PYTHONPATH": "${workspaceFolder}"
    }
}
```

**使用步骤：**
1. 打开任意 Python 文件 (如 `examples/gemm/example_gemm.py`)
2. 按 `F5` 或点击 "Run and Debug"
3. 在感兴趣的地方设置断点

**常见断点位置：**
- `tilelang/jit/__init__.py:compile()` - JIT 编译入口
- `tilelang/primitives/gemm.py` - GEMM 算子实现
- `tilelang/transform/*.py` - IR 变换 Pass

### 方法 2: C++ 后端调试 (gdb)

当需要调试 C++ 代码时：

```bash
# 1. 使用 gdb 启动 Python
gdb --args python examples/gemm/example_gemm.py

# 2. 在 gdb 中设置断点
(gdb) b tilelang::ir::GemmNode::make  # C++ 函数断点
(gdb) run

# 3. 查看调用栈
(gdb) bt
(gdb) frame 3  # 切换到第 3 帧
```

### 方法 3: 混合调试 (Python + C++)

使用 VSCode C/C++ 扩展的 attach 功能：

**launch.json 配置：**
```json
{
    "name": "(gdb) Attach to Python",
    "type": "cppdbg",
    "request": "attach",
    "program": "/path/to/python",
    "processId": "${command:pickProcess}",
    "MIMode": "gdb"
}
```

**调试流程：**
1. 先启动 Python 调试
2. 在 Python 代码中加入等待：`input("Press Enter to continue...")`
3. 使用 C++ attach 配置附加到进程
4. 在 C++ 代码中设置断点后按 Enter 继续

---

## 常用调试技巧

### 1. 打印生成的 CUDA 代码

```python
# 在你的 Python 脚本中
kernel = matmul(1024, 1024, 1024, 128, 128, 32)

# 查看生成的 CUDA 源码
print(kernel.get_kernel_source())

# 保存到文件
with open("generated_kernel.cu", "w") as f:
    f.write(kernel.get_kernel_source())
```

### 2. 查看 TVM IR

```python
import tilelang
from tilelang import tvm

# 获取 IR 模块
@tilelang.jit(target="cuda")
def my_kernel(...):
    @T.prim_func
    def func(...):
        ...
    return func

# 打印 IR
print(my_kernel.mod)  # 查看 TVM IRModule
print(my_kernel.mod.script())  # 以 TVMScript 格式打印
```

### 3. 使用 TileLang 内置调试工具

```python
import tilelang.language as T

@T.prim_func
def kernel(...):
    # 打印变量值 (会生成 printf 到 CUDA kernel 中)
    T.print("Debug: x =", x)
    
    # 打印 buffer 内容
    T.print("A_shared:", A_shared)
```

### 4. 环境变量调试开关

```bash
# 启用 TVM 调试日志
export TVM_LOG_DEBUG=1

# 保留中间文件 (如 PTX, CUBIN)
export TVM_DUMP_IR=1
export TVM_KEEP_SOURCE=1

# CUDA 同步以捕获错误
export CUDA_LAUNCH_BLOCKING=1
```

### 5. 单元测试调试

```bash
# 运行单个测试
pytest testing/test_gemm.py -v -s

# 使用 pdb 调试
pytest testing/test_gemm.py --pdb

# 使用 VSCode 调试测试
# 在测试文件中设置断点，然后使用 "Python: Debug Tests" 配置
```

---

## 代码修改后的工作流

### Python 代码修改

```bash
# 如果是 editable install，直接运行即可
python examples/gemm/example_gemm.py

# 如果不是 editable install
pip install -e . --no-build-isolation
```

### C++ 代码修改

```bash
cd build
make -j$(nproc)

# Python 包会自动使用新编译的 .so 文件
python examples/gemm/example_gemm.py
```

### 修改 CUTLASS/CuTe 模板

模板位于 `src/tl_templates/`，修改后需要重新编译：

```bash
cd build
make -j$(nproc)
```

---

## 性能分析

### 1. 使用内置 Profiler

```python
profiler = kernel.get_profiler()

# 基础性能测试
latency = profiler.do_bench()
print(f"Latency: {latency} ms")

# 使用 CUPTI 获取详细信息
latency = profiler.do_bench(backend="cupti")
```

### 2. NVIDIA Nsight Compute

```bash
# 生成性能报告
ncu --set full -o profile python examples/gemm/example_gemm.py

# 查看报告
ncu-ui profile.ncu-rep
```

### 3. NVIDIA Nsight Systems

```bash
nsys profile -o timeline python examples/gemm/example_gemm.py
nsys-ui timeline.nsys-rep
```

---

## 常见问题排查

### 1. 找不到 .so 文件

```bash
# 检查环境变量
echo $PYTHONPATH
echo $LD_LIBRARY_PATH

# 手动添加
export PYTHONPATH=/root/wkspace/tilelang:$PYTHONPATH
export LD_LIBRARY_PATH=/root/wkspace/tilelang/build:$LD_LIBRARY_PATH
```

### 2. CUDA 编译错误

```bash
# 检查 CUDA 版本
nvcc --version

# 确保 config.cmake 正确
cat build/config.cmake | grep CUDA

# 清理重新构建
rm -rf build
mkdir build && cd build
cp ../3rdparty/tvm/cmake/config.cmake .
# ... 重新配置和构建
```

### 3. FFI 调用失败

```python
# 检查 FFI 注册
import tilelang._ffi_api as ffi
print(dir(ffi))  # 查看所有可用的 FFI 函数
```

### 4. 运行时错误

```bash
# 启用详细错误信息
export TVM_BACKTRACE=1
export CUDA_LAUNCH_BLOCKING=1

# 使用 CUDA-GDB
cuda-gdb --args python examples/gemm/example_gemm.py
```

---

## 开发建议

1. **使用 editable install**: `pip install -e .` 避免频繁重装
2. **启用编译缓存**: TileLang 会缓存编译结果在 `~/.tilelang/cache`
3. **增量构建**: 修改 C++ 后只需 `make` 而无需 `cmake`
4. **代码格式化**: 运行 `./format.sh` 保持代码风格一致
5. **查看示例**: `examples/` 目录包含丰富的参考实现

---

## 进阶调试

### 查看编译管道

```python
# 在 compile 函数中添加 verbose=True
kernel = tilelang.jit(
    target="cuda",
    verbose=True  # 打印所有 Pass 信息
)(my_function)
```

### 自定义 Pass 调试

```python
from tilelang.transform import MyCustomPass

# 在 Pass 中添加打印
@tvm.tir.transform.prim_func_pass(opt_level=0)
def debug_pass(f, mod, ctx):
    print("Before transform:", f)
    # ... 变换逻辑
    print("After transform:", f)
    return f
```

### 使用 TVM 调试工具

```python
# 打印所有注册的 Pass
from tvm import transform
print(transform.PassContext.list_configs())

# 禁用特定 Pass
with tvm.transform.PassContext(disabled_pass=["vectorize"]):
    kernel = tilelang.jit(my_func)
```

---

## 快速参考

| 任务 | 命令/方法 |
|------|----------|
| 构建 C++ | `cd build && make -j` |
| 安装 Python | `pip install -e .` |
| 运行示例 | `python examples/gemm/example_gemm.py` |
| Python 调试 | VSCode F5 或 `python -m pdb script.py` |
| C++ 调试 | `gdb --args python script.py` |
| 查看 CUDA 代码 | `kernel.get_kernel_source()` |
| 性能分析 | `ncu` / `nsys` |
| 清理构建 | `rm -rf build` |

---

## 相关资源

- [TileLang 文档](https://tilelang.com/)
- [TVM 文档](https://tvm.apache.org/docs/)
- [CUTLASS 文档](https://github.com/NVIDIA/cutlass)
- [VSCode Python 调试](https://code.visualstudio.com/docs/python/debugging)
- [GDB 教程](https://sourceware.org/gdb/documentation/)
