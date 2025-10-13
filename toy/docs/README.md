# TileLang 开发与调试资源汇总

本目录包含完整的 TileLang 开发、调试和性能分析资源。

## 项目结构

```
tilelang/
├── 📄 DEBUGGING_GUIDE.md      ← 完整调试指南
├── 📄 QUICK_REFERENCE.md      ← 快速参考
├── 📄 COMMON_ERRORS.md        ← 常见错误
├── 📄 debug_example.py        ← 调试示例
├── 📄 Makefile.dev            ← 开发工具
│
├── tilelang/                  ← Python 前端
│   ├── language/              ← DSL 语法定义
│   ├── jit/                   ← JIT 编译器
│   │   ├── __init__.py        ← 编译入口 (设断点)
│   │   └── kernel.py          ← Kernel 包装
│   ├── primitives/            ← 算子原语
│   │   ├── gemm.py            ← GEMM 实现
│   │   └── copy.py            ← Copy 实现
│   └── transform/             ← IR 变换 Pass
│
├── src/                       ← C++ 后端
│   ├── ir.cc                  ← IR 节点定义
│   ├── transform/             ← C++ Pass
│   ├── target/                ← 代码生成
│   └── tl_templates/          ← CUTLASS/CuTe 模板
│
├── examples/                  ← 示例代码
│   ├── gemm/                  ← 矩阵乘法
│   ├── flash_attention/       ← FlashAttention
│   └── ...                    ← 更多算子
│
└── testing/                   ← 测试用例
```

---

## 📚 文档索引

### 快速开始
- **[QUICK_REFERENCE.md](02-QUICK_REFERENCE.md)** - 快速参考卡片，包含最常用的命令和技巧
- **[debug_example.py](../debug_example.py)** - 完整的调试示例脚本

### 详细指南
- **[DEBUGGING_GUIDE.md](01-DEBUGGING_GUIDE.md)** - 完整的联调指南，涵盖从构建到调试的所有方面
- **[COMMON_ERRORS.md](03-COMMON_ERRORS.md)** - 常见错误及解决方案

### 工具
- **[../Makefile.dev](../Makefile)** - 开发用 Makefile，简化常见任务
- **[../.vscode/launch.json](../.vscode/launch.json)** - VSCode 调试配置

---

## 快速上手

```bash
# 1. 构建项目
git clone --recursive https://github.com/tile-ai/tilelang
cd tilelang
mkdir build && cd build
cp ../3rdparty/tvm/cmake/config.cmake .
echo "set(USE_CUDA ON)" >> config.cmake
cmake .. && make -j$(nproc)
cd ..

# 2. 安装 (开发模式)
pip install -e .

# 3. 运行示例
python examples/gemm/example_gemm.py

# 4. 运行调试示例
python debug_example.py
```

---

## 按场景查找

#### 开发
1. 阅读 [DEBUGGING_GUIDE.md](DEBUGGING_GUIDE.md) 的"构建与安装"部分
2. 使用 `make -f Makefile.dev init` 初始化环境
3. 运行 `python debug_example.py` 验证安装

#### 调试 Python 代码
1. 在 VSCode 中打开项目
2. 按 `F5` 启动调试 (使用预配置的 launch.json)
3. 参考 [DEBUGGING_GUIDE.md](DEBUGGING_GUIDE.md) 的"调试配置"部分

#### 调试 C++ 代码
1. 查看 [DEBUGGING_GUIDE.md](DEBUGGING_GUIDE.md) 的"方法 2: C++ 后端调试"
2. 使用 `gdb --args python script.py`
3. 或使用 VSCode 的 "Attach to Python Process" 配置

#### 查看生成的代码
```python
kernel = matmul(1024, 1024, 1024, 128, 128, 32)
print(kernel.get_kernel_source())  # CUDA 源码
print(kernel.mod.script())         # TVM IR
```

#### 解决编译错误
1. 查看 [COMMON_ERRORS.md](COMMON_ERRORS.md) 找到类似错误
2. 启用详细日志: `export TVM_LOG_DEBUG=1`
3. 清理重新构建: `make -f Makefile.dev clean && make -f Makefile.dev build`

#### 性能分析
```bash
# 使用内置 profiler
profiler = kernel.get_profiler()
latency = profiler.do_bench()

# 使用 Nsight Compute
ncu -o profile --set full python script.py

# 使用 Nsight Systems
nsys profile -o timeline python script.py
```

#### 修改代码后测试
```bash
# Python 代码修改 (editable install)
python your_script.py  # 直接运行

# C++ 代码修改
cd build && make -j
python your_script.py
```

---

## 常用命令

| 任务 | 命令 |
|------|------|
| 构建 | `cd build && make -j` |
| 快速构建 | `make -f Makefile.dev rebuild` |
| 安装 | `pip install -e .` |
| 清理 | `make -f Makefile.dev clean` |
| 运行测试 | `pytest testing/ -v` |
| 格式化 | `./format.sh` |
| Python 调试 | VSCode `F5` 或 `python -m pdb script.py` |
| C++ 调试 | `gdb --args python script.py` |
| 性能分析 | `ncu`/`nsys` 或 `profiler.do_bench()` |

---


## 💡 最佳实践

### 开发环境
- ✅ 使用 `pip install -e .` 进行 editable install
- ✅ 使用 `RelWithDebInfo` 构建模式 (优化 + 调试符号)
- ✅ 启用 VSCode 的 Python 和 C++ 扩展
- ✅ 使用 git pre-commit hooks 保持代码质量

### 调试技巧
- ✅ 在关键路径设置断点 (jit/compile, primitives, transform)
- ✅ 使用 `kernel.get_kernel_source()` 查看生成代码
- ✅ 启用 `verbose=True` 查看编译 Pass
- ✅ 使用 `CUDA_LAUNCH_BLOCKING=1` 同步 CUDA 调用

### 性能优化
- ✅ 先确保正确性，再优化性能
- ✅ 使用 profiler 建立 baseline
- ✅ 一次只改一个参数，观察影响
- ✅ 参考 `examples/` 中的最佳实践
- ✅ 使用 Nsight 工具深入分析

### 代码贡献
- ✅ 运行 `./format.sh` 格式化代码
- ✅ 添加测试用例到 `testing/`
- ✅ 更新文档说明新功能
- ✅ 提交前运行 `pytest testing/ -v`

---

## 🎓 学习资源

### TileLang 相关
- [官方文档](https://tilelang.com/)
- [GitHub 仓库](https://github.com/tile-ai/tilelang)
- [示例代码](examples/)

### 依赖项目
- [TVM 文档](https://tvm.apache.org/docs/)
- [CUTLASS 文档](https://github.com/NVIDIA/cutlass)
- [CuTe 教程](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)

### 工具
- [VSCode Python 调试](https://code.visualstudio.com/docs/python/debugging)
- [GDB 教程](https://sourceware.org/gdb/documentation/)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)

---

## 性能基准

参考 [../../README.md](../../README.md) 中的 Benchmark Summary 部分，了解 TileLang 在各种算子和硬件上的性能表现。
