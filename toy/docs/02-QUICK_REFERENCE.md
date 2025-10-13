# TileLang 联调快速参考卡

## 🚀 快速开始

```bash
# 1. 克隆并构建
git clone --recursive https://github.com/tile-ai/tilelang
cd tilelang
make -f Makefile.dev init

# 2. 运行示例
python examples/gemm/example_gemm.py

# 3. 调试示例
python debug_example.py
```

## 🔧 常用命令

| 任务 | 命令 |
|------|------|
| 构建 C++ | `make -f Makefile.dev build` 或 `cd build && make -j` |
| 快速重编译 | `make -f Makefile.dev rebuild` |
| 安装 Python | `pip install -e .` (开发模式) |
| 清理 | `make -f Makefile.dev clean` |
| 运行测试 | `pytest testing/ -v` |
| 格式化代码 | `./format.sh` |

## 🐛 调试技巧

### Python 调试
```bash
# VSCode: 按 F5 (使用 .vscode/launch.json)
# 或命令行:
python -m pdb script.py
```

### C++ 调试
```bash
# 使用 GDB
gdb --args python examples/gemm/example_gemm.py
# 在 gdb 中:
(gdb) b tilelang::SomeFunction  # 设置断点
(gdb) run                       # 运行
(gdb) bt                        # 查看调用栈
```

### 查看生成的代码
```python
kernel = matmul(1024, 1024, 1024, 128, 128, 32)

# 查看 CUDA 源码
print(kernel.get_kernel_source())

# 查看 TVM IR
print(kernel.mod.script())
```

## 🔍 常用断点位置

| 位置 | 说明 |
|------|------|
| `tilelang/jit/__init__.py:compile()` | JIT 编译入口 |
| `tilelang/primitives/gemm.py` | GEMM 算子实现 |
| `tilelang/transform/*.py` | IR 变换 Pass |
| `tilelang/jit/adapter/*.py` | 后端适配器 |

## 🌍 环境变量

```bash
# 调试相关
export TVM_LOG_DEBUG=1              # TVM 详细日志
export TVM_BACKTRACE=1              # 详细回溯
export TVM_DUMP_IR=1                # 保存中间 IR
export CUDA_LAUNCH_BLOCKING=1       # CUDA 同步模式

# 性能相关
export TVM_KEEP_SOURCE=1            # 保留生成的源码
export TILELANG_CACHE_DIR=~/.tilelang/cache  # 缓存目录
```

## 📊 性能分析

```bash
# Nsight Compute (详细性能指标)
ncu -o profile --set full python script.py
ncu-ui profile.ncu-rep

# Nsight Systems (时间线分析)
nsys profile -o timeline python script.py
nsys-ui timeline.nsys-rep

# 内置 Profiler
profiler = kernel.get_profiler()
latency = profiler.do_bench()
```

## 📝 代码修改工作流

### 修改 Python 代码
```bash
# 如果是 editable install (-e)
# 直接运行即可，无需重新安装
python your_script.py
```

### 修改 C++ 代码
```bash
# 1. 修改 src/ 下的 C++ 文件
# 2. 重新编译
cd build && make -j
# 或
make -f Makefile.dev rebuild

# 3. Python 会自动使用新的 .so
python your_script.py
```

### 修改 CUTLASS/CuTe 模板
```bash
# 1. 修改 src/tl_templates/ 下的文件
# 2. 重新编译
cd build && make -j

# 3. 清理缓存 (可选)
rm -rf ~/.tilelang/cache
```

## 🔥 常见问题

### 找不到 .so 文件
```bash
export PYTHONPATH=/path/to/tilelang:$PYTHONPATH
export LD_LIBRARY_PATH=/path/to/tilelang/build:$LD_LIBRARY_PATH
```

### CUDA 编译失败
```bash
# 检查 CUDA 版本
nvcc --version

# 清理重新构建
rm -rf build
make -f Makefile.dev build
```

### 运行时错误
```bash
# 启用详细错误信息
export TVM_BACKTRACE=1
export CUDA_LAUNCH_BLOCKING=1

# 使用 CUDA-GDB
cuda-gdb --args python script.py
```

## 📚 项目结构速查

```
tilelang/
├── tilelang/           # Python 前端
│   ├── language/       # DSL 语法
│   ├── jit/           # JIT 编译器
│   ├── primitives/    # 算子原语
│   └── transform/     # IR 变换
├── src/               # C++ 后端
│   ├── ir.cc          # IR 节点
│   ├── transform/     # C++ Pass
│   ├── target/        # 代码生成
│   └── tl_templates/  # CUTLASS 模板
├── 3rdparty/
│   ├── tvm/          # TVM 编译器
│   ├── cutlass/      # NVIDIA CUTLASS
│   └── composable_kernel/  # AMD CK
├── examples/         # 示例代码
└── testing/          # 测试
```

## 🎯 调试检查清单

- [ ] 使用 `editable install`: `pip install -e .`
- [ ] 启用详细日志: `export TVM_LOG_DEBUG=1`
- [ ] VSCode 配置正确: 检查 `.vscode/launch.json`
- [ ] 断点设置在关键位置
- [ ] CUDA 同步模式: `export CUDA_LAUNCH_BLOCKING=1`
- [ ] 查看生成的代码: `kernel.get_kernel_source()`
- [ ] 使用 profiler: `profiler.do_bench()`

## 🆘 获取帮助

- 📖 文档: https://tilelang.com/
- 💬 Discord: https://discord.gg/TUrHyJnKPG
- 🐛 Issues: https://github.com/tile-ai/tilelang/issues
- 📧 邮件列表: (查看 README.md)

## ⚡ 性能优化提示

1. 使用 `RelWithDebInfo` 构建模式 (默认)
2. 启用编译缓存 (自动)
3. 调整 block size 参数
4. 使用 pipeline 和 swizzle 优化
5. 参考 `examples/` 中的最佳实践

---

**更多详细信息请查看 DEBUGGING_GUIDE.md**
