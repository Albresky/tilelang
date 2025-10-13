## Patch RTX 4090

The **block_size** used in **SMEM** should be adapted to physical NVGPU devices accordingly, otherwise, the following error would raise.

### Fix

Apply patch: [example_elementwise_add_tma_1d.diff](./example_elementwise_add_tma_1d.diff)

**Error log:**

```log
(vtilelang) ➜  tilelang git:(main) ✗ python examples/elementwise/test_example_elementwise.py
========================================================= test session starts =========================================================
platform linux -- Python 3.13.0, pytest-8.4.2, pluggy-1.6.0
rootdir: /root/wkspace/tilelang/examples
configfile: pytest.ini
plugins: durations-1.6.1, xdist-3.8.0, timeout-2.4.0
collected 2 items                                                                                                                     

examples/elementwise/test_example_elementwise.py .F                                                                             [100%]

============================================================== FAILURES ===============================================================
_________________________________________________ test_example_elementwise_add_tma_1d _________________________________________________

    def test_example_elementwise_add_tma_1d():
>       example_elementwise_add_tma_1d.main()

examples/elementwise/test_example_elementwise.py:11: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
examples/elementwise/example_elementwise_add_tma_1d.py:45: in main
    kernel = elementwise_add(M, N, **config, in_dtype="float32", out_dtype="float32")
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/root/miniconda3/envs/vtilelang/lib/python3.13/site-packages/tilelang/jit/__init__.py:216: in wrapper
    kernel_result = compile(
/root/miniconda3/envs/vtilelang/lib/python3.13/site-packages/tilelang/jit/__init__.py:81: in compile
    return cached(
/root/miniconda3/envs/vtilelang/lib/python3.13/site-packages/tilelang/cache/__init__.py:28: in cached
    return _kernel_cache_instance.cached(
/root/miniconda3/envs/vtilelang/lib/python3.13/site-packages/tilelang/cache/kernel_cache.py:138: in cached
    return JITKernel(
/root/miniconda3/envs/vtilelang/lib/python3.13/site-packages/tilelang/jit/kernel.py:129: in __init__
    adapter = self._compile_and_create_adapter(func, out_idx)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/root/miniconda3/envs/vtilelang/lib/python3.13/site-packages/tilelang/jit/kernel.py:258: in _compile_and_create_adapter
    adapter = CythonKernelAdapter(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <tilelang.jit.adapter.cython.adapter.CythonKernelAdapter object at 0x7775a897c7d0>
params = [KernelParam(dtype=torch.float32, shape=[128, 128]), KernelParam(dtype=torch.float32, shape=[128, 128]), KernelParam(dtype=torch.float32, shape=[128, 128])]
result_idx = [2], target = cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -thread_warp_size=32
func_or_mod = # from tvm.script import tir as T

@T.prim_func
def elem_add(A_handle: T.handle, B_handle: T.handle, C_handle: T.handl...      T.copy(T.region(C_shared[0, 0], 1, 128, 128), T.region(C[by * 128, bx * 128], 2, 128, 128), -1, T.bool(False), 0)
host_mod = # from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    de..., "elem_add_compute_"):
            T.call_packed("elem_add_kernel", A, B, C, 1, 1, 128, 1, 1, 131072)
        return 0
device_mod = # from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    de...          C_1[i * 512 + tx * 4:i * 512 + tx * 4 + 4] = C_shared[T.Ramp(i * 512 + tx * 4, 1, 4) + T.Broadcast(16384, 4)]
kernel_global_source = '#include <tl_templates/cuda/gemm.h>\n#include <tl_templates/cuda/copy.h>\n#include <tl_templates/cuda/reduce.h>\n#inc...dIdx.x) * 4))) = *(float4*)(((float*)buf_dyn_shmem) + (((i_4 * 512) + (((int)threadIdx.x) * 4)) + 16384));\n  }\n}\n\n'
verbose = False, pass_configs = {}, compile_flags = None

    def __init__(self,
                 params: List[KernelParam],
                 result_idx: List[int],
                 target: Union[str, Target],
                 func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
                 host_mod: Optional[tvm.IRModule] = None,
                 device_mod: Optional[tvm.IRModule] = None,
                 kernel_global_source: Optional[str] = None,
                 verbose: bool = False,
                 pass_configs: Optional[Dict[str, Any]] = None,
                 compile_flags: Optional[List[str]] = None):
        """Initialize the adapter with the given TIR function or module.
    
        Args:
            params: List of tensor types for inputs/outputs
            result_idx: Indices of output tensors
            target: Target platform (e.g., 'cuda')
            func_or_mod: TIR function or module to be compiled
            verbose: Enable verbose logging
        """
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.kernel_global_source = kernel_global_source
    
        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod
    
        self.target = Target.canon_target(determine_target(target))
    
        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self.buffer_dtype_map = self._process_buffer_dtype()
        self.ptr_map = self._process_ptr_map()
        self.buffer_device_map = self._process_buffer_device()
    
        static_buffer_infos = self._process_static_buffer_infos()
        self.static_shape_map = static_buffer_infos[0]
        self.static_strides_map = static_buffer_infos[1]
        self.static_contiguous_list = static_buffer_infos[2]
    
        self.verbose = verbose
        self.wrapper = TLWrapper(self.target)
        self.lib_generator = LibraryGenerator(self.target, verbose=verbose)
        self.lib_generator.assign_pass_configs(pass_configs)
        self.lib_generator.assign_compile_flags(compile_flags)
    
        self.wrapper.assign_optimized_module(self.ir_module)
        self.wrapper.assign_pass_configs(pass_configs)
        self.wrapper.assign_host_module(host_mod)
        self.wrapper.assign_device_module(device_mod)
        self.wrapped_source = self.wrapper.wrap(self.get_kernel_source(kernel_only=True))
    
        self.lib_generator.update_lib_code(self.wrapped_source)
        self.lib_generator.compile_lib()
        self.lib = self.lib_generator.load_lib()
    
        self.lib.get_last_error.restype = ctypes.c_char_p
        result = self.lib.init()
        if result != 0:
            error_msg = self.lib.get_last_error().decode('utf-8')
            error_msg += f"\n{self.lib_code}"
>           raise RuntimeError(f"Initialization failed: {error_msg}")
E           RuntimeError: Initialization failed: Failed to set the allowed dynamic shared memory size to 131072 with error: invalid argument
E           #include <tl_templates/cuda/gemm.h>
E           #include <tl_templates/cuda/copy.h>
E           #include <tl_templates/cuda/reduce.h>
E           #include <tl_templates/cuda/ldsm.h>
E           #include <tl_templates/cuda/threadblock_swizzle.h>
E           #include <tl_templates/cuda/debug.h>
E           #ifdef ENABLE_BF16
E           #include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
E           #endif
E           
E           extern "C" __global__ void elem_add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C);
E           extern "C" __global__ void __launch_bounds__(128, 1) elem_add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
E             extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
E             float C_local[128];
E             #pragma unroll
E             for (int i = 0; i < 32; ++i) {
E               *(float4*)(((float*)buf_dyn_shmem) + (((i * 512) + (((int)threadIdx.x) * 4)) + 16384)) = *(float4*)(A + ((i * 512) + (((int)threadIdx.x) * 4)));
E             }
E             #pragma unroll
E             for (int i_1 = 0; i_1 < 32; ++i_1) {
E               *(float4*)(((float*)buf_dyn_shmem) + ((i_1 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(B + ((i_1 * 512) + (((int)threadIdx.x) * 4)));
E             }
E             __syncthreads();
E             #pragma unroll
E             for (int i_2 = 0; i_2 < 32; ++i_2) {
E               float4 __1;
E                 float4 v_ = *(float4*)(((float*)buf_dyn_shmem) + (((i_2 * 512) + (((int)threadIdx.x) * 4)) + 16384));
E                 float4 v__1 = *(float4*)(((float*)buf_dyn_shmem) + ((i_2 * 512) + (((int)threadIdx.x) * 4)));
E                 __1.x = (v_.x+v__1.x);
E                 __1.y = (v_.y+v__1.y);
E                 __1.z = (v_.z+v__1.z);
E                 __1.w = (v_.w+v__1.w);
E               *(float4*)(C_local + (i_2 * 4)) = __1;
E             }
E             __syncthreads();
E             #pragma unroll
E             for (int i_3 = 0; i_3 < 32; ++i_3) {
E               *(float4*)(((float*)buf_dyn_shmem) + (((i_3 * 512) + (((int)threadIdx.x) * 4)) + 16384)) = *(float4*)(C_local + (i_3 * 4));
E             }
E             __syncthreads();
E             #pragma unroll
E             for (int i_4 = 0; i_4 < 32; ++i_4) {
E               *(float4*)(C + ((i_4 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(((float*)buf_dyn_shmem) + (((i_4 * 512) + (((int)threadIdx.x) * 4)) + 16384));
E             }
E           }
E           
E           
E           #define ERROR_BUF_SIZE 1024
E           static char error_buf[ERROR_BUF_SIZE];
E           
E           extern "C" const char* get_last_error() {
E               return error_buf;
E           }
E           
E           extern "C" int init() {
E               error_buf[0] = '\0';
E               
E               cudaError_t result_elem_add_kernel = cudaFuncSetAttribute(elem_add_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 131072);
E               if (result_elem_add_kernel != CUDA_SUCCESS) {
E                   snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 131072, cudaGetErrorString(result_elem_add_kernel));
E                   return -1;
E               }
E           
E               return 0;
E           }
E           
E           extern "C" int call(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {
E               elem_add_kernel<<<dim3(1, 1, 1), dim3(128, 1, 1), 131072, stream>>>(A, B, C);
E               TILELANG_CHECK_LAST_ERROR("elem_add_kernel");
E           
E               return 0;
E           }

/root/miniconda3/envs/vtilelang/lib/python3.13/site-packages/tilelang/jit/adapter/cython/adapter.py:277: RuntimeError
-------------------------------------------------------- Captured stdout call ---------------------------------------------------------
2025-10-13 06:00:43  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `elem_add` with `out_idx=[-1]`
======================================================== fixture duration top =========================================================
total          name                                                                         num med            min           
       0:00:00 grand total                                                                    0        0:00:00        0:00:00
======================================================= test call duration top ========================================================
total          name                                                                         num med            min           
0:00:08.419308 elementwise/test_example_elementwise.py::test_example_elementwise_add          1 0:00:08.419308 0:00:08.419308
0:00:08.301374 elementwise/test_example_elementwise.py::test_example_elementwise_add_tma_1d   1 0:00:08.301374 0:00:08.301374
0:00:16.720682 grand total                                                                    2 0:00:08.360341 0:00:08.301374
======================================================= test setup duration top =======================================================
total          name                                                                         num med            min           
0:00:00.000194 grand total                                                                    2 0:00:00.000097 0:00:00.000091
===================================================== test teardown duration top ======================================================
total          name                                                                         num med            min           
0:00:00.000250 grand total                                                                    2 0:00:00.000125 0:00:00.000109
======================================================= short test summary info =======================================================
FAILED examples/elementwise/test_example_elementwise.py::test_example_elementwise_add_tma_1d - RuntimeError: Initialization failed: Failed to set the allowed dynamic shared memory size to 131072 with error: invalid argument
==================================================== 1 failed, 1 passed in 16.80s =====================================================
(vtilelang) ➜  tilelang git:(main) ✗ python examples/elementwise/test_example_elementwise.py
========================================================= test session starts =========================================================
platform linux -- Python 3.13.0, pytest-8.4.2, pluggy-1.6.0
rootdir: /root/wkspace/tilelang/examples
configfile: pytest.ini
plugins: durations-1.6.1, xdist-3.8.0, timeout-2.4.0
collected 2 items                                                                                                                     

examples/elementwise/test_example_elementwise.py .F                                                                             [100%]

============================================================== FAILURES ===============================================================
_________________________________________________ test_example_elementwise_add_tma_1d _________________________________________________

    def test_example_elementwise_add_tma_1d():
>       example_elementwise_add_tma_1d.main()

examples/elementwise/test_example_elementwise.py:11: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
examples/elementwise/example_elementwise_add_tma_1d.py:45: in main
    kernel = elementwise_add(M, N, **config, in_dtype="float32", out_dtype="float32")
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/root/miniconda3/envs/vtilelang/lib/python3.13/site-packages/tilelang/jit/__init__.py:216: in wrapper
    kernel_result = compile(
/root/miniconda3/envs/vtilelang/lib/python3.13/site-packages/tilelang/jit/__init__.py:81: in compile
    return cached(
/root/miniconda3/envs/vtilelang/lib/python3.13/site-packages/tilelang/cache/__init__.py:28: in cached
    return _kernel_cache_instance.cached(
/root/miniconda3/envs/vtilelang/lib/python3.13/site-packages/tilelang/cache/kernel_cache.py:138: in cached
    return JITKernel(
/root/miniconda3/envs/vtilelang/lib/python3.13/site-packages/tilelang/jit/kernel.py:129: in __init__
    adapter = self._compile_and_create_adapter(func, out_idx)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/root/miniconda3/envs/vtilelang/lib/python3.13/site-packages/tilelang/jit/kernel.py:258: in _compile_and_create_adapter
    adapter = CythonKernelAdapter(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <tilelang.jit.adapter.cython.adapter.CythonKernelAdapter object at 0x79e2f9e0c190>
params = [KernelParam(dtype=torch.float32, shape=[128, 128]), KernelParam(dtype=torch.float32, shape=[128, 128]), KernelParam(dtype=torch.float32, shape=[128, 128])]
result_idx = [2], target = cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -thread_warp_size=32
func_or_mod = # from tvm.script import tir as T

@T.prim_func
def elem_add(A_handle: T.handle, B_handle: T.handle, C_handle: T.handl...      T.copy(T.region(C_shared[0, 0], 1, 128, 128), T.region(C[by * 128, bx * 128], 2, 128, 128), -1, T.bool(False), 0)
host_mod = # from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    de..., "elem_add_compute_"):
            T.call_packed("elem_add_kernel", A, B, C, 1, 1, 128, 1, 1, 131072)
        return 0
device_mod = # from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    de...          C_1[i * 512 + tx * 4:i * 512 + tx * 4 + 4] = C_shared[T.Ramp(i * 512 + tx * 4, 1, 4) + T.Broadcast(16384, 4)]
kernel_global_source = '#include <tl_templates/cuda/gemm.h>\n#include <tl_templates/cuda/copy.h>\n#include <tl_templates/cuda/reduce.h>\n#inc...dIdx.x) * 4))) = *(float4*)(((float*)buf_dyn_shmem) + (((i_4 * 512) + (((int)threadIdx.x) * 4)) + 16384));\n  }\n}\n\n'
verbose = False, pass_configs = {}, compile_flags = None

    def __init__(self,
                 params: List[KernelParam],
                 result_idx: List[int],
                 target: Union[str, Target],
                 func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
                 host_mod: Optional[tvm.IRModule] = None,
                 device_mod: Optional[tvm.IRModule] = None,
                 kernel_global_source: Optional[str] = None,
                 verbose: bool = False,
                 pass_configs: Optional[Dict[str, Any]] = None,
                 compile_flags: Optional[List[str]] = None):
        """Initialize the adapter with the given TIR function or module.
    
        Args:
            params: List of tensor types for inputs/outputs
            result_idx: Indices of output tensors
            target: Target platform (e.g., 'cuda')
            func_or_mod: TIR function or module to be compiled
            verbose: Enable verbose logging
        """
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.kernel_global_source = kernel_global_source
    
        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod
    
        self.target = Target.canon_target(determine_target(target))
    
        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self.buffer_dtype_map = self._process_buffer_dtype()
        self.ptr_map = self._process_ptr_map()
        self.buffer_device_map = self._process_buffer_device()
    
        static_buffer_infos = self._process_static_buffer_infos()
        self.static_shape_map = static_buffer_infos[0]
        self.static_strides_map = static_buffer_infos[1]
        self.static_contiguous_list = static_buffer_infos[2]
    
        self.verbose = verbose
        self.wrapper = TLWrapper(self.target)
        self.lib_generator = LibraryGenerator(self.target, verbose=verbose)
        self.lib_generator.assign_pass_configs(pass_configs)
        self.lib_generator.assign_compile_flags(compile_flags)
    
        self.wrapper.assign_optimized_module(self.ir_module)
        self.wrapper.assign_pass_configs(pass_configs)
        self.wrapper.assign_host_module(host_mod)
        self.wrapper.assign_device_module(device_mod)
        self.wrapped_source = self.wrapper.wrap(self.get_kernel_source(kernel_only=True))
    
        self.lib_generator.update_lib_code(self.wrapped_source)
        self.lib_generator.compile_lib()
        self.lib = self.lib_generator.load_lib()
    
        self.lib.get_last_error.restype = ctypes.c_char_p
        result = self.lib.init()
        if result != 0:
            error_msg = self.lib.get_last_error().decode('utf-8')
            error_msg += f"\n{self.lib_code}"
>           raise RuntimeError(f"Initialization failed: {error_msg}")
E           RuntimeError: Initialization failed: Failed to set the allowed dynamic shared memory size to 131072 with error: invalid argument
E           #include <tl_templates/cuda/gemm.h>
E           #include <tl_templates/cuda/copy.h>
E           #include <tl_templates/cuda/reduce.h>
E           #include <tl_templates/cuda/ldsm.h>
E           #include <tl_templates/cuda/threadblock_swizzle.h>
E           #include <tl_templates/cuda/debug.h>
E           #ifdef ENABLE_BF16
E           #include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
E           #endif
E           
E           extern "C" __global__ void elem_add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C);
E           extern "C" __global__ void __launch_bounds__(128, 1) elem_add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
E             extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
E             float C_local[128];
E             #pragma unroll
E             for (int i = 0; i < 32; ++i) {
E               *(float4*)(((float*)buf_dyn_shmem) + (((i * 512) + (((int)threadIdx.x) * 4)) + 16384)) = *(float4*)(A + ((i * 512) + (((int)threadIdx.x) * 4)));
E             }
E             #pragma unroll
E             for (int i_1 = 0; i_1 < 32; ++i_1) {
E               *(float4*)(((float*)buf_dyn_shmem) + ((i_1 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(B + ((i_1 * 512) + (((int)threadIdx.x) * 4)));
E             }
E             __syncthreads();
E             #pragma unroll
E             for (int i_2 = 0; i_2 < 32; ++i_2) {
E               float4 __1;
E                 float4 v_ = *(float4*)(((float*)buf_dyn_shmem) + (((i_2 * 512) + (((int)threadIdx.x) * 4)) + 16384));
E                 float4 v__1 = *(float4*)(((float*)buf_dyn_shmem) + ((i_2 * 512) + (((int)threadIdx.x) * 4)));
E                 __1.x = (v_.x+v__1.x);
E                 __1.y = (v_.y+v__1.y);
E                 __1.z = (v_.z+v__1.z);
E                 __1.w = (v_.w+v__1.w);
E               *(float4*)(C_local + (i_2 * 4)) = __1;
E             }
E             __syncthreads();
E             #pragma unroll
E             for (int i_3 = 0; i_3 < 32; ++i_3) {
E               *(float4*)(((float*)buf_dyn_shmem) + (((i_3 * 512) + (((int)threadIdx.x) * 4)) + 16384)) = *(float4*)(C_local + (i_3 * 4));
E             }
E             __syncthreads();
E             #pragma unroll
E             for (int i_4 = 0; i_4 < 32; ++i_4) {
E               *(float4*)(C + ((i_4 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(((float*)buf_dyn_shmem) + (((i_4 * 512) + (((int)threadIdx.x) * 4)) + 16384));
E             }
E           }
E           
E           
E           #define ERROR_BUF_SIZE 1024
E           static char error_buf[ERROR_BUF_SIZE];
E           
E           extern "C" const char* get_last_error() {
E               return error_buf;
E           }
E           
E           extern "C" int init() {
E               error_buf[0] = '\0';
E               
E               cudaError_t result_elem_add_kernel = cudaFuncSetAttribute(elem_add_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 131072);
E               if (result_elem_add_kernel != CUDA_SUCCESS) {
E                   snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 131072, cudaGetErrorString(result_elem_add_kernel));
E                   return -1;
E               }
E           
E               return 0;
E           }
E           
E           extern "C" int call(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {
E               elem_add_kernel<<<dim3(1, 1, 1), dim3(128, 1, 1), 131072, stream>>>(A, B, C);
E               TILELANG_CHECK_LAST_ERROR("elem_add_kernel");
E           
E               return 0;
E           }

/root/miniconda3/envs/vtilelang/lib/python3.13/site-packages/tilelang/jit/adapter/cython/adapter.py:277: RuntimeError
-------------------------------------------------------- Captured stdout call ---------------------------------------------------------
2025-10-13 06:01:15  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `elem_add` with `out_idx=[-1]`
======================================================== fixture duration top =========================================================
total          name                                                                         num med            min           
       0:00:00 grand total                                                                    0        0:00:00        0:00:00
======================================================= test call duration top ========================================================
total          name                                                                         num med            min           
0:00:08.654897 elementwise/test_example_elementwise.py::test_example_elementwise_add          1 0:00:08.654897 0:00:08.654897
0:00:07.840129 elementwise/test_example_elementwise.py::test_example_elementwise_add_tma_1d   1 0:00:07.840129 0:00:07.840129
0:00:16.495026 grand total                                                                    2 0:00:08.247513 0:00:07.840129
======================================================= test setup duration top =======================================================
total          name                                                                         num med            min           
0:00:00.000191 grand total                                                                    2 0:00:00.000095 0:00:00.000089
===================================================== test teardown duration top ======================================================
total          name                                                                         num med            min           
0:00:00.000222 grand total                                                                    2 0:00:00.000111 0:00:00.000098
======================================================= short test summary info =======================================================
FAILED examples/elementwise/test_example_elementwise.py::test_example_elementwise_add_tma_1d - RuntimeError: Initialization failed: Failed to set the allowed dynamic shared memory size to 131072 with error: invalid argument
==================================================== 1 failed, 1 passed in 16.57s =====================================================
```