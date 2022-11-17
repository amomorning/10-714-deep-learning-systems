#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

namespace needle {
namespace metal {

using scalar_t = float;
constexpr int kBaseThreadNum = 256;
constexpr int kTileSize = 8;
constexpr int kElemSize = sizeof(scalar_t);

struct MetalDims {
  MTL::Size num_threads_per_group;
  MTL::Size num_threads_per_grid;
};

MetalDims MetalOneDim(size_t size) {
  MetalDims dim;
  dim.num_threads_per_group = MTL::Size(kBaseThreadNum, 1, 1);
  dim.num_threads_per_grid = MTL::Size(size, 1, 1);
  return dim;
}

template<typename T = scalar_t>
struct MetalArray {
  explicit MetalArray(size_t size);
  ~MetalArray();

  size_t ptr_as_int() { return (size_t)ptr; }

  size_t size = 0;
  MTL::Buffer* buffer = nullptr;
  T* ptr = nullptr;
};

template<typename T>
MetalArray<T> VecToMetal(const std::vector<T>& vec) {
  MetalArray<T> arr(vec.size());
  std::memcpy(arr.ptr, vec.data(), vec.size() * sizeof(T));
  return arr;
}

class MyMetal {
 public:
  static MyMetal* GetInstance();

  ~MyMetal();

  MTL::Buffer*
  NewBuffer(NS::UInteger length,
            MTL::ResourceOptions options = MTL::ResourceStorageModeShared);

  MTL::Device* device() { return device_; }
  MTL::CommandQueue* command_queue() { return command_queue_; }

  void RegisterKernel(const std::string& kernel_name);
  MTL::ComputePipelineState*
  GetComputePipelineState(const std::string& kernel_name);

 private:
  MyMetal();

  void LoadKernelsFromFile();

  MTL::Device* device_ = nullptr;
  MTL::CommandQueue* command_queue_ = nullptr;
  MTL::Library* library_ = nullptr;
  std::unordered_map<std::string, MTL::ComputePipelineState*> kernel_name2cps_;
};

// ************************** Implementation ***************************

template<typename T>
MetalArray<T>::MetalArray(size_t size) : size(size) {
  buffer = MyMetal::GetInstance()->NewBuffer(size * sizeof(T));
  ptr = reinterpret_cast<T*>(buffer->contents());
}

template<typename T>
MetalArray<T>::~MetalArray() {
  if (buffer)
    buffer->release();
}

MyMetal* MyMetal::GetInstance() {
  static std::shared_ptr<MyMetal> instance = nullptr;
  static std::mutex mutex{};
  if (instance == nullptr) {
    mutex.lock();
    if (instance == nullptr) {
      instance = std::shared_ptr<MyMetal>(new MyMetal);
    }
    mutex.unlock();
  }
  return instance.get();
}

MyMetal::MyMetal() {
  device_ = MTL::CreateSystemDefaultDevice();
  assert(device_);
  command_queue_ = device_->newCommandQueue();
  assert(command_queue_);
  LoadKernelsFromFile();
}

MyMetal::~MyMetal() {
  for (auto& [kernel_name, cps] : kernel_name2cps_) {
    if (cps)
      cps->release();
  }
  if (library_)
    library_->release();
  if (command_queue_)
    command_queue_->release();
  if (device_)
    device_->release();
}

MTL::Buffer* MyMetal::NewBuffer(NS::UInteger length,
                                MTL::ResourceOptions options) {
  assert(device_);
  return device_->newBuffer(length, options);
}

void MyMetal::LoadKernelsFromFile() {
  NS::Error* error = nullptr;
  library_ = device_->newLibrary(
      NS::String::string("kernels.metallib", NS::UTF8StringEncoding), &error);
  if (!library_) {
    __builtin_printf("%s", error->localizedDescription()->utf8String());
    assert(false);
  }
}

void MyMetal::RegisterKernel(const std::string& kernel_name) {
  if (kernel_name2cps_.count(kernel_name))
    return;
  NS::Error* error = nullptr;
  MTL::Function* func = library_->newFunction(
      NS::String::string(kernel_name.c_str(), NS::UTF8StringEncoding));
  MTL::ComputePipelineState* cps =
      device_->newComputePipelineState(func, &error);
  if (!cps) {
    __builtin_printf("%s", error->localizedDescription()->utf8String());
    assert(false);
  }
  assert(kernel_name2cps_.emplace(kernel_name, cps).second);
  func->release();
}

MTL::ComputePipelineState*
MyMetal::GetComputePipelineState(const std::string& kernel_name) {
  auto iter = kernel_name2cps_.find(kernel_name);
  if (iter == kernel_name2cps_.end()) {
    RegisterKernel(kernel_name);
    return kernel_name2cps_[kernel_name];
  }
  return iter->second;
}

#define BEGIN_COMPUTE_COMMAND(command_kernel_name)                             \
  MyMetal* metal = MyMetal::GetInstance();                                     \
  MTL::CommandBuffer* command_buffer =                                         \
      metal->command_queue()->commandBuffer();                                 \
  MTL::ComputeCommandEncoder* command_encoder =                                \
      command_buffer->computeCommandEncoder();                                 \
  command_encoder->setComputePipelineState(                                    \
      metal->GetComputePipelineState(command_kernel_name));

#define END_COMPUTE_COMMAND                                                    \
  command_encoder->endEncoding();                                              \
  command_buffer->commit();                                                    \
  command_buffer->waitUntilCompleted();                                        \
  command_encoder->release();                                                  \
  command_buffer->release();

void Fill(MetalArray<scalar_t>* out, scalar_t val, size_t size) {
  BEGIN_COMPUTE_COMMAND("FillKernel")

  command_encoder->setBuffer(out->buffer, 0, 0);
  MetalArray<scalar_t> val_arr =
      VecToMetal<scalar_t>(std::vector<scalar_t>{val});
  command_encoder->setBuffer(val_arr.buffer, 0, 1);
  MetalDims dim = MetalOneDim(size);
  command_encoder->dispatchThreads(dim.num_threads_per_grid,
                                   dim.num_threads_per_group);

  END_COMPUTE_COMMAND
}

void Compact(const MetalArray<scalar_t>& a, MetalArray<scalar_t>* out,
             std::vector<uint32_t> shape, std::vector<uint32_t> strides,
             size_t offset) {
  BEGIN_COMPUTE_COMMAND("CompactKernel")
  command_encoder->setBuffer(a.buffer, 0, 0);
  command_encoder->setBuffer(out->buffer, 0, 1);

  MetalArray<uint32_t> shape_arr = VecToMetal<uint32_t>(shape);
  command_encoder->setBuffer(shape_arr.buffer, 0, 2);

  MetalArray<uint32_t> strides_arr = VecToMetal<uint32_t>(strides);
  command_encoder->setBuffer(strides_arr.buffer, 0, 3);

  MetalArray<size_t> dim_arr = VecToMetal<size_t>(std::vector<size_t>{shape.size()});
  command_encoder->setBuffer(dim_arr.buffer, 0, 4);

  MetalArray<size_t> offset_arr = VecToMetal<size_t>(std::vector<size_t>{offset});
  command_encoder->setBuffer(offset_arr.buffer, 0, 5);

  MetalDims dim = MetalOneDim(out->size);
  command_encoder->dispatchThreads(dim.num_threads_per_grid, dim.num_threads_per_group);

  END_COMPUTE_COMMAND
}

// ************ begin your implementation below this line ****************

void EwiseSetitem(const MetalArray<scalar_t>& a, MetalArray<scalar_t>* out,
                  std::vector<uint32_t> shape, std::vector<uint32_t> strides,
                  size_t offset) {
  BEGIN_COMPUTE_COMMAND("EwiseSetitemKernel")
   command_encoder->setBuffer(a.buffer, 0, 0);
  command_encoder->setBuffer(out->buffer, 0, 1);

  MetalArray<uint32_t> shape_arr = VecToMetal<uint32_t>(shape);
  command_encoder->setBuffer(shape_arr.buffer, 0, 2);

  MetalArray<uint32_t> strides_arr = VecToMetal<uint32_t>(strides);
  command_encoder->setBuffer(strides_arr.buffer, 0, 3);

  MetalArray<size_t> dim_arr = VecToMetal<size_t>(std::vector<size_t>{shape.size()});
  command_encoder->setBuffer(dim_arr.buffer, 0, 4);

  MetalArray<size_t> offset_arr = VecToMetal<size_t>(std::vector<size_t>{offset});
  command_encoder->setBuffer(offset_arr.buffer, 0, 5);

  MetalDims dim = MetalOneDim(a.size);
  command_encoder->dispatchThreads(dim.num_threads_per_grid, dim.num_threads_per_group); 

  END_COMPUTE_COMMAND
}

void ScalarSetitem(size_t size, scalar_t val, MetalArray<scalar_t>* out,
                   std::vector<uint32_t> shape, std::vector<uint32_t> strides,
                   size_t offset) {
  BEGIN_COMPUTE_COMMAND("ScalarSetitemKernel")

  MetalArray<scalar_t> val_arr = VecToMetal<scalar_t>(std::vector<scalar_t>{val});
  command_encoder->setBuffer(val_arr.buffer, 0, 0);

  command_encoder->setBuffer(out->buffer, 0, 1);

  MetalArray<uint32_t> shape_arr = VecToMetal<uint32_t>(shape);
  command_encoder->setBuffer(shape_arr.buffer, 0, 2);

  MetalArray<uint32_t> strides_arr = VecToMetal<uint32_t>(strides);
  command_encoder->setBuffer(strides_arr.buffer, 0, 3);

  MetalArray<size_t> dim_arr = VecToMetal<size_t>(std::vector<size_t>{shape.size()});
  command_encoder->setBuffer(dim_arr.buffer, 0, 4);

  MetalArray<size_t> offset_arr = VecToMetal<size_t>(std::vector<size_t>{offset});
  command_encoder->setBuffer(offset_arr.buffer, 0, 5);

  MetalDims dim = MetalOneDim(size);
  command_encoder->dispatchThreads(dim.num_threads_per_grid, dim.num_threads_per_group); 


  END_COMPUTE_COMMAND
}

void EwiseAdd(const MetalArray<scalar_t>& a, const MetalArray<scalar_t>& b,
              MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("EwiseAddKernel")

  command_encoder->setBuffer(a.buffer, 0, 0);
  command_encoder->setBuffer(b.buffer, 0, 1);
  command_encoder->setBuffer(out->buffer, 0, 2);
  MetalDims dim = MetalOneDim(a.size);
  command_encoder->dispatchThreads(dim.num_threads_per_grid,
                                   dim.num_threads_per_group);

  END_COMPUTE_COMMAND
}

void ScalarAdd(const MetalArray<scalar_t>& a, scalar_t val,
               MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("ScalarAddKernel")

  command_encoder->setBuffer(a.buffer, 0, 0);
  MetalArray<scalar_t> val_arr =
      VecToMetal<scalar_t>(std::vector<scalar_t>{val});
  command_encoder->setBuffer(val_arr.buffer, 0, 1);
  command_encoder->setBuffer(out->buffer, 0, 2);
  MetalDims dim = MetalOneDim(a.size);
  command_encoder->dispatchThreads(dim.num_threads_per_grid,
                                   dim.num_threads_per_group);

  END_COMPUTE_COMMAND
}

void EwiseMul(const MetalArray<scalar_t>& a, const MetalArray<scalar_t>& b,
              MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("EwiseMulKernel")


  END_COMPUTE_COMMAND
}

void ScalarMul(const MetalArray<scalar_t>& a, scalar_t val,
               MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("ScalarMulKernel")


  END_COMPUTE_COMMAND
}

void EwiseDiv(const MetalArray<scalar_t>& a, const MetalArray<scalar_t>& b,
              MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("EwiseDivKernel")

  END_COMPUTE_COMMAND
}

void ScalarDiv(const MetalArray<scalar_t>& a, scalar_t val,
               MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("ScalarDivKernel")


  END_COMPUTE_COMMAND
}

void ScalarPower(const MetalArray<scalar_t>& a, scalar_t val,
                 MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("ScalarPowerKernel")


  END_COMPUTE_COMMAND
}

void EwiseMaximum(const MetalArray<scalar_t>& a, const MetalArray<scalar_t>& b,
                  MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("EwiseMaximumKernel")


  END_COMPUTE_COMMAND
}

void ScalarMaximum(const MetalArray<scalar_t>& a, scalar_t val,
                   MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("ScalarMaximumKernel")


  END_COMPUTE_COMMAND
}

void EwiseEq(const MetalArray<scalar_t>& a, const MetalArray<scalar_t>& b,
             MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("EwiseEqKernel")


  END_COMPUTE_COMMAND
}

void ScalarEq(const MetalArray<scalar_t>& a, scalar_t val,
              MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("ScalarEqKernel")


  END_COMPUTE_COMMAND
}

void EwiseGe(const MetalArray<scalar_t>& a, const MetalArray<scalar_t>& b,
             MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("EwiseGeKernel")


  END_COMPUTE_COMMAND
}

void ScalarGe(const MetalArray<scalar_t>& a, scalar_t val,
              MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("ScalarGeKernel")


  END_COMPUTE_COMMAND
}

void EwiseLog(const MetalArray<scalar_t>& a, MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("EwiseLogKernel")


  END_COMPUTE_COMMAND
}

void EwiseExp(const MetalArray<scalar_t>& a, MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("EwiseExpKernel")


  END_COMPUTE_COMMAND
}

void EwiseTanh(const MetalArray<scalar_t>& a, MetalArray<scalar_t>* out) {
  BEGIN_COMPUTE_COMMAND("EwiseTanhKernel")


  END_COMPUTE_COMMAND
}

void Matmul(const MetalArray<scalar_t>& a, const MetalArray<scalar_t>& b,
            MetalArray<scalar_t>* out, uint32_t M, uint32_t N, uint32_t P) {
  BEGIN_COMPUTE_COMMAND("MatmulKernel")


  END_COMPUTE_COMMAND
}

void ReduceMax(const MetalArray<scalar_t>& a, MetalArray<scalar_t>* out,
               size_t reduce_size) {
  BEGIN_COMPUTE_COMMAND("ReduceMaxKernel")


  END_COMPUTE_COMMAND
}

void ReduceSum(const MetalArray<scalar_t>& a, MetalArray<scalar_t>* out,
               size_t reduce_size) {
  BEGIN_COMPUTE_COMMAND("ReduceSumKernel")


  END_COMPUTE_COMMAND
}

} // namespace metal
} // namespace needle

PYBIND11_MODULE(ndarray_backend_metal, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace needle::metal;

  m.attr("__device_name__") = "metal";
  m.attr("__tile_size__") = kTileSize;

  py::class_<MetalArray<scalar_t>>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &MetalArray<scalar_t>::ptr_as_int)
      .def_readonly("size", &MetalArray<scalar_t>::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const MetalArray<scalar_t>& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(),
                   numpy_strides.begin(),
                   [](size_t& c) { return c * kElemSize; });

    // copy memory to host
    // Since the MetalArray is marked as shared(between cpu and gpu), directly
    // memcpy is OK.
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * kElemSize);
    std::memcpy(host_ptr, a.ptr, a.size * kElemSize);

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset,
                                 deallocate_buffer);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, MetalArray<scalar_t>* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * kElemSize);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  // m.def("ewise_mul", EwiseMul);
  // m.def("scalar_mul", ScalarMul);
  // m.def("ewise_div", EwiseDiv);
  // m.def("scalar_div", ScalarDiv);
  // m.def("scalar_power", ScalarPower);

  // m.def("ewise_maximum", EwiseMaximum);
  // m.def("scalar_maximum", ScalarMaximum);
  // m.def("ewise_eq", EwiseEq);
  // m.def("scalar_eq", ScalarEq);
  // m.def("ewise_ge", EwiseGe);
  // m.def("scalar_ge", ScalarGe);

  // m.def("ewise_log", EwiseLog);
  // m.def("ewise_exp", EwiseExp);
  // m.def("ewise_tanh", EwiseTanh);

  // m.def("matmul", Matmul);

  // m.def("reduce_max", ReduceMax);
  // m.def("reduce_sum", ReduceSum);
}
