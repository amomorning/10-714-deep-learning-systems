#include <metal_math>

kernel void FillKernel(device float* out [[buffer(0)]],
                       device const float* val [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
  out[index] = val[0];
}

kernel void CompactKernel(device const float* a [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          device const uint* shape [[buffer(2)]],
                          device const uint* strides [[buffer(3)]],
                          device const size_t* dim [[buffer(4)]],
                          device const size_t* offset [[buffer(5)]],
                          uint index [[thread_position_in_grid]]) {
  uint i = index;
  uint pos = offset[0];
  for (int j = dim[0] - 1; j >= 0; j --) {
    pos += i % shape[j] * strides[j];
    i /= shape[j];
  }
  out[index] = a[pos];
}

kernel void EwiseSetitemKernel(device const float* a [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               device const uint* shape [[buffer(2)]],
                               device const uint* strides [[buffer(3)]],
                               device const size_t* dim [[buffer(4)]],
                               device const size_t* offset [[buffer(5)]],
                               uint index [[thread_position_in_grid]]) {
  uint i = index;
  uint pos = offset[0];
  for (int j = dim[0] - 1; j >= 0; j --) {
    pos += i % shape[j] * strides[j];
    i /= shape[j];
  }
  out[pos] = a[index];
}

kernel void ScalarSetitemKernel(device float* val [[buffer(0)]],
                                device float* out [[buffer(1)]],
                                device const uint* shape [[buffer(2)]],
                                device const uint* strides [[buffer(3)]],
                                device const size_t* dim [[buffer(4)]],
                                device const size_t* offset [[buffer(5)]],
                                uint index [[thread_position_in_grid]]) {
  uint i = index;
  uint pos = offset[0];
  for (int j = dim[0] - 1; j >= 0; j --) {
    pos += i % shape[j] * strides[j];
    i /= shape[j];
  }
  out[pos] = val[0];

}

kernel void EwiseAddKernel(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
  out[index] = a[index] + b[index];
}

kernel void ScalarAddKernel(device const float* a [[buffer(0)]],
                            device const float* val [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
  out[index] = a[index] + val[0];
}

kernel void EwiseMulKernel(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
  out[index] = a[index] * b[index];
}

kernel void ScalarMulKernel(device const float* a [[buffer(0)]],
                            device const float* val [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
  out[index] = a[index] * val[0];
}

kernel void EwiseDivKernel(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
  out[index] = a[index] / b[index];
}

kernel void ScalarDivKernel(device const float* a [[buffer(0)]],
                            device const float* val [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
  out[index] = a[index] / val[0];
}

kernel void ScalarPowerKernel(device const float* a [[buffer(0)]],
                              device const float* val [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
  out[index] = metal::pow(a[index], val[0]);
}

kernel void EwiseMaximumKernel(device const float* a [[buffer(0)]],
                               device const float* b [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
  out[index] = metal::max(a[index], b[index]);
}

kernel void ScalarMaximumKernel(device const float* a [[buffer(0)]],
                                device const float* val [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                uint index [[thread_position_in_grid]]) {
  out[index] = metal::max(a[index], val[0]);
}

kernel void EwiseEqKernel(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
  out[index] = a[index] == b[index];
}

kernel void ScalarEqKernel(device const float* a [[buffer(0)]],
                           device const float* val [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
  out[index] = a[index] == val[0];
}

kernel void EwiseGeKernel(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
  out[index] = a[index] >= b[index];
}

kernel void ScalarGeKernel(device const float* a [[buffer(0)]],
                           device const float* val [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
  out[index] = a[index] >= val[0];
}

kernel void EwiseLogKernel(device const float* a [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           uint index [[thread_position_in_grid]]) {
  out[index] = metal::log(a[index]);
}

kernel void EwiseExpKernel(device const float* a [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           uint index [[thread_position_in_grid]]) {
  out[index] = metal::exp(a[index]);
}

kernel void EwiseTanhKernel(device const float* a [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            uint index [[thread_position_in_grid]]) {
  float tmp = metal::tanh(a[index]);
  out[index] = metal::isnan(tmp) ? 1.0: tmp;
}

kernel void ReduceMaxKernel(device const float* a [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            device const size_t* reduce_size [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
  uint start = index * reduce_size[0];
  uint end = (index + 1) * reduce_size[0];
  float val = a[start];
  for (uint i = start; i < end; i++) {
    val = metal::max(val, a[i]);
  }
  out[index] = val;
}

kernel void ReduceSumKernel(device const float* a [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            device const size_t* reduce_size [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
                            
  uint start = index * reduce_size[0];
  uint end = (index + 1) * reduce_size[0];
  float val = 0.0;
  for (uint i = start; i < end; i++) {
    val += a[i];
  }
  out[index] = val;
}

kernel void MatmulKernel(device const float* a [[buffer(0)]],
                         device const float* b [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         device const uint* dim [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
  size_t i = index / dim[2];
  size_t j = index % dim[2];
  float v = 0.0;
  for (size_t k = 0; k < dim[1]; ++ k) {
    v += a[i * dim[1] + k] * b[k * dim[2] + j];
  }
  out[index] = v;
}
