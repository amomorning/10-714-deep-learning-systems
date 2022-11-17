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

}

kernel void EwiseSetitemKernel(device const float* a [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               device const uint* shape [[buffer(2)]],
                               device const uint* strides [[buffer(3)]],
                               device const size_t* dim [[buffer(4)]],
                               device const size_t* offset [[buffer(5)]],
                               uint index [[thread_position_in_grid]]) {
}

kernel void ScalarSetitemKernel(device float* val [[buffer(0)]],
                                device float* out [[buffer(1)]],
                                device const uint* shape [[buffer(2)]],
                                device const uint* strides [[buffer(3)]],
                                device const size_t* dim [[buffer(4)]],
                                device const size_t* offset [[buffer(5)]],
                                uint index [[thread_position_in_grid]]) {
}

kernel void EwiseAddKernel(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
}

kernel void ScalarAddKernel(device const float* a [[buffer(0)]],
                            device const float* val [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
}

kernel void EwiseMulKernel(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
}

kernel void ScalarMulKernel(device const float* a [[buffer(0)]],
                            device const float* val [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
}

kernel void EwiseDivKernel(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
}

kernel void ScalarDivKernel(device const float* a [[buffer(0)]],
                            device const float* val [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
}

kernel void ScalarPowerKernel(device const float* a [[buffer(0)]],
                              device const float* val [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
}

kernel void EwiseMaximumKernel(device const float* a [[buffer(0)]],
                               device const float* b [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
}

kernel void ScalarMaximumKernel(device const float* a [[buffer(0)]],
                                device const float* val [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                uint index [[thread_position_in_grid]]) {
}

kernel void EwiseEqKernel(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
}

kernel void ScalarEqKernel(device const float* a [[buffer(0)]],
                           device const float* val [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
}

kernel void EwiseGeKernel(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
}

kernel void ScalarGeKernel(device const float* a [[buffer(0)]],
                           device const float* val [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
}

kernel void EwiseLogKernel(device const float* a [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           uint index [[thread_position_in_grid]]) {
}

kernel void EwiseExpKernel(device const float* a [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           uint index [[thread_position_in_grid]]) {
}

kernel void EwiseTanhKernel(device const float* a [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            uint index [[thread_position_in_grid]]) {
}

kernel void ReduceMaxKernel(device const float* a [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            device const size_t* reduce_size [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
}

kernel void ReduceSumKernel(device const float* a [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            device const size_t* reduce_size [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
}

kernel void MatmulKernel(device const float* a [[buffer(0)]],
                         device const float* b [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         device const uint* dim [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
}
