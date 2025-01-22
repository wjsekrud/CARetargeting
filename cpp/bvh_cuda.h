#ifndef BVH_CUDA_H
#define BVH_CUDA_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA API error checking
#define CHECK_CUDA(call)                                                  \
    do {                                                                 \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__,        \
                   cudaGetErrorString(err));                            \
            return err;                                                 \
        }                                                               \
    } while (0)

// Exported functions
cudaError_t buildBVH(
    const float* positions,       // [numPrimitives * 3]
    const int numPrimitives,
    float** deviceNodes,         // Output: BVH nodes on device
    int** devicePrimitiveIndices // Output: Reordered primitive indices
);

cudaError_t detectContacts(
    const float* handPositions,   // [handCount * 3]
    const float* otherPositions,  // [otherCount * 3]
    const float* handVelocities,  // [handCount * 3]
    const float* otherVelocities, // [otherCount * 3]
    const void* handBVH,         // BVH nodes from buildBVH
    const void* otherBVH,        // BVH nodes from buildBVH
    int* contactPairs,           // Output: [maxContacts * 2 + 1]
    float* distances,            // Output: [maxContacts]
    const int handCount,
    const int otherCount,
    const float distanceThreshold,
    const float velocityThreshold,
    const int maxContacts
);

// Memory management helpers
cudaError_t freeDeviceMemory(void* ptr);

#ifdef __cplusplus
}
#endif

#endif 