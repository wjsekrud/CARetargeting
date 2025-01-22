#include <cuda_runtime.h>
#include <cfloat>

extern "C" {

    // Atomic operations for float
    __device__ float atomicMinFloat(float* addr, float value) {
        float old;
        old = (value >= 0) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
            __int_as_float(atomicMax((int*)addr, __float_as_int(value)));
        return old;
    }

    __device__ float atomicMaxFloat(float* addr, float value) {
        float old;
        old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
            __int_as_float(atomicMin((int*)addr, __float_as_int(value)));
        return old;
    }

    struct float4x4 {
        float m00, m01, m02, m03;
        float m10, m11, m12, m13;
        float m20, m21, m22, m23;
        float m30, m31, m32, m33;
    };

    // AABB 구조체 
    struct AABB {
        float3 min;
        float3 max;
    };

    // BVH 노드 구조체
    struct BVHNode {
        AABB bounds;
        int leftFirst;  // 왼쪽 자식 인덱스 (leaf면 첫번째 삼각형/vertex 인덱스)
        int triCount;   // 음수면 내부노드, 양수면 leaf
    };

    // CUDA 커널: vertex들의 AABB 계산
    __global__ void computeVertexAABBs(
        float3* positions,
        AABB* aabbs,
        int numVertices,
        float margin
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numVertices) return;
        
        float3 pos = positions[idx];
        AABB aabb;
        aabb.min = make_float3(pos.x - margin, pos.y - margin, pos.z - margin);
        aabb.max = make_float3(pos.x + margin, pos.y + margin, pos.z + margin);
        aabbs[idx] = aabb;
    }


    __device__ float surfaceArea(const AABB& aabb) {
        float dx = aabb.max.x - aabb.min.x;
        float dy = aabb.max.y - aabb.min.y;
        float dz = aabb.max.z - aabb.min.z;
        return 2.0f * (dx * dy + dy * dz + dz * dx);
    }

    __device__ bool checkOverlap(const AABB& a, const AABB& b) {
        return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
            (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
            (a.min.z <= b.max.z && a.max.z >= b.min.z);
    }

    // AABB 병합
    __device__ AABB mergeAABB(const AABB& a, const AABB& b) {
        AABB result;
        result.min.x = fminf(a.min.x, b.min.x);
        result.min.y = fminf(a.min.y, b.min.y);
        result.min.z = fminf(a.min.z, b.min.z);
        result.max.x = fmaxf(a.max.x, b.max.x);
        result.max.y = fmaxf(a.max.y, b.max.y);
        result.max.z = fmaxf(a.max.z, b.max.z);
        return result;
    }


    // Contact detection 커널
    __device__ float cosineSimilarity(float3 v1, float3 v2) {
        float dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        float len1 = sqrtf(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
        float len2 = sqrtf(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
        if (len1 < 1e-6f || len2 < 1e-6f) return 0.0f;
        return dot / (len1 * len2);
    }

    // Binning을 위한 bin 구조체
    struct Bin {
        AABB bounds;
        int count;
    };

    // BVH 구축 커널
    __global__ void buildBVHKernel(
        float3* positions,
        int* indices,
        BVHNode* nodes,
        int numVertices,
        float margin
    ) {
        __shared__ Bin bins[32];  // 32개의 bin 사용
        
        // 루트 노드 초기화
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            nodes[0].triCount = numVertices;
            nodes[0].leftFirst = 0;
        }
        
        __syncthreads();
        
        // BVH 구축 반복
        for (int nodeIdx = 0; nodeIdx < numVertices - 1; nodeIdx++) {
            if (nodes[nodeIdx].triCount <= 0) continue;  // 내부노드 스킵
            
            // Binning axis 선택
            float3 extent = make_float3(
                nodes[nodeIdx].bounds.max.x - nodes[nodeIdx].bounds.min.x,
                nodes[nodeIdx].bounds.max.y - nodes[nodeIdx].bounds.min.y,
                nodes[nodeIdx].bounds.max.z - nodes[nodeIdx].bounds.min.z
            );
            
            int axis = 0;
            if (extent.y > extent.x) axis = 1;
            if (extent.z > extent.x) axis = 2;
            
            // Binning
            for (int i = threadIdx.x; i < 32; i += blockDim.x) {
                bins[i].count = 0;
                bins[i].bounds.min = make_float3(FLT_MAX,FLT_MAX,FLT_MAX);
                bins[i].bounds.max = make_float3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
            }
            
            __syncthreads();
            
            // Primitive들을 bin에 할당
            for (int i = nodes[nodeIdx].leftFirst; 
                i < nodes[nodeIdx].leftFirst + nodes[nodeIdx].triCount; 
                i++) {
                float3 centroid = positions[indices[i]];
                float normalized = 0;
                int binIdx = 0;
                switch (axis) {
                    case 0:
                        normalized = (centroid.x - nodes[nodeIdx].bounds.min.x) /
                                (nodes[nodeIdx].bounds.max.x - nodes[nodeIdx].bounds.min.x);
                        binIdx = min(31, (int)(normalized * 32));
                    
                    case 1:
                        normalized = (centroid.y - nodes[nodeIdx].bounds.min.y) /
                                (nodes[nodeIdx].bounds.max.y - nodes[nodeIdx].bounds.min.y);
                        binIdx = min(31, (int)(normalized * 32));
                    
                    case 2:
                        normalized = (centroid.z - nodes[nodeIdx].bounds.min.z) /
                                (nodes[nodeIdx].bounds.max.z - nodes[nodeIdx].bounds.min.z);
                        binIdx = min(31, (int)(normalized * 32));
                }
                
                atomicAdd(&bins[binIdx].count, 1);
                
                // AABB 업데이트
                AABB primBounds;
                primBounds.min = make_float3(
                    centroid.x - margin,
                    centroid.y - margin,
                    centroid.z - margin
                );
                primBounds.max = make_float3(
                    centroid.x + margin,
                    centroid.y + margin,
                    centroid.z + margin
                );
                
                atomicMinFloat(&bins[binIdx].bounds.min.x, primBounds.min.x);
                atomicMinFloat(&bins[binIdx].bounds.min.y, primBounds.min.y);
                atomicMinFloat(&bins[binIdx].bounds.min.z, primBounds.min.z);
                atomicMaxFloat(&bins[binIdx].bounds.max.x, primBounds.max.x);
                atomicMaxFloat(&bins[binIdx].bounds.max.y, primBounds.max.y);
                atomicMaxFloat(&bins[binIdx].bounds.max.z, primBounds.max.z);
            }
            
            __syncthreads();
            
            // 최적의 split 위치 찾기
            if (threadIdx.x == 0) {
                float minCost = FLT_MAX;
                int bestSplit = -1;
                
                AABB leftBox, rightBox;
                int leftCount = 0, rightCount = nodes[nodeIdx].triCount;
                
                // 각 split 위치에서의 비용 계산
                for (int i = 0; i < 31; i++) {
                    leftBox = bins[0].bounds;
                    leftCount = bins[0].count;
                    
                    for (int j = 1; j <= i; j++) {
                        if (bins[j].count > 0) {
                            leftBox = mergeAABB(leftBox, bins[j].bounds);
                            leftCount += bins[j].count;
                        }
                    }
                    
                    rightBox = bins[31].bounds;
                    rightCount = bins[31].count;
                    
                    for (int j = 30; j > i; j--) {
                        if (bins[j].count > 0) {
                            rightBox = mergeAABB(rightBox, bins[j].bounds);
                            rightCount += bins[j].count;
                        }
                    }
                    
                    float cost = leftCount * surfaceArea(leftBox) + 
                                rightCount * surfaceArea(rightBox);
                    
                    if (cost < minCost && leftCount > 0 && rightCount > 0) {
                        minCost = cost;
                        bestSplit = i;
                    }
                }
                
                if (bestSplit != -1) {
                    // 새로운 자식 노드 생성
                    int leftNodeIdx = nodeIdx * 2 + 1;
                    int rightNodeIdx = nodeIdx * 2 + 2;
                    
                    nodes[leftNodeIdx].leftFirst = nodes[nodeIdx].leftFirst;
                    nodes[rightNodeIdx].leftFirst = nodes[nodeIdx].leftFirst + leftCount;
                    
                    nodes[leftNodeIdx].triCount = leftCount;
                    nodes[rightNodeIdx].triCount = rightCount;
                    
                    // 부모 노드를 내부 노드로 표시
                    nodes[nodeIdx].triCount = -1;
                    nodes[nodeIdx].leftFirst = leftNodeIdx;
                }
            }
            
            __syncthreads();
        }
    }

    __global__ void detectContactsKernel(
        BVHNode* handNodes,
        BVHNode* otherNodes,
        float3* handPositions,
        float3* handVelocities,
        float3* otherPositions,
        float3* otherVelocities,
        int* contactPairs,
        float* distances,
        int* contactCount,
        int maxContacts
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= maxContacts) return;
        
        __shared__ int stack[64];
        __shared__ int stackPtr;
        
        if (threadIdx.x == 0) stackPtr = 0;
        
        __syncthreads();
        
        stack[stackPtr++] = 0;
        
        while (stackPtr > 0) {
            int nodeIdx = stack[--stackPtr];
            BVHNode* hand = &handNodes[nodeIdx];
            
            for (int otherIdx = 0; otherIdx < otherNodes[0].triCount; otherIdx++) {
                BVHNode* other = &otherNodes[otherIdx];
                
                if (!checkOverlap(hand->bounds, other->bounds)) continue;
                
                if (hand->triCount < 0) {
                    stack[stackPtr++] = hand->leftFirst;
                    stack[stackPtr++] = hand->leftFirst + 1;
                } else {
                    for (int i = hand->leftFirst; i < hand->leftFirst + hand->triCount; i++) {
                        for (int j = other->leftFirst; j < other->leftFirst + other->triCount; j++) {
                            float3 diff;
                            diff.x = handPositions[i].x - otherPositions[j].x;
                            diff.y = handPositions[i].y - otherPositions[j].y;
                            diff.z = handPositions[i].z - otherPositions[j].z;
                            
                            float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
                            float similarity = cosineSimilarity(handVelocities[i], otherVelocities[j]);
                            
                            if (dist < 2.0f || similarity > 0.9f) {
                                int contactIdx = atomicAdd(contactCount, 1);
                                if (contactIdx < maxContacts) {
                                    contactPairs[contactIdx * 2] = i;
                                    contactPairs[contactIdx * 2 + 1] = j;
                                    distances[contactIdx] = dist;
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    __global__ void computeSkinningKernel(
        float* restPositions,
        float* boneMatrices,      // flattened 16*N matrices
        int* boneIndices,
        float* boneWeights,
        float* skinnedPositions,
        int numVertices
    ) {
        //*/
        int idx = blockIdx.x;
        if (idx >= numVertices) return;
        
        float3 pos;// = restPositions[idx];
        pos.x = restPositions[idx + 0];
        pos.y = restPositions[idx + 1];
        pos.z = restPositions[idx + 2];

        int index = boneIndices[idx];
        float weight = boneWeights[idx];
        
        float3 finalPos = make_float3(0.0f, 0.0f, 0.0f);
        
        // 16개 요소를 가진 행렬에 접근
        int matrixOffset = index * 16;
        float3 transformedPos;
        transformedPos.x = boneMatrices[matrixOffset + 0] * pos.x + 
                        boneMatrices[matrixOffset + 1] * pos.y + 
                        boneMatrices[matrixOffset + 2] * pos.z + 
                        boneMatrices[matrixOffset + 3];
        transformedPos.y = boneMatrices[matrixOffset + 4] * pos.x + 
                        boneMatrices[matrixOffset + 5] * pos.y + 
                        boneMatrices[matrixOffset + 6] * pos.z + 
                        boneMatrices[matrixOffset + 7];
        transformedPos.z = boneMatrices[matrixOffset + 8] * pos.x + 
                        boneMatrices[matrixOffset + 9] * pos.y + 
                        boneMatrices[matrixOffset + 10] * pos.z + 
                        boneMatrices[matrixOffset + 11];
        
        finalPos.x += weight * transformedPos.x;
        finalPos.y += weight * transformedPos.y;
        finalPos.z += weight * transformedPos.z;

        skinnedPositions[idx + 0] = finalPos.x;
        skinnedPositions[idx + 1] = finalPos.y;
        skinnedPositions[idx + 2] = finalPos.z;
        //*/
        
    }


}