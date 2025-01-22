# CUDA kernel definitions with matrix structure
cuda_transform_kernel = """
// 4x4 matrix structure definition
struct float4x4 {
    float m[16];  // Column-major order
};

__global__ void transform_vertices(float3 *positions, float4x4 *bone_matrices, 
                                 int *bone_indices, float *weights, 
                                 float3 *output_positions, int num_vertices, 
                                 int max_weights) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;
    
    float3 final_pos = positions[idx];
    float3 transformed_pos = make_float3(0.0f, 0.0f, 0.0f);
    
    for (int w = 0; w < max_weights; w++) {
        int bone_idx = bone_indices[idx * max_weights + w];
        float weight = weights[idx * max_weights + w];
        if (weight == 0.0f) continue;
        
        float4x4 bone_matrix = bone_matrices[bone_idx];
        float3 pos = final_pos;
        float3 temp;
        
        // Matrix multiplication (column-major order)
        temp.x = bone_matrix.m[0] * pos.x + bone_matrix.m[4] * pos.y + 
                 bone_matrix.m[8] * pos.z + bone_matrix.m[12];
        temp.y = bone_matrix.m[1] * pos.x + bone_matrix.m[5] * pos.y + 
                 bone_matrix.m[9] * pos.z + bone_matrix.m[13];
        temp.z = bone_matrix.m[2] * pos.x + bone_matrix.m[6] * pos.y + 
                 bone_matrix.m[10] * pos.z + bone_matrix.m[14];
        
        transformed_pos.x += temp.x * weight;
        transformed_pos.y += temp.y * weight;
        transformed_pos.z += temp.z * weight;
    }
    
    output_positions[idx] = transformed_pos;
}
"""

# Contact detection kernel remains unchanged
cuda_contact_kernel = """
__global__ void detect_contacts(float3 *hand_positions, float3 *other_positions,
                              float3 *hand_velocities, float3 *other_velocities,
                              int *contact_pairs, float *distances,
                              int hand_count, int other_count,
                              float distance_threshold, float velocity_threshold) {
    int hand_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int other_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (hand_idx >= hand_count || other_idx >= other_count) return;
    
    float3 hand_pos = hand_positions[hand_idx];
    float3 other_pos = other_positions[other_idx];
    
    // Calculate distance
    float dx = hand_pos.x - other_pos.x;
    float dy = hand_pos.y - other_pos.y;
    float dz = hand_pos.z - other_pos.z;
    float distance = sqrt(dx*dx + dy*dy + dz*dz);

    if (distance < distance_threshold) {
        int pair_idx = atomicAdd(contact_pairs, 1);
        distances[pair_idx] = distance;
        contact_pairs[pair_idx * 2 + 1] = hand_idx;
        contact_pairs[pair_idx * 2 + 2] = other_idx;
        return;
    }

    // Check velocity similarity
    float3 hand_vel = hand_velocities[hand_idx];
    float3 other_vel = other_velocities[other_idx];
    
    float dot_product = hand_vel.x * other_vel.x + 
                        hand_vel.y * other_vel.y + 
                        hand_vel.z * other_vel.z;
    float hand_vel_length = sqrt(hand_vel.x*hand_vel.x + 
                                hand_vel.y*hand_vel.y + 
                                hand_vel.z*hand_vel.z);
    float other_vel_length = sqrt(other_vel.x*other_vel.x + 
                                other_vel.y*other_vel.y + 
                                other_vel.z*other_vel.z);

    if (hand_vel_length > 0.0f && other_vel_length > 0.0f) {
            float cos_sim = dot_product / (hand_vel_length * other_vel_length);
            if (cos_sim > 0.9) {
                int pair_idx = atomicAdd(contact_pairs, 1);
                distances[pair_idx] = distance;
                contact_pairs[pair_idx * 2 + 1] = hand_idx;
                contact_pairs[pair_idx * 2 + 2] = other_idx;
            }
        }
}
"""