// ===== topk_warp_kernel.cu =====

#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

__device__ inline void insert_desc_smallk(float* vals, int* ids, int K, float v, int id) {
    if (v >= vals[0]) return; // not good enough
    vals[0] = v;
    ids[0]  = id;
    // bubble down
    for (int i = 1; i < K; i++) {
        if (vals[i] > vals[i-1]) {
            float tv = vals[i];
            int ti   = ids[i];
            vals[i]   = vals[i-1];
            ids[i]    = ids[i-1];
            vals[i-1] = tv;
            ids[i-1]  = ti;
        } else {
            break;
        }
    }
}

extern "C"
__global__ void topk_warp_kernel(const float* __restrict__ dists,
                                 int nq,
                                 int n_cluster,
                                 int K,
                                 int* __restrict__ out_topk)
{
    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warpIdInBlock = tid / WARP_SIZE;
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    int globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock;
    if (globalWarpId >= nq) return;

    extern __shared__ unsigned char shmbuf[];
    float* shm_vals = reinterpret_cast<float*>(shmbuf);
    int*   shm_ids  = reinterpret_cast<int*>(shmbuf + sizeof(float) * warpsPerBlock * WARP_SIZE * K);

    const int MAXK = 64; // adjust if needed
    if (K > MAXK) return;

    float local_vals[MAXK];
    int   local_ids[MAXK];

    for (int i = 0; i < K; i++) {
        local_vals[i] = 1e30f;
        local_ids[i] = -1;
    }

    const float* row = dists + (size_t)globalWarpId * n_cluster;

    for (int idx = lane; idx < n_cluster; idx += WARP_SIZE) {
        float v = row[idx];
        if (v < local_vals[0]) {
            insert_desc_smallk(local_vals, local_ids, K, v, idx);
        }
    }

    int slot = (warpIdInBlock * WARP_SIZE + lane) * K;
    float* vals_slot = shm_vals + slot;
    int*   ids_slot  = shm_ids  + slot;

    for (int i = 0; i < K; i++) {
        vals_slot[i] = local_vals[i];
        ids_slot[i]  = local_ids[i];
    }

    __syncwarp();

    if (lane == 0) {
        float agg_vals[MAXK];
        int   agg_ids[MAXK];
        for (int i = 0; i < K; i++) {
            agg_vals[i] = 1e30f;
            agg_ids[i]  = -1;
        }

        int base = warpIdInBlock * WARP_SIZE * K;
        for (int t = 0; t < WARP_SIZE * K; t++) {
            float v = shm_vals[base + t];
            int   id = shm_ids[base + t];
            if (id >= 0 && v < agg_vals[0]) {
                insert_desc_smallk(agg_vals, agg_ids, K, v, id);
            }
        }

        int out_base = globalWarpId * K;
        for (int k = 0; k < K; k++) {
            out_topk[out_base + k] = agg_ids[K - 1 - k];
        }
    }
}
