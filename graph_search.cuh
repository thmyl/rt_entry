#pragma once
#include "warpselect/structure_on_device.cuh"
#include <cuda_runtime.h>
// #include <nvToolsExt.h>
#include "head.h"
#include "cache/page_cache.h"
#define SIZE 512
#define num_hash 3
#define max_hash_iteration 4

#ifdef COMPUTE_COUNT
// 全局设备变量用于统计计算次数
// 使用 unsigned long long 类型，因为 compute_dim 可能很大
__device__ unsigned long long compute_dim = 0;
__device__ unsigned long long compute_dis = 0;
#endif

#ifdef USE_CACHE
#ifdef ENABLE_CONSTANT_CLUSTER_MAP
    __constant__ int d_cluster_to_page[MAX_CLUSTER_TO_PAGE];
    int h_cluster_to_page_size;
    __constant__ int d_cluster_to_page_size;
#endif

__device__ inline int fetch_cluster_cache_page(int global_page,
                                               const int* cluster_to_page) {
    #ifdef ENABLE_CONSTANT_CLUSTER_MAP
        return d_cluster_to_page[global_page];
    #else
        return cluster_to_page[global_page];
    #endif

    // 随机返回一个cache page（使用设备端随机数生成）
    // uint seed = static_cast<uint>(global_page) * 1103515245U + 12345U;
    // seed = seed * 1103515245U + 12345U;
    // const int MAX_CACHE_PAGES = 300;
    // return seed % MAX_CACHE_PAGES;
}
#endif



__device__
void set_bit(uint32_t x,uint32_t* data){
    unsigned int mask = 1U << (x / SIZE);
    atomicOr(&data[x % SIZE], mask);
}

__device__
int hash_(int h,uint32_t x,uint32_t* random_number){ 
    x ^= x >> 16;
    x *= random_number[h << 1];
    x ^= x >> 13;
    x *= random_number[(h << 1) + 1];
    x ^= x >> 16;
    //return x & 31;
    return x % ((SIZE << 5));
}

__device__
bool test_bit(int offset,uint32_t x,uint32_t* data){
    //return ((data[offset] >> x) & 1);
    return ((data[x % SIZE] >> (x / SIZE)) & 1);
}

__device__
void add(uint32_t x,uint32_t* random_number,uint32_t* data){
    for(int i = 0;i < num_hash;++i)
        set_bit(hash_(i,x,random_number),data);
}

__device__
bool test(uint32_t x,uint32_t* random_number,uint32_t* data){
    int offset = 0; //get_offset(x,random_number);
    bool ok = true;
    for(int i = 0;i < num_hash;++i)
        ok &= test_bit(offset,hash_(i,x,random_number),data);
    return ok;
}

__device__
int _abs(int x){
    return x < 0 ? -x : x;
}

__device__
int random_entry(int l,int r, uint seed){
  seed = seed * 1103515245 + 12345;
  return l + (seed) % (r - l);
}

#ifdef USE_CACHE
__device__ inline bool is_cluster_in_top(const int* __restrict__ top_clusters,
                                         int cluster_top_t,
                                         int cluster_id) {
    if (!top_clusters || cluster_top_t <= 0 || cluster_id < 0) {
        return false;
    }
    for (int i = 0; i < cluster_top_t; ++i) {
        if (top_clusters[i] == cluster_id) {
            return true;
        }
    }
    return false;
}

struct return_type{
    bool in_cache;
    const float* page_ptr;
};

__device__ inline return_type whether_cache(int point_id,
                                        const PointInfo* point_infos,
                                        const int* cluster_to_page,
                                        const float* cache_data,
                                        int page_size) {
    return_type ret;
    ret.in_cache = false;
    ret.page_ptr = nullptr;

    const PointInfo* info_ptr = &point_infos[point_id];
    int global_page = __ldg(&info_ptr->global_page_id);
    if(global_page < 0) return ret;
    int cache_page = __ldg(&cluster_to_page[global_page]);
    // return ret;

    if(cache_page != -1){
        int offset = __ldg(&info_ptr->offset);
        unsigned long long page_idx = static_cast<unsigned long long>(cache_page);
        unsigned long long flat_offset = (page_idx * page_size + offset);
        unsigned long long byte_offset = flat_offset * PARTIAL_DIM;
        ret.page_ptr = cache_data + byte_offset;
        // ret.page_ptr = nullptr;
        ret.in_cache = true;
        return ret;
    }
    return ret;
}

#endif

template<typename IdType, typename FloatType, int WARP_SIZE>
__global__ void GraphSearchKernel(float* d_data, const float* query_full, int* d_results, int* d_graph, int* d_candidates, int np,
                      int query_base, int* query_batch_ids,
                      int offset_shift, int n_candidates, int topk, int search_width, int* d_entries,
                      int* d_hits, int max_iter, int ALGO,
                      const PointInfo* point_infos,
                      const int* cluster_to_page,
                      const float* cache_data,
                      int page_size,
                      int dim_partial,
                      int dim_total,
                      // const int* query_top_clusters,
                      int cluster_top_t,
                      const float* linear_w,
                      const float* linear_b,
                      int linear_dim){
  
  int t_id = threadIdx.x;
  int b_id = blockIdx.x;//query batch index
  int q_id = query_batch_ids[query_base + b_id];
  int blockSize = blockDim.x;
  int n_warp = blockSize / WARP_SIZE;
  int lane_id = t_id % WARP_SIZE;
  int warp_id = t_id / WARP_SIZE;
  uint random_entry_seed = b_id*blockSize + t_id;
//   int max_iter = 10;
//   int max_iter = n_candidates / search_width;

  int entries_st;
//   int n_entries = n_candidates;
  int n_entries = search_width * (1 << offset_shift);
  if(ALGO==1 && d_hits)
    entries_st = d_hits[q_id] * n_candidates;

  int* crt_results = d_results + q_id * topk;
  int degree = (1<<offset_shift);
  int n_points_per_batch = (search_width << offset_shift);
  int n_compare = n_candidates;

  if(n_points_per_batch < n_candidates)
    n_compare = n_points_per_batch;

  extern __shared__ KernelPair<float, int> shared_memory_[];
  KernelPair<float, int>* neighbors_array = shared_memory_;

  __shared__ uint data[SIZE];
  uint32_t random_number[10 * 2] = {
    0x924ed183U,0xd854fc0aU,0xecf5e3b7U,
    0x1bead407U,0x28a30449U,0xbfc4d99fU,
    0x715030e2U,0xffcfb45bU,0x6e4ce166U,
    0xeb53c362U,0xa93c4f40U,0xcecde0a4U,
    0x0288592dU,0x362c37bcU,0x9d4824f0U,
    0xfdbdd68bU,0x63258c85U,0x6726905cU,
    0x609500f9U,0x4de48422U
  };

// read d_query
// #region DIM d_query loading
    #if DIM > 0
        float q1 = 0;
        if (lane_id < DIM) {
            q1 = query_full[q_id * FULL_DIM + lane_id];
        }
    #endif
    #if DIM > 32
        float q2 = 0;
        if (lane_id + 32 < DIM) {
            q2 = query_full[q_id * FULL_DIM + lane_id + 32];
        }
    #endif
    #if DIM > 64
        float q3 = 0;
        if (lane_id + 64 < DIM) {
            q3 = query_full[q_id * FULL_DIM + lane_id + 64];
        }
    #endif
    #if DIM > 96
        float q4 = 0;
        if (lane_id + 96 < DIM) {
            q4 = query_full[q_id * FULL_DIM + lane_id + 96];
        }
    #endif
    #if DIM > 128
        float q5 = 0;
        if (lane_id + 128 < DIM) {
            q5 = query_full[q_id * FULL_DIM + lane_id + 128];
        }
    #endif
    #if DIM > 160
        float q6 = 0;
        if (lane_id + 160 < DIM) {
            q6 = query_full[q_id * FULL_DIM + lane_id + 160];
        }
    #endif
    #if DIM > 192
        float q7 = 0;
        if (lane_id + 192 < DIM) {
            q7 = query_full[q_id * FULL_DIM + lane_id + 192];
        }
    #endif
    #if DIM > 224
        float q8 = 0;
        if (lane_id + 224 < DIM) {
            q8 = query_full[q_id * FULL_DIM + lane_id + 224];
        }
    #endif
    #if DIM > 256
        float q9 = 0;
        if (lane_id + 256 < DIM) {
            q9 = query_full[q_id * FULL_DIM + lane_id + 256];
        }
    #endif
    #if DIM > 288
        float q10 = 0;
        if (lane_id + 288 < DIM) {
            q10 = query_full[q_id * FULL_DIM + lane_id + 288];
        }
    #endif
    #if DIM > 320
        float q11 = 0;
        if (lane_id + 320 < DIM) {
            q11 = query_full[q_id * FULL_DIM + lane_id + 320];
        }
    #endif
    #if DIM > 352
        float q12 = 0;
        if (lane_id + 352 < DIM) {
            q12 = query_full[q_id * FULL_DIM + lane_id + 352];
        }
    #endif
    #if DIM > 384
        float q13 = 0;
        if (lane_id + 384 < DIM) {
            q13 = query_full[q_id * FULL_DIM + lane_id + 384];
        }
    #endif
    #if DIM > 416
        float q14 = 0;
        if (lane_id + 416 < DIM) {
            q14 = query_full[q_id * FULL_DIM + lane_id + 416];
        }
    #endif
    #if DIM > 448
        float q15 = 0;
        if (lane_id + 448 < DIM) {
            q15 = query_full[q_id * FULL_DIM + lane_id + 448];
        }
    #endif
    #if DIM > 480
        float q16 = 0;
        if (lane_id + 480 < DIM) {
            q16 = query_full[q_id * FULL_DIM + lane_id + 480];
        }
    #endif
    #if DIM > 512
        float q17 = 0;
        if (lane_id + 512 < DIM) {
            q17 = query_full[q_id * FULL_DIM + lane_id + 512];
        }
    #endif
    #if DIM > 544
        float q18 = 0;
        if (lane_id + 544 < DIM) {
            q18 = query_full[q_id * FULL_DIM + lane_id + 544];
        }
    #endif
    #if DIM > 576
        float q19 = 0;
        if (lane_id + 576 < DIM) {
            q19 = query_full[q_id * FULL_DIM + lane_id + 576];
        }
    #endif
    #if DIM > 608
        float q20 = 0;
        if (lane_id + 608 < DIM) {
            q20 = query_full[q_id * FULL_DIM + lane_id + 608];
        }
    #endif
    #if DIM > 640
        float q21 = 0;
        if (lane_id + 640 < DIM) {
            q21 = query_full[q_id * FULL_DIM + lane_id + 640];
        }
    #endif
    #if DIM > 672
        float q22 = 0;
        if (lane_id + 672 < DIM) {
            q22 = query_full[q_id * FULL_DIM + lane_id + 672];
        }
    #endif
    #if DIM > 704
        float q23 = 0;
        if (lane_id + 704 < DIM) {
            q23 = query_full[q_id * FULL_DIM + lane_id + 704];
        }
    #endif
    #if DIM > 736
        float q24 = 0;
        if (lane_id + 736 < DIM) {
            q24 = query_full[q_id * FULL_DIM + lane_id + 736];
        }
    #endif
    #if DIM > 768
        float q25 = 0;
        if (lane_id + 768 < DIM) {
            q25 = query_full[q_id * FULL_DIM + lane_id + 768];
        }
    #endif
    #if DIM > 800
        float q26 = 0;
        if (lane_id + 800 < DIM) {
            q26 = query_full[q_id * FULL_DIM + lane_id + 800];
        }
    #endif
    #if DIM > 832
        float q27 = 0;
        if (lane_id + 832 < DIM) {
            q27 = query_full[q_id * FULL_DIM + lane_id + 832];
        }
    #endif
    #if DIM > 864
        float q28 = 0;
        if (lane_id + 864 < DIM) {
            q28 = query_full[q_id * FULL_DIM + lane_id + 864];
        }
    #endif
    #if DIM > 896
        float q29 = 0;
        if (lane_id + 896 < DIM) {
            q29 = query_full[q_id * FULL_DIM + lane_id + 896];
        }
    #endif
    #if DIM > 928
        float q30 = 0;
        if (lane_id + 928 < DIM) {
            q30 = query_full[q_id * FULL_DIM + lane_id + 928];
        }
    #endif
// #endregion DIM d_query loading

#ifdef USE_CACHE
//read query
// #region PARTIAL_DIM d_query loading
    #if PARTIAL_DIM > 0
    float q1_tail = 0;
    if (lane_id < PARTIAL_DIM) {
        q1_tail = query_full[q_id * FULL_DIM + DIM + lane_id];
    }
    #endif
    #if PARTIAL_DIM > 32
    float q2_tail = 0;
    if (lane_id + 32 < PARTIAL_DIM) {
        q2_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 32];
    }
    #endif
    #if PARTIAL_DIM > 64
    float q3_tail = 0;
    if (lane_id + 64 < PARTIAL_DIM) {
        q3_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 64];
    }
    #endif
    #if PARTIAL_DIM > 96
    float q4_tail = 0;
    if (lane_id + 96 < PARTIAL_DIM) {
        q4_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 96];
    }
    #endif
    #if PARTIAL_DIM > 128
    float q5_tail = 0;
    if (lane_id + 128 < PARTIAL_DIM) {
        q5_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 128];
    }
    #endif
    #if PARTIAL_DIM > 160
    float q6_tail = 0;
    if (lane_id + 160 < PARTIAL_DIM) {
        q6_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 160];
    }
    #endif
    #if PARTIAL_DIM > 192
    float q7_tail = 0;
    if (lane_id + 192 < PARTIAL_DIM) {
        q7_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 192];
    }
    #endif
    #if PARTIAL_DIM > 224
    float q8_tail = 0;
    if (lane_id + 224 < PARTIAL_DIM) {
        q8_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 224];
    }
    #endif
    #if PARTIAL_DIM > 256
    float q9_tail = 0;
    if (lane_id + 256 < PARTIAL_DIM) {
        q9_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 256];
    }
    #endif
    #if PARTIAL_DIM > 288
    float q10_tail = 0;
    if (lane_id + 288 < PARTIAL_DIM) {
        q10_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 288];
    }
    #endif
    #if PARTIAL_DIM > 320
    float q11_tail = 0;
    if (lane_id + 320 < PARTIAL_DIM) {
        q11_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 320];
    }
    #endif
    #if PARTIAL_DIM > 352
    float q12_tail = 0;
    if (lane_id + 352 < PARTIAL_DIM) {
        q12_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 352];
    }
    #endif
    #if PARTIAL_DIM > 384
    float q13_tail = 0;
    if (lane_id + 384 < PARTIAL_DIM) {
        q13_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 384];
    }
    #endif
    #if PARTIAL_DIM > 416
    float q14_tail = 0;
    if (lane_id + 416 < PARTIAL_DIM) {
        q14_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 416];
    }
    #endif
    #if PARTIAL_DIM > 448
    float q15_tail = 0;
    if (lane_id + 448 < PARTIAL_DIM) {
        q15_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 448];
    }
    #endif
    #if PARTIAL_DIM > 480
    float q16_tail = 0;
    if (lane_id + 480 < PARTIAL_DIM) {
        q16_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 480];
    }
    #endif
    #if PARTIAL_DIM > 512
    float q17_tail = 0;
    if (lane_id + 512 < PARTIAL_DIM) {
        q17_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 512];
    }
    #endif
    #if PARTIAL_DIM > 544
    float q18_tail = 0;
    if (lane_id + 544 < PARTIAL_DIM) {
        q18_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 544];
    }
    #endif
    #if PARTIAL_DIM > 576
    float q19_tail = 0;
    if (lane_id + 576 < PARTIAL_DIM) {
        q19_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 576];
    }
    #endif
    #if PARTIAL_DIM > 608
    float q20_tail = 0;
    if (lane_id + 608 < PARTIAL_DIM) {
        q20_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 608];
    }
    #endif
    #if PARTIAL_DIM > 640
    float q21_tail = 0;
    if (lane_id + 640 < PARTIAL_DIM) {
        q21_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 640];
    }
    #endif
    #if PARTIAL_DIM > 672
    float q22_tail = 0;
    if (lane_id + 672 < PARTIAL_DIM) {
        q22_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 672];
    }
    #endif
    #if PARTIAL_DIM > 704
    float q23_tail = 0;
    if (lane_id + 704 < PARTIAL_DIM) {
        q23_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 704];
    }
    #endif
    #if PARTIAL_DIM > 736
    float q24_tail = 0;
    if (lane_id + 736 < PARTIAL_DIM) {
        q24_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 736];
    }
    #endif
    #if PARTIAL_DIM > 768
    float q25_tail = 0;
    if (lane_id + 768 < PARTIAL_DIM) {
        q25_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 768];
    }
    #endif
    #if PARTIAL_DIM > 800
    float q26_tail = 0;
    if (lane_id + 800 < PARTIAL_DIM) {
        q26_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 800];
    }
    #endif
    #if PARTIAL_DIM > 832
    float q27_tail = 0;
    if (lane_id + 832 < PARTIAL_DIM) {
        q27_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 832];
    }
    #endif
    #if PARTIAL_DIM > 864
    float q28_tail = 0;
    if (lane_id + 864 < PARTIAL_DIM) {
        q28_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 864];
    }
    #endif
    #if PARTIAL_DIM > 896
    float q29_tail = 0;
    if (lane_id + 896 < PARTIAL_DIM) {
        q29_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 896];
    }
    #endif
    #if PARTIAL_DIM > 928
    float q30_tail = 0;
    if (lane_id + 928 < PARTIAL_DIM) {
        q30_tail = query_full[q_id * FULL_DIM + DIM + lane_id + 928];
    }
    #endif
// #endregion PARTIAL_DIM d_query loading
#endif

// insert entry points
  // int* enter_points = d_entries + q_id * n_entries;

  int iteration;
  int step_id;
  int substep_id;
  KernelPair<float, int> tmp_neighbor;

  //初始化neighbors_array，均匀分配到每个线程
  for(int i = t_id; i < n_candidates + n_points_per_batch; i += blockSize){
    neighbors_array[i].first = MAX;
    neighbors_array[i].second = np;
  }
  __syncthreads();

  iteration = (n_entries + n_points_per_batch - 1) / n_points_per_batch;
  for(int iter = 0; iter < iteration; iter++){
  //add entry points
    for(int i = t_id; i < n_points_per_batch; i+=blockSize){
      int unrollt_id = iter * n_points_per_batch + i;
      if(unrollt_id < n_entries){
        if(ALGO == 0 || ALGO == 2)//random entry
          neighbors_array[n_candidates + i].second = random_entry(0, np, unrollt_id);
            // neighbors_array[n_candidates + i].second = unrollt_id;
        else if(ALGO == 1)//rt entry
          neighbors_array[n_candidates + i].second = d_entries[entries_st + unrollt_id];

        add(neighbors_array[n_candidates + i].second, random_number, data);
      }
      else
        neighbors_array[n_candidates + i].second = np;

      if(neighbors_array[n_candidates + i].second >= np)
        neighbors_array[n_candidates + i].first = MAX;
    }
    __syncthreads();

    for(int i = warp_id; i < n_points_per_batch; i+=n_warp){
      long long p_id = neighbors_array[n_candidates + i].second;
      if(p_id >= np) continue;
    // read point
    // #region DIM d_data loading
      #if DIM > 0
      float p1 = 0;
      if (lane_id < DIM) {
          p1 = d_data[p_id * DIM + lane_id];
      }
      #endif
      #if DIM > 32
      float p2 = 0;
      if (lane_id + 32 < DIM) {
          p2 = d_data[p_id * DIM + lane_id + 32];
      }
      #endif
      #if DIM > 64
      float p3 = 0;
      if (lane_id + 64 < DIM) {
          p3 = d_data[p_id * DIM + lane_id + 64];
      }
      #endif
      #if DIM > 96
      float p4 = 0;
      if (lane_id + 96 < DIM) {
          p4 = d_data[p_id * DIM + lane_id + 96];
      }
      #endif
      #if DIM > 128
      float p5 = 0;
      if (lane_id + 128 < DIM) {
          p5 = d_data[p_id * DIM + lane_id + 128];
      }
      #endif
      #if DIM > 160
      float p6 = 0;
      if (lane_id + 160 < DIM) {
          p6 = d_data[p_id * DIM + lane_id + 160];
      }
      #endif
      #if DIM > 192
      float p7 = 0;
      if (lane_id + 192 < DIM) {
          p7 = d_data[p_id * DIM + lane_id + 192];
      }
      #endif
      #if DIM > 224
      float p8 = 0;
      if (lane_id + 224 < DIM) {
          p8 = d_data[p_id * DIM + lane_id + 224];
      }
      #endif
      #if DIM > 256
      float p9 = 0;
      if (lane_id + 256 < DIM) {
          p9 = d_data[p_id * DIM + lane_id + 256];
      }
      #endif
      #if DIM > 288
      float p10 = 0;
      if (lane_id + 288 < DIM) {
          p10 = d_data[p_id * DIM + lane_id + 288];
      }
      #endif
      #if DIM > 320
      float p11 = 0;
      if (lane_id + 320 < DIM) {
          p11 = d_data[p_id * DIM + lane_id + 320];
      }
      #endif
      #if DIM > 352
      float p12 = 0;
      if (lane_id + 352 < DIM) {
          p12 = d_data[p_id * DIM + lane_id + 352];
      }
      #endif
      #if DIM > 384
      float p13 = 0;
      if (lane_id + 384 < DIM) {
          p13 = d_data[p_id * DIM + lane_id + 384];
      }
      #endif
      #if DIM > 416
      float p14 = 0;
      if (lane_id + 416 < DIM) {
          p14 = d_data[p_id * DIM + lane_id + 416];
      }
      #endif
      #if DIM > 448
      float p15 = 0;
      if (lane_id + 448 < DIM) {
          p15 = d_data[p_id * DIM + lane_id + 448];
      }
      #endif
      #if DIM > 480
      float p16 = 0;
      if (lane_id + 480 < DIM) {
          p16 = d_data[p_id * DIM + lane_id + 480];
      }
      #endif
      #if DIM > 512
      float p17 = 0;
      if (lane_id + 512 < DIM) {
          p17 = d_data[p_id * DIM + lane_id + 512];
      }
      #endif
      #if DIM > 544
      float p18 = 0;
      if (lane_id + 544 < DIM) {
          p18 = d_data[p_id * DIM + lane_id + 544];
      }
      #endif
      #if DIM > 576
      float p19 = 0;
      if (lane_id + 576 < DIM) {
          p19 = d_data[p_id * DIM + lane_id + 576];
      }
      #endif
      #if DIM > 608
      float p20 = 0;
      if (lane_id + 608 < DIM) {
          p20 = d_data[p_id * DIM + lane_id + 608];
      }
      #endif
      #if DIM > 640
      float p21 = 0;
      if (lane_id + 640 < DIM) {
          p21 = d_data[p_id * DIM + lane_id + 640];
      }
      #endif
      #if DIM > 672
      float p22 = 0;
      if (lane_id + 672 < DIM) {
          p22 = d_data[p_id * DIM + lane_id + 672];
      }
      #endif
      #if DIM > 704
      float p23 = 0;
      if (lane_id + 704 < DIM) {
          p23 = d_data[p_id * DIM + lane_id + 704];
      }
      #endif
      #if DIM > 736
      float p24 = 0;
      if (lane_id + 736 < DIM) {
          p24 = d_data[p_id * DIM + lane_id + 736];
      }
      #endif
      #if DIM > 768
      float p25 = 0;
      if (lane_id + 768 < DIM) {
          p25 = d_data[p_id * DIM + lane_id + 768];
      }
      #endif
      #if DIM > 800
      float p26 = 0;
      if (lane_id + 800 < DIM) {
          p26 = d_data[p_id * DIM + lane_id + 800];
      }
      #endif
      #if DIM > 832
      float p27 = 0;
      if (lane_id + 832 < DIM) {
          p27 = d_data[p_id * DIM + lane_id + 832];
      }
      #endif
      #if DIM > 864
      float p28 = 0;
      if (lane_id + 864 < DIM) {
          p28 = d_data[p_id * DIM + lane_id + 864];
      }
      #endif
      #if DIM > 896
      float p29 = 0;
      if (lane_id + 896 < DIM) {
          p29 = d_data[p_id * DIM + lane_id + 896];
      }
      #endif
      #if DIM > 928
      float p30 = 0;
      if (lane_id + 928 < DIM) {
          p30 = d_data[p_id * DIM + lane_id + 928];
      }
      #endif
    // #endregion DIM d_data loading

    // compute distance
    // #region DIM distance computation
      #ifdef USE_L2_DIST_
      #if DIM > 0
          float delta1 = (p1 - q1) * (p1 - q1);
      #endif
      #if DIM > 32
          float delta2 = (p2 - q2) * (p2 - q2);
      #endif
      #if DIM > 64
          float delta3 = (p3 - q3) * (p3 - q3);
      #endif
      #if DIM > 96
          float delta4 = (p4 - q4) * (p4 - q4);
      #endif
      #if DIM > 128
          float delta5 = (p5 - q5) * (p5 - q5);
      #endif
      #if DIM > 160
          float delta6 = (p6 - q6) * (p6 - q6);
      #endif
      #if DIM > 192
          float delta7 = (p7 - q7) * (p7 - q7);
      #endif
      #if DIM > 224
          float delta8 = (p8 - q8) * (p8 - q8);
      #endif
      #if DIM > 256
          float delta9 = (p9 - q9) * (p9 - q9);
      #endif
      #if DIM > 288
          float delta10 = (p10 - q10) * (p10 - q10);
      #endif
      #if DIM > 320
          float delta11 = (p11 - q11) * (p11 - q11);
      #endif
      #if DIM > 352
          float delta12 = (p12 - q12) * (p12 - q12);
      #endif
      #if DIM > 384
          float delta13 = (p13 - q13) * (p13 - q13);
      #endif
      #if DIM > 416
          float delta14 = (p14 - q14) * (p14 - q14);
      #endif
      #if DIM > 448
          float delta15 = (p15 - q15) * (p15 - q15);
      #endif
      #if DIM > 480
          float delta16 = (p16 - q16) * (p16 - q16);
      #endif
      #if DIM > 512
          float delta17 = (p17 - q17) * (p17 - q17);
      #endif
      #if DIM > 544
          float delta18 = (p18 - q18) * (p18 - q18);
      #endif
      #if DIM > 576
          float delta19 = (p19 - q19) * (p19 - q19);
      #endif
      #if DIM > 608
          float delta20 = (p20 - q20) * (p20 - q20);
      #endif
      #if DIM > 640
          float delta21 = (p21 - q21) * (p21 - q21);
      #endif
      #if DIM > 672
          float delta22 = (p22 - q22) * (p22 - q22);
      #endif
      #if DIM > 704
          float delta23 = (p23 - q23) * (p23 - q23);
      #endif
      #if DIM > 736
          float delta24 = (p24 - q24) * (p24 - q24);
      #endif
      #if DIM > 768
          float delta25 = (p25 - q25) * (p25 - q25);
      #endif
      #if DIM > 800
          float delta26 = (p26 - q26) * (p26 - q26);
      #endif
      #if DIM > 832
          float delta27 = (p27 - q27) * (p27 - q27);
      #endif
      #if DIM > 864
          float delta28 = (p28 - q28) * (p28 - q28);
      #endif
      #if DIM > 896
          float delta29 = (p29 - q29) * (p29 - q29);
      #endif
      #if DIM > 928
          float delta30 = (p30 - q30) * (p30 - q30);
      #endif
      #endif
    // #endregion DIM distance computation

    #ifdef COMPUTE_COUNT
    // 统计维度计算次数和距离计算次数
    if(lane_id == 0){
      atomicAdd(&compute_dim, (unsigned long long)DIM);
      atomicAdd(&compute_dis, 1ULL);
    }
    #endif

    // reduce
    // #region DIM distance reduction
      #ifdef USE_L2_DIST_
          float dist = 0;
      #if DIM > 0
          dist += delta1;
      #endif
      #if DIM > 32
          dist += delta2;
      #endif
      #if DIM > 64
          dist += delta3;
      #endif
      #if DIM > 96
          dist += delta4;
      #endif
      #if DIM > 128
          dist += delta5;
      #endif
      #if DIM > 160
          dist += delta6;
      #endif
      #if DIM > 192
          dist += delta7;
      #endif
      #if DIM > 224
          dist += delta8;
      #endif
      #if DIM > 256
          dist += delta9;
      #endif
      #if DIM > 288
          dist += delta10;
      #endif
      #if DIM > 320
          dist += delta11;
      #endif
      #if DIM > 352
          dist += delta12;
      #endif
      #if DIM > 384
          dist += delta13;
      #endif
      #if DIM > 416
          dist += delta14;
      #endif
      #if DIM > 448
          dist += delta15;
      #endif
      #if DIM > 480
          dist += delta16;
      #endif
      #if DIM > 512
          dist += delta17;
      #endif
      #if DIM > 544
          dist += delta18;
      #endif
      #if DIM > 576
          dist += delta19;
      #endif
      #if DIM > 608
          dist += delta20;
      #endif
      #if DIM > 640
          dist += delta21;
      #endif
      #if DIM > 672
          dist += delta22;
      #endif
      #if DIM > 704
          dist += delta23;
      #endif
      #if DIM > 736
          dist += delta24;
      #endif
      #if DIM > 768
          dist += delta25;
      #endif
      #if DIM > 800
          dist += delta26;
      #endif
      #if DIM > 832
          dist += delta27;
      #endif
      #if DIM > 864
          dist += delta28;
      #endif
      #if DIM > 896
          dist += delta29;
      #endif
      #if DIM > 928
          dist += delta30;
      #endif
      #endif
      #ifdef USE_L2_DIST_
      dist += __shfl_down_sync(FULL_MASK, dist, 16);
      dist += __shfl_down_sync(FULL_MASK, dist, 8);
      dist += __shfl_down_sync(FULL_MASK, dist, 4);
      dist += __shfl_down_sync(FULL_MASK, dist, 2);
      dist += __shfl_down_sync(FULL_MASK, dist, 1);
      #endif
    // #endregion DIM distance reduction
    
    // insert
      #ifdef USE_CACHE
        const float* page_ptr = nullptr;
        return_type ret;
        if(lane_id == 0){
            ret = whether_cache(static_cast<int>(p_id),
                point_infos, cluster_to_page,
                cache_data, page_size);
        }
        bool in_cache = __shfl_sync(FULL_MASK, ret.in_cache, 0);
        unsigned long long page_u64 = __shfl_sync(FULL_MASK, reinterpret_cast<unsigned long long>(ret.page_ptr), 0);
        page_ptr = reinterpret_cast<float*>(page_u64);

        if(in_cache){
            float partial_dist = 0;
            //read point
            // #region PARTIAL_DIM d_data loading
                #if PARTIAL_DIM > 0
                float p1 = 0;
                if (lane_id < PARTIAL_DIM) {
                    p1 = page_ptr[lane_id];
                }
                #endif
                #if PARTIAL_DIM > 32
                float p2 = 0;
                if (lane_id + 32 < PARTIAL_DIM) {
                    p2 = page_ptr[lane_id + 32];
                }
                #endif
                #if PARTIAL_DIM > 64
                float p3 = 0;
                if (lane_id + 64 < PARTIAL_DIM) {
                    p3 = page_ptr[lane_id + 64];
                }
                #endif
                #if PARTIAL_DIM > 96
                float p4 = 0;
                if (lane_id + 96 < PARTIAL_DIM) {
                    p4 = page_ptr[lane_id + 96];
                }
                #endif
                #if PARTIAL_DIM > 128
                float p5 = 0;
                if (lane_id + 128 < PARTIAL_DIM) {
                    p5 = page_ptr[lane_id + 128];
                }
                #endif
                #if PARTIAL_DIM > 160
                float p6 = 0;
                if (lane_id + 160 < PARTIAL_DIM) {
                    p6 = page_ptr[lane_id + 160];
                }
                #endif
                #if PARTIAL_DIM > 192
                float p7 = 0;
                if (lane_id + 192 < PARTIAL_DIM) {
                    p7 = page_ptr[lane_id + 192];
                }
                #endif
                #if PARTIAL_DIM > 224
                float p8 = 0;
                if (lane_id + 224 < PARTIAL_DIM) {
                    p8 = page_ptr[lane_id + 224];
                }
                #endif
                #if PARTIAL_DIM > 256
                float p9 = 0;
                if (lane_id + 256 < PARTIAL_DIM) {
                    p9 = page_ptr[lane_id + 256];
                }
                #endif
                #if PARTIAL_DIM > 288
                float p10 = 0;
                if (lane_id + 288 < PARTIAL_DIM) {
                    p10 = page_ptr[lane_id + 288];
                }
                #endif
                #if PARTIAL_DIM > 320
                float p11 = 0;
                if (lane_id + 320 < PARTIAL_DIM) {
                    p11 = page_ptr[lane_id + 320];
                }
                #endif
                #if PARTIAL_DIM > 352
                float p12 = 0;
                if (lane_id + 352 < PARTIAL_DIM) {
                    p12 = page_ptr[lane_id + 352];
                }
                #endif
                #if PARTIAL_DIM > 384
                float p13 = 0;
                if (lane_id + 384 < PARTIAL_DIM) {
                    p13 = page_ptr[lane_id + 384];
                }
                #endif
                #if PARTIAL_DIM > 416
                float p14 = 0;
                if (lane_id + 416 < PARTIAL_DIM) {
                    p14 = page_ptr[lane_id + 416];
                }
                #endif
                #if PARTIAL_DIM > 448
                float p15 = 0;
                if (lane_id + 448 < PARTIAL_DIM) {
                    p15 = page_ptr[lane_id + 448];
                }
                #endif
                #if PARTIAL_DIM > 480
                float p16 = 0;
                if (lane_id + 480 < PARTIAL_DIM) {
                    p16 = page_ptr[lane_id + 480];
                }
                #endif
                #if PARTIAL_DIM > 512
                float p17 = 0;
                if (lane_id + 512 < PARTIAL_DIM) {
                    p17 = page_ptr[lane_id + 512];
                }
                #endif
                #if PARTIAL_DIM > 544
                float p18 = 0;
                if (lane_id + 544 < PARTIAL_DIM) {
                    p18 = page_ptr[lane_id + 544];
                }
                #endif
                #if PARTIAL_DIM > 576
                float p19 = 0;
                if (lane_id + 576 < PARTIAL_DIM) {
                    p19 = page_ptr[lane_id + 576];
                }
                #endif
                #if PARTIAL_DIM > 608
                float p20 = 0;
                if (lane_id + 608 < PARTIAL_DIM) {
                    p20 = page_ptr[lane_id + 608];
                }
                #endif
                #if PARTIAL_DIM > 640
                float p21 = 0;
                if (lane_id + 640 < PARTIAL_DIM) {
                    p21 = page_ptr[lane_id + 640];
                }
                #endif
                #if PARTIAL_DIM > 672
                float p22 = 0;
                if (lane_id + 672 < PARTIAL_DIM) {
                    p22 = page_ptr[lane_id + 672];
                }
                #endif
                #if PARTIAL_DIM > 704
                float p23 = 0;
                if (lane_id + 704 < PARTIAL_DIM) {
                    p23 = page_ptr[lane_id + 704];
                }
                #endif
                #if PARTIAL_DIM > 736
                float p24 = 0;
                if (lane_id + 736 < PARTIAL_DIM) {
                    p24 = page_ptr[lane_id + 736];
                }
                #endif
                #if PARTIAL_DIM > 768
                float p25 = 0;
                if (lane_id + 768 < PARTIAL_DIM) {
                    p25 = page_ptr[lane_id + 768];
                }
                #endif
                #if PARTIAL_DIM > 800
                float p26 = 0;
                if (lane_id + 800 < PARTIAL_DIM) {
                    p26 = page_ptr[lane_id + 800];
                }
                #endif
                #if PARTIAL_DIM > 832
                float p27 = 0;
                if (lane_id + 832 < PARTIAL_DIM) {
                    p27 = page_ptr[lane_id + 832];
                }
                #endif
                #if PARTIAL_DIM > 864
                float p28 = 0;
                if (lane_id + 864 < PARTIAL_DIM) {
                    p28 = page_ptr[lane_id + 864];
                }
                #endif
                #if PARTIAL_DIM > 896
                float p29 = 0;
                if (lane_id + 896 < PARTIAL_DIM) {
                    p29 = page_ptr[lane_id + 896];
                }
                #endif
                #if PARTIAL_DIM > 928
                float p30 = 0;
                if (lane_id + 928 < PARTIAL_DIM) {
                    p30 = page_ptr[lane_id + 928];
                }
                #endif
            // #endregion PARTIAL_DIM d_data loading

            //compute distance
            // #region PARTIAL_DIM distance computation
                #ifdef USE_L2_DIST_
                #if PARTIAL_DIM > 0
                    float delta1 = (p1 - q1_tail) * (p1 - q1_tail);
                #endif
                #if PARTIAL_DIM > 32
                    float delta2 = (p2 - q2_tail) * (p2 - q2_tail);
                #endif
                #if PARTIAL_DIM > 64
                    float delta3 = (p3 - q3_tail) * (p3 - q3_tail);
                #endif
                #if PARTIAL_DIM > 96
                    float delta4 = (p4 - q4_tail) * (p4 - q4_tail);
                #endif
                #if PARTIAL_DIM > 128
                    float delta5 = (p5 - q5_tail) * (p5 - q5_tail);
                #endif
                #if PARTIAL_DIM > 160
                    float delta6 = (p6 - q6_tail) * (p6 - q6_tail);
                #endif
                #if PARTIAL_DIM > 192
                    float delta7 = (p7 - q7_tail) * (p7 - q7_tail);
                #endif
                #if PARTIAL_DIM > 224
                    float delta8 = (p8 - q8_tail) * (p8 - q8_tail);
                #endif
                #if PARTIAL_DIM > 256
                    float delta9 = (p9 - q9_tail) * (p9 - q9_tail);
                #endif
                #if PARTIAL_DIM > 288
                    float delta10 = (p10 - q10_tail) * (p10 - q10_tail);
                #endif
                #if PARTIAL_DIM > 320
                    float delta11 = (p11 - q11_tail) * (p11 - q11_tail);
                #endif
                #if PARTIAL_DIM > 352
                    float delta12 = (p12 - q12_tail) * (p12 - q12_tail);
                #endif
                #if PARTIAL_DIM > 384
                    float delta13 = (p13 - q13_tail) * (p13 - q13_tail);
                #endif
                #if PARTIAL_DIM > 416
                    float delta14 = (p14 - q14_tail) * (p14 - q14_tail);
                #endif
                #if PARTIAL_DIM > 448
                    float delta15 = (p15 - q15_tail) * (p15 - q15_tail);
                #endif
                #if PARTIAL_DIM > 480
                    float delta16 = (p16 - q16_tail) * (p16 - q16_tail);
                #endif
                #if PARTIAL_DIM > 512
                    float delta17 = (p17 - q17_tail) * (p17 - q17_tail);
                #endif
                #if PARTIAL_DIM > 544
                    float delta18 = (p18 - q18_tail) * (p18 - q18_tail);
                #endif
                #if PARTIAL_DIM > 576
                    float delta19 = (p19 - q19_tail) * (p19 - q19_tail);
                #endif
                #if PARTIAL_DIM > 608
                    float delta20 = (p20 - q20_tail) * (p20 - q20_tail);
                #endif
                #if PARTIAL_DIM > 640
                    float delta21 = (p21 - q21_tail) * (p21 - q21_tail);
                #endif
                #if PARTIAL_DIM > 672
                    float delta22 = (p22 - q22_tail) * (p22 - q22_tail);
                #endif
                #if PARTIAL_DIM > 704
                    float delta23 = (p23 - q23_tail) * (p23 - q23_tail);
                #endif
                #if PARTIAL_DIM > 736
                    float delta24 = (p24 - q24_tail) * (p24 - q24_tail);
                #endif
                #if PARTIAL_DIM > 768
                    float delta25 = (p25 - q25_tail) * (p25 - q25_tail);
                #endif
                #if PARTIAL_DIM > 800
                    float delta26 = (p26 - q26_tail) * (p26 - q26_tail);
                #endif
                #if PARTIAL_DIM > 832
                    float delta27 = (p27 - q27_tail) * (p27 - q27_tail);
                #endif
                #if PARTIAL_DIM > 864
                    float delta28 = (p28 - q28_tail) * (p28 - q28_tail);
                #endif
                #if PARTIAL_DIM > 896
                    float delta29 = (p29 - q29_tail) * (p29 - q29_tail);
                #endif
                #if PARTIAL_DIM > 928
                    float delta30 = (p30 - q30_tail) * (p30 - q30_tail);
                #endif
                #endif
            // #endregion PARTIAL_DIM distance computation

            #ifdef COMPUTE_COUNT
            // 统计维度计算次数和距离计算次数
            if(lane_id == 0){
              atomicAdd(&compute_dim, (unsigned long long)PARTIAL_DIM);
            //   atomicAdd(&compute_dis, 1ULL);
            }
            #endif

            // #region PARTIAL_DIM distance reduction
                #ifdef USE_L2_DIST_
                    
                #if PARTIAL_DIM > 0
                    partial_dist += delta1;
                #endif
                #if PARTIAL_DIM > 32
                    partial_dist += delta2;
                #endif
                #if PARTIAL_DIM > 64
                    partial_dist += delta3;
                #endif
                #if PARTIAL_DIM > 96
                    partial_dist += delta4;
                #endif
                #if PARTIAL_DIM > 128
                    partial_dist += delta5;
                #endif
                #if PARTIAL_DIM > 160
                    partial_dist += delta6;
                #endif
                #if PARTIAL_DIM > 192
                    partial_dist += delta7;
                #endif
                #if PARTIAL_DIM > 224
                    partial_dist += delta8;
                #endif
                #if PARTIAL_DIM > 256
                    partial_dist += delta9;
                #endif
                #if PARTIAL_DIM > 288
                    partial_dist += delta10;
                #endif
                #if PARTIAL_DIM > 320
                    partial_dist += delta11;
                #endif
                #if PARTIAL_DIM > 352
                    partial_dist += delta12;
                #endif
                #if PARTIAL_DIM > 384
                    partial_dist += delta13;
                #endif
                #if PARTIAL_DIM > 416
                    partial_dist += delta14;
                #endif
                #if PARTIAL_DIM > 448
                    partial_dist += delta15;
                #endif
                #if PARTIAL_DIM > 480
                    partial_dist += delta16;
                #endif
                #if PARTIAL_DIM > 512
                    partial_dist += delta17;
                #endif
                #if PARTIAL_DIM > 544
                    partial_dist += delta18;
                #endif
                #if PARTIAL_DIM > 576
                    partial_dist += delta19;
                #endif
                #if PARTIAL_DIM > 608
                    partial_dist += delta20;
                #endif
                #if PARTIAL_DIM > 640
                    partial_dist += delta21;
                #endif
                #if PARTIAL_DIM > 672
                    partial_dist += delta22;
                #endif
                #if PARTIAL_DIM > 704
                    partial_dist += delta23;
                #endif
                #if PARTIAL_DIM > 736
                    partial_dist += delta24;
                #endif
                #if PARTIAL_DIM > 768
                    partial_dist += delta25;
                #endif
                #if PARTIAL_DIM > 800
                    partial_dist += delta26;
                #endif
                #if PARTIAL_DIM > 832
                    partial_dist += delta27;
                #endif
                #if PARTIAL_DIM > 864
                    partial_dist += delta28;
                #endif
                #if PARTIAL_DIM > 896
                    partial_dist += delta29;
                #endif
                #if PARTIAL_DIM > 928
                    partial_dist += delta30;
                #endif
                #endif
                #ifdef USE_L2_DIST_
                partial_dist += __shfl_down_sync(FULL_MASK, partial_dist, 16);
                partial_dist += __shfl_down_sync(FULL_MASK, partial_dist, 8);
                partial_dist += __shfl_down_sync(FULL_MASK, partial_dist, 4);
                partial_dist += __shfl_down_sync(FULL_MASK, partial_dist, 2);
                partial_dist += __shfl_down_sync(FULL_MASK, partial_dist, 1);
                #endif
            // #endregion PARTIAL_DIM distance reduction
            if(lane_id == 0){
                dist += partial_dist;
                if(PARTIAL_DIM < FULL_DIM - DIM && linear_dim > 0 && linear_w && linear_b){
                    int idx = (linear_dim < PARTIAL_DIM + DIM ? linear_dim : PARTIAL_DIM + DIM);
                    if (idx >= 0) {
                        dist = linear_w[idx] * dist + linear_b[idx];
                        dist = max(dist, 0.0f);
                    }
                }
                neighbors_array[n_candidates + i].first = dist;
            }
            // if(lane_id == 0){
            //     if (linear_dim > 0 && linear_w && linear_b) {
            //         int idx = (linear_dim < DIM ? linear_dim : DIM);
            //         if (idx >= 0) {
            //             dist = linear_w[idx] * dist + linear_b[idx];
            //             dist = max(dist, 0.0f);
            //         }
            //     }
            //     neighbors_array[n_candidates + i].first = dist;
            // }
        }
        else{
            if(lane_id == 0){
                if (linear_dim > 0 && linear_w && linear_b) {
                    int idx = (linear_dim < DIM ? linear_dim : DIM);
                    if (idx >= 0) {
                        dist = linear_w[idx] * dist + linear_b[idx];
                        dist = max(dist, 0.0f);
                    }
                }
                neighbors_array[n_candidates + i].first = dist;
            }
        }

      #else
        if(lane_id == 0){
            neighbors_array[n_candidates + i].first = dist;
        }
      #endif
    }
    __syncthreads();

  //bitonic-sort
    step_id = 1;
    substep_id = 1;

    for(; step_id <= n_points_per_batch/2; step_id *= 2){
      substep_id = step_id;

      for(; substep_id >= 1; substep_id /= 2){
        // for(int temparory_id = 0; temparory_id < (n_points_per_batch/2 + blockSize - 1)/blockSize; )
        for(int i=t_id; i<n_points_per_batch; i+=blockSize){
          int unrollt_id = n_candidates + (i/substep_id) * 2 * substep_id + (i&(substep_id-1));
          if(unrollt_id < n_candidates + n_points_per_batch && unrollt_id + substep_id < n_candidates + n_points_per_batch){
            if((i/step_id)%2 == 0){
              if(neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first){
                tmp_neighbor = neighbors_array[unrollt_id];
                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                neighbors_array[unrollt_id + substep_id] = tmp_neighbor;
              }
            }
            else{
              if(neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first){
                tmp_neighbor = neighbors_array[unrollt_id];
                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                neighbors_array[unrollt_id + substep_id] = tmp_neighbor;
              }
            }
          }
        }
      }
      __syncthreads();
    }
    __syncthreads();

    for(int i = t_id; i < n_compare; i += blockSize){
      int unrollt_id = n_candidates - n_compare + i;
      if(unrollt_id < n_candidates){
        if(neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + n_points_per_batch].first){
          tmp_neighbor = neighbors_array[unrollt_id];
          neighbors_array[unrollt_id] = neighbors_array[unrollt_id + n_points_per_batch];
          neighbors_array[unrollt_id + n_points_per_batch] = tmp_neighbor;
        }
      }
    }
    __syncthreads();

    step_id = n_candidates / 2;
    substep_id = n_candidates / 2;
    for(; substep_id >= 1; substep_id /= 2){
      for(int i = t_id; i < n_candidates/2; i += blockSize){
        int unrollt_id = (i/substep_id) * 2 * substep_id + (i & (substep_id - 1));
        if(unrollt_id < n_candidates && unrollt_id + substep_id < n_candidates){
          if((i/step_id) % 2 == 0){
            if(neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first){
              tmp_neighbor = neighbors_array[unrollt_id];
              neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
              neighbors_array[unrollt_id + substep_id] = tmp_neighbor;
            }
          }
          else{
            if(neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first){
              tmp_neighbor = neighbors_array[unrollt_id];
              neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
              neighbors_array[unrollt_id + substep_id] = tmp_neighbor;
            }
          }
        }
      }
      __syncthreads();
    }
    __syncthreads();
  }
  __syncthreads();

// graph_search
  //TODO: max_iter
  // int max_iter = 0;
  int flag_all_blocks = 1;
  int iter = 0;
  int crt_flag = 0;
  int check_zero = 0;
  int search_cnt;
  int hash_iteration = 0;
  int tmp_flag = (1 << min(min(n_candidates, WARP_SIZE), n_entries)) - 1;
//   if(t_id == 0 && b_id == 0){
//     printf("n_candidate=%d n_entries=%d tmp_flag=%d\n", n_candidates, n_entries, tmp_flag);
//   }
  long long first_position_of_flag;
  iteration = 0;
  while(flag_all_blocks && iter < max_iter){
    iter ++;

    for(int i=t_id; i<n_points_per_batch; i+=blockSize){
      neighbors_array[n_candidates + i].first = MAX;
      neighbors_array[n_candidates + i].second = np;
    }
    __syncthreads();

    search_cnt = 0;
    while(search_cnt < search_width && tmp_flag != 0){
      int first_position_of_tmp_flag = __ffs(tmp_flag) - 1; //__ffs返回第一个非零位
      int neighbor_loc = iteration * WARP_SIZE + first_position_of_tmp_flag;
      first_position_of_flag = _abs(neighbors_array[neighbor_loc].second);
      if(t_id == 0){
        neighbors_array[neighbor_loc].second = -neighbors_array[neighbor_loc].second;
      }
      tmp_flag &= ~(1 << first_position_of_tmp_flag);

      //read neighbors
      long long offset = first_position_of_flag << (1LL * offset_shift);
      int* neighbors = d_graph + offset;
      for(int i=t_id; i<degree; i+=blockSize){
        int target_point = neighbors[i];
        if(target_point == np || test(target_point, random_number, data)){
          neighbors_array[n_candidates + i + search_cnt * degree].second = np;
        }
        else{
          // if(q_id == 0) printf("target_point=%d\n", target_point);
          neighbors_array[n_candidates + i + search_cnt * degree].second = target_point;
          add(target_point, random_number, data);
        }
      }
      __syncthreads();

      search_cnt++;
      if(search_cnt == search_width) break;

      while(tmp_flag == 0 && iteration < (n_candidates + WARP_SIZE-1)/WARP_SIZE){
        iteration ++;
        int unrollt_id = lane_id + WARP_SIZE * iteration;
        crt_flag = 0;
        if(unrollt_id < n_candidates){
          if(neighbors_array[unrollt_id].second > 0 && neighbors_array[unrollt_id].second < np){
            crt_flag = 1;
          }
          else if(neighbors_array[unrollt_id].second == 0){
            if(check_zero == 0){
              check_zero = 1;
              crt_flag = 1;
            }
          }
        }
        tmp_flag = __ballot_sync(FULL_MASK, crt_flag);
      }
    }
    __syncthreads();
    if(search_cnt == 0) break;

    //compute distance between query and search_width points
    for(int i = warp_id; i < search_cnt * degree; i += n_warp){
      long long p_id = neighbors_array[n_candidates + i].second;
      if(p_id >= np) {
        neighbors_array[n_candidates + i].first = MAX;
        continue;
      }
    //read point
    // #region DIM d_data loading
      #if DIM > 0
      float p1 = 0;
      if (lane_id < DIM) {
          p1 = d_data[p_id * DIM + lane_id];
      }
      #endif
      #if DIM > 32
      float p2 = 0;
      if (lane_id + 32 < DIM) {
          p2 = d_data[p_id * DIM + lane_id + 32];
      }
      #endif
      #if DIM > 64
      float p3 = 0;
      if (lane_id + 64 < DIM) {
          p3 = d_data[p_id * DIM + lane_id + 64];
      }
      #endif
      #if DIM > 96
      float p4 = 0;
      if (lane_id + 96 < DIM) {
          p4 = d_data[p_id * DIM + lane_id + 96];
      }
      #endif
      #if DIM > 128
      float p5 = 0;
      if (lane_id + 128 < DIM) {
          p5 = d_data[p_id * DIM + lane_id + 128];
      }
      #endif
      #if DIM > 160
      float p6 = 0;
      if (lane_id + 160 < DIM) {
          p6 = d_data[p_id * DIM + lane_id + 160];
      }
      #endif
      #if DIM > 192
      float p7 = 0;
      if (lane_id + 192 < DIM) {
          p7 = d_data[p_id * DIM + lane_id + 192];
      }
      #endif
      #if DIM > 224
      float p8 = 0;
      if (lane_id + 224 < DIM) {
          p8 = d_data[p_id * DIM + lane_id + 224];
      }
      #endif
      #if DIM > 256
      float p9 = 0;
      if (lane_id + 256 < DIM) {
          p9 = d_data[p_id * DIM + lane_id + 256];
      }
      #endif
      #if DIM > 288
      float p10 = 0;
      if (lane_id + 288 < DIM) {
          p10 = d_data[p_id * DIM + lane_id + 288];
      }
      #endif
      #if DIM > 320
      float p11 = 0;
      if (lane_id + 320 < DIM) {
          p11 = d_data[p_id * DIM + lane_id + 320];
      }
      #endif
      #if DIM > 352
      float p12 = 0;
      if (lane_id + 352 < DIM) {
          p12 = d_data[p_id * DIM + lane_id + 352];
      }
      #endif
      #if DIM > 384
      float p13 = 0;
      if (lane_id + 384 < DIM) {
          p13 = d_data[p_id * DIM + lane_id + 384];
      }
      #endif
      #if DIM > 416
      float p14 = 0;
      if (lane_id + 416 < DIM) {
          p14 = d_data[p_id * DIM + lane_id + 416];
      }
      #endif
      #if DIM > 448
      float p15 = 0;
      if (lane_id + 448 < DIM) {
          p15 = d_data[p_id * DIM + lane_id + 448];
      }
      #endif
      #if DIM > 480
      float p16 = 0;
      if (lane_id + 480 < DIM) {
          p16 = d_data[p_id * DIM + lane_id + 480];
      }
      #endif
      #if DIM > 512
      float p17 = 0;
      if (lane_id + 512 < DIM) {
          p17 = d_data[p_id * DIM + lane_id + 512];
      }
      #endif
      #if DIM > 544
      float p18 = 0;
      if (lane_id + 544 < DIM) {
          p18 = d_data[p_id * DIM + lane_id + 544];
      }
      #endif
      #if DIM > 576
      float p19 = 0;
      if (lane_id + 576 < DIM) {
          p19 = d_data[p_id * DIM + lane_id + 576];
      }
      #endif
      #if DIM > 608
      float p20 = 0;
      if (lane_id + 608 < DIM) {
          p20 = d_data[p_id * DIM + lane_id + 608];
      }
      #endif
      #if DIM > 640
      float p21 = 0;
      if (lane_id + 640 < DIM) {
          p21 = d_data[p_id * DIM + lane_id + 640];
      }
      #endif
      #if DIM > 672
      float p22 = 0;
      if (lane_id + 672 < DIM) {
          p22 = d_data[p_id * DIM + lane_id + 672];
      }
      #endif
      #if DIM > 704
      float p23 = 0;
      if (lane_id + 704 < DIM) {
          p23 = d_data[p_id * DIM + lane_id + 704];
      }
      #endif
      #if DIM > 736
      float p24 = 0;
      if (lane_id + 736 < DIM) {
          p24 = d_data[p_id * DIM + lane_id + 736];
      }
      #endif
      #if DIM > 768
      float p25 = 0;
      if (lane_id + 768 < DIM) {
          p25 = d_data[p_id * DIM + lane_id + 768];
      }
      #endif
      #if DIM > 800
      float p26 = 0;
      if (lane_id + 800 < DIM) {
          p26 = d_data[p_id * DIM + lane_id + 800];
      }
      #endif
      #if DIM > 832
      float p27 = 0;
      if (lane_id + 832 < DIM) {
          p27 = d_data[p_id * DIM + lane_id + 832];
      }
      #endif
      #if DIM > 864
      float p28 = 0;
      if (lane_id + 864 < DIM) {
          p28 = d_data[p_id * DIM + lane_id + 864];
      }
      #endif
      #if DIM > 896
      float p29 = 0;
      if (lane_id + 896 < DIM) {
          p29 = d_data[p_id * DIM + lane_id + 896];
      }
      #endif
      #if DIM > 928
      float p30 = 0;
      if (lane_id + 928 < DIM) {
          p30 = d_data[p_id * DIM + lane_id + 928];
      }
      #endif
    // #endregion DIM d_data loading

    //compute distance
    // #region DIM distance computation
      #ifdef USE_L2_DIST_
      #if DIM > 0
          float delta1 = (p1 - q1) * (p1 - q1);
      #endif
      #if DIM > 32
          float delta2 = (p2 - q2) * (p2 - q2);
      #endif
      #if DIM > 64
          float delta3 = (p3 - q3) * (p3 - q3);
      #endif
      #if DIM > 96
          float delta4 = (p4 - q4) * (p4 - q4);
      #endif
      #if DIM > 128
          float delta5 = (p5 - q5) * (p5 - q5);
      #endif
      #if DIM > 160
          float delta6 = (p6 - q6) * (p6 - q6);
      #endif
      #if DIM > 192
          float delta7 = (p7 - q7) * (p7 - q7);
      #endif
      #if DIM > 224
          float delta8 = (p8 - q8) * (p8 - q8);
      #endif
      #if DIM > 256
          float delta9 = (p9 - q9) * (p9 - q9);
      #endif
      #if DIM > 288
          float delta10 = (p10 - q10) * (p10 - q10);
      #endif
      #if DIM > 320
          float delta11 = (p11 - q11) * (p11 - q11);
      #endif
      #if DIM > 352
          float delta12 = (p12 - q12) * (p12 - q12);
      #endif
      #if DIM > 384
          float delta13 = (p13 - q13) * (p13 - q13);
      #endif
      #if DIM > 416
          float delta14 = (p14 - q14) * (p14 - q14);
      #endif
      #if DIM > 448
          float delta15 = (p15 - q15) * (p15 - q15);
      #endif
      #if DIM > 480
          float delta16 = (p16 - q16) * (p16 - q16);
      #endif
      #if DIM > 512
          float delta17 = (p17 - q17) * (p17 - q17);
      #endif
      #if DIM > 544
          float delta18 = (p18 - q18) * (p18 - q18);
      #endif
      #if DIM > 576
          float delta19 = (p19 - q19) * (p19 - q19);
      #endif
      #if DIM > 608
          float delta20 = (p20 - q20) * (p20 - q20);
      #endif
      #if DIM > 640
          float delta21 = (p21 - q21) * (p21 - q21);
      #endif
      #if DIM > 672
          float delta22 = (p22 - q22) * (p22 - q22);
      #endif
      #if DIM > 704
          float delta23 = (p23 - q23) * (p23 - q23);
      #endif
      #if DIM > 736
          float delta24 = (p24 - q24) * (p24 - q24);
      #endif
      #if DIM > 768
          float delta25 = (p25 - q25) * (p25 - q25);
      #endif
      #if DIM > 800
          float delta26 = (p26 - q26) * (p26 - q26);
      #endif
      #if DIM > 832
          float delta27 = (p27 - q27) * (p27 - q27);
      #endif
      #if DIM > 864
          float delta28 = (p28 - q28) * (p28 - q28);
      #endif
      #if DIM > 896
          float delta29 = (p29 - q29) * (p29 - q29);
      #endif
      #if DIM > 928
          float delta30 = (p30 - q30) * (p30 - q30);
      #endif
      #endif
    // #endregion DIM distance computation

    #ifdef COMPUTE_COUNT
    // 统计维度计算次数和距离计算次数
    if(lane_id == 0){
      atomicAdd(&compute_dim, (unsigned long long)DIM);
      atomicAdd(&compute_dis, 1ULL);
    }
    #endif

    // #region DIM distance reduction
      #ifdef USE_L2_DIST_
          float dist = 0;
      #if DIM > 0
          dist += delta1;
      #endif
      #if DIM > 32
          dist += delta2;
      #endif
      #if DIM > 64
          dist += delta3;
      #endif
      #if DIM > 96
          dist += delta4;
      #endif
      #if DIM > 128
          dist += delta5;
      #endif
      #if DIM > 160
          dist += delta6;
      #endif
      #if DIM > 192
          dist += delta7;
      #endif
      #if DIM > 224
          dist += delta8;
      #endif
      #if DIM > 256
          dist += delta9;
      #endif
      #if DIM > 288
          dist += delta10;
      #endif
      #if DIM > 320
          dist += delta11;
      #endif
      #if DIM > 352
          dist += delta12;
      #endif
      #if DIM > 384
          dist += delta13;
      #endif
      #if DIM > 416
          dist += delta14;
      #endif
      #if DIM > 448
          dist += delta15;
      #endif
      #if DIM > 480
          dist += delta16;
      #endif
      #if DIM > 512
          dist += delta17;
      #endif
      #if DIM > 544
          dist += delta18;
      #endif
      #if DIM > 576
          dist += delta19;
      #endif
      #if DIM > 608
          dist += delta20;
      #endif
      #if DIM > 640
          dist += delta21;
      #endif
      #if DIM > 672
          dist += delta22;
      #endif
      #if DIM > 704
          dist += delta23;
      #endif
      #if DIM > 736
          dist += delta24;
      #endif
      #if DIM > 768
          dist += delta25;
      #endif
      #if DIM > 800
          dist += delta26;
      #endif
      #if DIM > 832
          dist += delta27;
      #endif
      #if DIM > 864
          dist += delta28;
      #endif
      #if DIM > 896
          dist += delta29;
      #endif
      #if DIM > 928
          dist += delta30;
      #endif
      #endif
      #ifdef USE_L2_DIST_
      dist += __shfl_down_sync(FULL_MASK, dist, 16);
      dist += __shfl_down_sync(FULL_MASK, dist, 8);
      dist += __shfl_down_sync(FULL_MASK, dist, 4);
      dist += __shfl_down_sync(FULL_MASK, dist, 2);
      dist += __shfl_down_sync(FULL_MASK, dist, 1);
      #endif
    // #endregion DIM distance reduction

    // insert
    //   if(lane_id == 0){
    //     neighbors_array[n_candidates + i].first = finalize_distance(dist,
    //       static_cast<int>(p_id), q_id,
    //       query_top_clusters, cluster_top_t,
    //       point_infos, cluster_to_page,
    //       cache_data, page_size, dim_partial, dim_total,
    //       query_full, linear_w, linear_b, linear_dim);
    //     // neighbors_array[n_candidates + i].first = dist;
    //   }
      #ifdef USE_CACHE
        const float* page_ptr = nullptr;
        return_type ret;
        if(lane_id == 0){
            ret = whether_cache(static_cast<int>(p_id),
                point_infos, cluster_to_page,
                cache_data, page_size);
        }
        bool in_cache = __shfl_sync(FULL_MASK, ret.in_cache, 0);
        unsigned long long page_u64 = __shfl_sync(FULL_MASK, reinterpret_cast<unsigned long long>(ret.page_ptr), 0);
        page_ptr = reinterpret_cast<float*>(page_u64);

        if(in_cache){
            float partial_dist = 0;
            //read point
            // #region PARTIAL_DIM d_data loading
                #if PARTIAL_DIM > 0
                float p1 = 0;
                if (lane_id < PARTIAL_DIM) {
                    p1 = page_ptr[lane_id];
                }
                #endif
                #if PARTIAL_DIM > 32
                float p2 = 0;
                if (lane_id + 32 < PARTIAL_DIM) {
                    p2 = page_ptr[lane_id + 32];
                }
                #endif
                #if PARTIAL_DIM > 64
                float p3 = 0;
                if (lane_id + 64 < PARTIAL_DIM) {
                    p3 = page_ptr[lane_id + 64];
                }
                #endif
                #if PARTIAL_DIM > 96
                float p4 = 0;
                if (lane_id + 96 < PARTIAL_DIM) {
                    p4 = page_ptr[lane_id + 96];
                }
                #endif
                #if PARTIAL_DIM > 128
                float p5 = 0;
                if (lane_id + 128 < PARTIAL_DIM) {
                    p5 = page_ptr[lane_id + 128];
                }
                #endif
                #if PARTIAL_DIM > 160
                float p6 = 0;
                if (lane_id + 160 < PARTIAL_DIM) {
                    p6 = page_ptr[lane_id + 160];
                }
                #endif
                #if PARTIAL_DIM > 192
                float p7 = 0;
                if (lane_id + 192 < PARTIAL_DIM) {
                    p7 = page_ptr[lane_id + 192];
                }
                #endif
                #if PARTIAL_DIM > 224
                float p8 = 0;
                if (lane_id + 224 < PARTIAL_DIM) {
                    p8 = page_ptr[lane_id + 224];
                }
                #endif
                #if PARTIAL_DIM > 256
                float p9 = 0;
                if (lane_id + 256 < PARTIAL_DIM) {
                    p9 = page_ptr[lane_id + 256];
                }
                #endif
                #if PARTIAL_DIM > 288
                float p10 = 0;
                if (lane_id + 288 < PARTIAL_DIM) {
                    p10 = page_ptr[lane_id + 288];
                }
                #endif
                #if PARTIAL_DIM > 320
                float p11 = 0;
                if (lane_id + 320 < PARTIAL_DIM) {
                    p11 = page_ptr[lane_id + 320];
                }
                #endif
                #if PARTIAL_DIM > 352
                float p12 = 0;
                if (lane_id + 352 < PARTIAL_DIM) {
                    p12 = page_ptr[lane_id + 352];
                }
                #endif
                #if PARTIAL_DIM > 384
                float p13 = 0;
                if (lane_id + 384 < PARTIAL_DIM) {
                    p13 = page_ptr[lane_id + 384];
                }
                #endif
                #if PARTIAL_DIM > 416
                float p14 = 0;
                if (lane_id + 416 < PARTIAL_DIM) {
                    p14 = page_ptr[lane_id + 416];
                }
                #endif
                #if PARTIAL_DIM > 448
                float p15 = 0;
                if (lane_id + 448 < PARTIAL_DIM) {
                    p15 = page_ptr[lane_id + 448];
                }
                #endif
                #if PARTIAL_DIM > 480
                float p16 = 0;
                if (lane_id + 480 < PARTIAL_DIM) {
                    p16 = page_ptr[lane_id + 480];
                }
                #endif
                #if PARTIAL_DIM > 512
                float p17 = 0;
                if (lane_id + 512 < PARTIAL_DIM) {
                    p17 = page_ptr[lane_id + 512];
                }
                #endif
                #if PARTIAL_DIM > 544
                float p18 = 0;
                if (lane_id + 544 < PARTIAL_DIM) {
                    p18 = page_ptr[lane_id + 544];
                }
                #endif
                #if PARTIAL_DIM > 576
                float p19 = 0;
                if (lane_id + 576 < PARTIAL_DIM) {
                    p19 = page_ptr[lane_id + 576];
                }
                #endif
                #if PARTIAL_DIM > 608
                float p20 = 0;
                if (lane_id + 608 < PARTIAL_DIM) {
                    p20 = page_ptr[lane_id + 608];
                }
                #endif
                #if PARTIAL_DIM > 640
                float p21 = 0;
                if (lane_id + 640 < PARTIAL_DIM) {
                    p21 = page_ptr[lane_id + 640];
                }
                #endif
                #if PARTIAL_DIM > 672
                float p22 = 0;
                if (lane_id + 672 < PARTIAL_DIM) {
                    p22 = page_ptr[lane_id + 672];
                }
                #endif
                #if PARTIAL_DIM > 704
                float p23 = 0;
                if (lane_id + 704 < PARTIAL_DIM) {
                    p23 = page_ptr[lane_id + 704];
                }
                #endif
                #if PARTIAL_DIM > 736
                float p24 = 0;
                if (lane_id + 736 < PARTIAL_DIM) {
                    p24 = page_ptr[lane_id + 736];
                }
                #endif
                #if PARTIAL_DIM > 768
                float p25 = 0;
                if (lane_id + 768 < PARTIAL_DIM) {
                    p25 = page_ptr[lane_id + 768];
                }
                #endif
                #if PARTIAL_DIM > 800
                float p26 = 0;
                if (lane_id + 800 < PARTIAL_DIM) {
                    p26 = page_ptr[lane_id + 800];
                }
                #endif
                #if PARTIAL_DIM > 832
                float p27 = 0;
                if (lane_id + 832 < PARTIAL_DIM) {
                    p27 = page_ptr[lane_id + 832];
                }
                #endif
                #if PARTIAL_DIM > 864
                float p28 = 0;
                if (lane_id + 864 < PARTIAL_DIM) {
                    p28 = page_ptr[lane_id + 864];
                }
                #endif
                #if PARTIAL_DIM > 896
                float p29 = 0;
                if (lane_id + 896 < PARTIAL_DIM) {
                    p29 = page_ptr[lane_id + 896];
                }
                #endif
                #if PARTIAL_DIM > 928
                float p30 = 0;
                if (lane_id + 928 < PARTIAL_DIM) {
                    p30 = page_ptr[lane_id + 928];
                }
                #endif
            // #endregion PARTIAL_DIM d_data loading

            //compute distance
            // #region PARTIAL_DIM distance computation
                #ifdef USE_L2_DIST_
                #if PARTIAL_DIM > 0
                    float delta1 = (p1 - q1_tail) * (p1 - q1_tail);
                #endif
                #if PARTIAL_DIM > 32
                    float delta2 = (p2 - q2_tail) * (p2 - q2_tail);
                #endif
                #if PARTIAL_DIM > 64
                    float delta3 = (p3 - q3_tail) * (p3 - q3_tail);
                #endif
                #if PARTIAL_DIM > 96
                    float delta4 = (p4 - q4_tail) * (p4 - q4_tail);
                #endif
                #if PARTIAL_DIM > 128
                    float delta5 = (p5 - q5_tail) * (p5 - q5_tail);
                #endif
                #if PARTIAL_DIM > 160
                    float delta6 = (p6 - q6_tail) * (p6 - q6_tail);
                #endif
                #if PARTIAL_DIM > 192
                    float delta7 = (p7 - q7_tail) * (p7 - q7_tail);
                #endif
                #if PARTIAL_DIM > 224
                    float delta8 = (p8 - q8_tail) * (p8 - q8_tail);
                #endif
                #if PARTIAL_DIM > 256
                    float delta9 = (p9 - q9_tail) * (p9 - q9_tail);
                #endif
                #if PARTIAL_DIM > 288
                    float delta10 = (p10 - q10_tail) * (p10 - q10_tail);
                #endif
                #if PARTIAL_DIM > 320
                    float delta11 = (p11 - q11_tail) * (p11 - q11_tail);
                #endif
                #if PARTIAL_DIM > 352
                    float delta12 = (p12 - q12_tail) * (p12 - q12_tail);
                #endif
                #if PARTIAL_DIM > 384
                    float delta13 = (p13 - q13_tail) * (p13 - q13_tail);
                #endif
                #if PARTIAL_DIM > 416
                    float delta14 = (p14 - q14_tail) * (p14 - q14_tail);
                #endif
                #if PARTIAL_DIM > 448
                    float delta15 = (p15 - q15_tail) * (p15 - q15_tail);
                #endif
                #if PARTIAL_DIM > 480
                    float delta16 = (p16 - q16_tail) * (p16 - q16_tail);
                #endif
                #if PARTIAL_DIM > 512
                    float delta17 = (p17 - q17_tail) * (p17 - q17_tail);
                #endif
                #if PARTIAL_DIM > 544
                    float delta18 = (p18 - q18_tail) * (p18 - q18_tail);
                #endif
                #if PARTIAL_DIM > 576
                    float delta19 = (p19 - q19_tail) * (p19 - q19_tail);
                #endif
                #if PARTIAL_DIM > 608
                    float delta20 = (p20 - q20_tail) * (p20 - q20_tail);
                #endif
                #if PARTIAL_DIM > 640
                    float delta21 = (p21 - q21_tail) * (p21 - q21_tail);
                #endif
                #if PARTIAL_DIM > 672
                    float delta22 = (p22 - q22_tail) * (p22 - q22_tail);
                #endif
                #if PARTIAL_DIM > 704
                    float delta23 = (p23 - q23_tail) * (p23 - q23_tail);
                #endif
                #if PARTIAL_DIM > 736
                    float delta24 = (p24 - q24_tail) * (p24 - q24_tail);
                #endif
                #if PARTIAL_DIM > 768
                    float delta25 = (p25 - q25_tail) * (p25 - q25_tail);
                #endif
                #if PARTIAL_DIM > 800
                    float delta26 = (p26 - q26_tail) * (p26 - q26_tail);
                #endif
                #if PARTIAL_DIM > 832
                    float delta27 = (p27 - q27_tail) * (p27 - q27_tail);
                #endif
                #if PARTIAL_DIM > 864
                    float delta28 = (p28 - q28_tail) * (p28 - q28_tail);
                #endif
                #if PARTIAL_DIM > 896
                    float delta29 = (p29 - q29_tail) * (p29 - q29_tail);
                #endif
                #if PARTIAL_DIM > 928
                    float delta30 = (p30 - q30_tail) * (p30 - q30_tail);
                #endif
                #endif
            // #endregion PARTIAL_DIM distance computation

            #ifdef COMPUTE_COUNT
            // 统计维度计算次数和距离计算次数
            if(lane_id == 0){
              atomicAdd(&compute_dim, (unsigned long long)PARTIAL_DIM);
            //   atomicAdd(&compute_dis, 1ULL);
            }
            #endif

            // #region PARTIAL_DIM distance reduction
                #ifdef USE_L2_DIST_
                    
                #if PARTIAL_DIM > 0
                    partial_dist += delta1;
                #endif
                #if PARTIAL_DIM > 32
                    partial_dist += delta2;
                #endif
                #if PARTIAL_DIM > 64
                    partial_dist += delta3;
                #endif
                #if PARTIAL_DIM > 96
                    partial_dist += delta4;
                #endif
                #if PARTIAL_DIM > 128
                    partial_dist += delta5;
                #endif
                #if PARTIAL_DIM > 160
                    partial_dist += delta6;
                #endif
                #if PARTIAL_DIM > 192
                    partial_dist += delta7;
                #endif
                #if PARTIAL_DIM > 224
                    partial_dist += delta8;
                #endif
                #if PARTIAL_DIM > 256
                    partial_dist += delta9;
                #endif
                #if PARTIAL_DIM > 288
                    partial_dist += delta10;
                #endif
                #if PARTIAL_DIM > 320
                    partial_dist += delta11;
                #endif
                #if PARTIAL_DIM > 352
                    partial_dist += delta12;
                #endif
                #if PARTIAL_DIM > 384
                    partial_dist += delta13;
                #endif
                #if PARTIAL_DIM > 416
                    partial_dist += delta14;
                #endif
                #if PARTIAL_DIM > 448
                    partial_dist += delta15;
                #endif
                #if PARTIAL_DIM > 480
                    partial_dist += delta16;
                #endif
                #if PARTIAL_DIM > 512
                    partial_dist += delta17;
                #endif
                #if PARTIAL_DIM > 544
                    partial_dist += delta18;
                #endif
                #if PARTIAL_DIM > 576
                    partial_dist += delta19;
                #endif
                #if PARTIAL_DIM > 608
                    partial_dist += delta20;
                #endif
                #if PARTIAL_DIM > 640
                    partial_dist += delta21;
                #endif
                #if PARTIAL_DIM > 672
                    partial_dist += delta22;
                #endif
                #if PARTIAL_DIM > 704
                    partial_dist += delta23;
                #endif
                #if PARTIAL_DIM > 736
                    partial_dist += delta24;
                #endif
                #if PARTIAL_DIM > 768
                    partial_dist += delta25;
                #endif
                #if PARTIAL_DIM > 800
                    partial_dist += delta26;
                #endif
                #if PARTIAL_DIM > 832
                    partial_dist += delta27;
                #endif
                #if PARTIAL_DIM > 864
                    partial_dist += delta28;
                #endif
                #if PARTIAL_DIM > 896
                    partial_dist += delta29;
                #endif
                #if PARTIAL_DIM > 928
                    partial_dist += delta30;
                #endif
                #endif
                #ifdef USE_L2_DIST_
                partial_dist += __shfl_down_sync(FULL_MASK, partial_dist, 16);
                partial_dist += __shfl_down_sync(FULL_MASK, partial_dist, 8);
                partial_dist += __shfl_down_sync(FULL_MASK, partial_dist, 4);
                partial_dist += __shfl_down_sync(FULL_MASK, partial_dist, 2);
                partial_dist += __shfl_down_sync(FULL_MASK, partial_dist, 1);
                #endif
            // #endregion PARTIAL_DIM distance reduction
            if(lane_id == 0){
                dist += partial_dist;
                if(PARTIAL_DIM < FULL_DIM - DIM && linear_dim > 0 && linear_w && linear_b){
                    int idx = (linear_dim < PARTIAL_DIM + DIM ? linear_dim : PARTIAL_DIM + DIM);
                    if (idx >= 0) {
                        dist = linear_w[idx] * dist + linear_b[idx];
                        dist = max(dist, 0.0f);
                    }
                }
                neighbors_array[n_candidates + i].first = dist;
            }
            // if(lane_id == 0){
            //     if (linear_dim > 0 && linear_w && linear_b) {
            //         int idx = (linear_dim < DIM ? linear_dim : DIM);
            //         if (idx >= 0) {
            //             dist = linear_w[idx] * dist + linear_b[idx];
            //             dist = max(dist, 0.0f);
            //         }
            //     }
            //     neighbors_array[n_candidates + i].first = dist;
            // }
        }
        else{
            if(lane_id == 0){
                if (linear_dim > 0 && linear_w && linear_b) {
                    int idx = (linear_dim < DIM ? linear_dim : DIM);
                    if (idx >= 0) {
                        dist = linear_w[idx] * dist + linear_b[idx];
                        dist = max(dist, 0.0f);
                    }
                }
                neighbors_array[n_candidates + i].first = dist;
            }
        }

      #else
        if(lane_id == 0){
            neighbors_array[n_candidates + i].first = dist;
            // if(q_id < 10) printf("n_candidates + i=%d dist=%f\n", n_candidates + i, dist);
        }
      #endif
    }
    __syncthreads();

    //sort neighbors
    step_id = 1;
    substep_id = 1;

    for(; step_id <= n_points_per_batch/2; step_id *= 2){
      substep_id = step_id;
      for(; substep_id >= 1; substep_id /= 2){
        for(int i = t_id; i < n_points_per_batch; i += blockSize){
          int unrollt_id = n_candidates + (i/substep_id) * 2 * substep_id + (i & (substep_id - 1));
          if(unrollt_id < n_candidates + n_points_per_batch && unrollt_id + substep_id < n_candidates + n_points_per_batch){
            if((i/step_id) % 2 == 0){
              if(neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first){
                tmp_neighbor = neighbors_array[unrollt_id];
                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                neighbors_array[unrollt_id + substep_id] = tmp_neighbor;
              }
            }
            else{
              if(neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first){
                tmp_neighbor = neighbors_array[unrollt_id];
                neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                neighbors_array[unrollt_id + substep_id] = tmp_neighbor;
              }
            }
          }
        }
      }
      __syncthreads();
    }
    __syncthreads();

    //merge
    for(int i = t_id; i < n_compare; i += blockSize){
      int unrollt_id = n_candidates - n_compare + i;
      if(unrollt_id < n_candidates){
        if(neighbors_array[unrollt_id + n_points_per_batch].first == MAX) continue;
        if(neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + n_points_per_batch].first){
          tmp_neighbor = neighbors_array[unrollt_id];
          neighbors_array[unrollt_id] = neighbors_array[unrollt_id + n_points_per_batch];
          neighbors_array[unrollt_id + n_points_per_batch] = tmp_neighbor;
        }
      }
    }
    __syncthreads();

    step_id = n_candidates / 2;
    substep_id = n_candidates / 2;
    for(; substep_id >= 1; substep_id /= 2){
      for(int i = t_id; i < n_candidates/2; i += blockSize){
        int unrollt_id = (i/substep_id) * 2 * substep_id + (i & (substep_id - 1));
        if(unrollt_id < n_candidates && unrollt_id + substep_id < n_candidates){
          if((i/step_id) % 2 == 0){
            if(neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first){
              tmp_neighbor = neighbors_array[unrollt_id];
              neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
              neighbors_array[unrollt_id + substep_id] = tmp_neighbor;
            }
          }
          else{
            if(neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first){
              tmp_neighbor = neighbors_array[unrollt_id];
              neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
              neighbors_array[unrollt_id + substep_id] = tmp_neighbor;
            }
          }
        }
      }
      __syncthreads();
    }
    __syncthreads();

    for(iteration = 0; iteration < (n_candidates + WARP_SIZE - 1) / WARP_SIZE; iteration++){
      int unrollt_id = lane_id + WARP_SIZE * iteration;
      crt_flag = 0;
      if(unrollt_id < n_candidates){
        if(neighbors_array[unrollt_id].second > 0 && neighbors_array[unrollt_id].second < np){
          crt_flag = 1;
        }
        else if(neighbors_array[unrollt_id].second == 0){
          if(check_zero == 0){
            check_zero = 1;
            crt_flag = 1;
          }
        }
      }
      tmp_flag = __ballot_sync(FULL_MASK, crt_flag);
      if(tmp_flag != 0){
        break;
      }
      else if(iteration == (n_candidates + WARP_SIZE - 1) / WARP_SIZE - 1){
        flag_all_blocks = 0;
      }
    }

    if(hash_iteration == max_hash_iteration){
      for(int i=t_id; i<SIZE; i+=blockSize)
        data[i] = 0;
      __syncthreads();

      for(int i=t_id; i<n_candidates * num_hash; i+=blockSize){
        int index = _abs(neighbors_array[i/num_hash].second);
        if(index < np){
          set_bit(hash_(i % num_hash, index, random_number), data);
        }
      }
      hash_iteration = 0;
    }
    hash_iteration++;
  }

// 重排序
  /*#ifdef REORDER
    if(DIM != DIM){
        for(int i = warp_id; i < n_candidates; i += n_warp){
        int p_id = _abs(neighbors_array[i].second);
        //   int p_id = neighbors_array[i].second;
        if(p_id >= np) continue;
        //read point
            #if DIM > 0
            float p1 = 0;
            if (lane_id < DIM){
                p1 = d_data[p_id * DIM + lane_id];
            }
            #endif
            #if DIM > 32
            float p2 = 0;
            if (lane_id + 32 < DIM) {
                p2 = d_data[p_id * DIM + lane_id + 32];
            }
            #endif
            #if DIM > 64
            float p3 = 0;
            if (lane_id + 64 < DIM) {
                p3 = d_data[p_id * DIM + lane_id + 64];
            }
            #endif
            #if DIM > 96
            float p4 = 0;
            if (lane_id + 96 < DIM) {
                p4 = d_data[p_id * DIM + lane_id + 96];
            }
            #endif
            #if DIM > 128
            float p5 = 0;
            if (lane_id + 128 < DIM) {
                p5 = d_data[p_id * DIM + lane_id + 128];
            }
            #endif
            #if DIM > 160
            float p6 = 0;
            if (lane_id + 160 < DIM) {
                p6 = d_data[p_id * DIM + lane_id + 160];
            }
            #endif
            #if DIM > 192
            float p7 = 0;
            if (lane_id + 192 < DIM) {
                p7 = d_data[p_id * DIM + lane_id + 192];
            }
            #endif
            #if DIM > 224
            float p8 = 0;
            if (lane_id + 224 < DIM) {
                p8 = d_data[p_id * DIM + lane_id + 224];
            }
            #endif
            #if DIM > 256
            float p9 = 0;
            if (lane_id + 256 < DIM) {
                p9 = d_data[p_id * DIM + lane_id + 256];
            }
            #endif
            #if DIM > 288
            float p10 = 0;
            if (lane_id + 288 < DIM) {
                p10 = d_data[p_id * DIM + lane_id + 288];
            }
            #endif
            #if DIM > 320
            float p11 = 0;
            if (lane_id + 320 < DIM) {
                p11 = d_data[p_id * DIM + lane_id + 320];
            }
            #endif
            #if DIM > 352
            float p12 = 0;
            if (lane_id + 352 < DIM) {
                p12 = d_data[p_id * DIM + lane_id + 352];
            }
            #endif
            #if DIM > 384
            float p13 = 0;
            if (lane_id + 384 < DIM) {
                p13 = d_data[p_id * DIM + lane_id + 384];
            }
            #endif
            #if DIM > 416
            float p14 = 0;
            if (lane_id + 416 < DIM) {
                p14 = d_data[p_id * DIM + lane_id + 416];
            }
            #endif
            #if DIM > 448
            float p15 = 0;
            if (lane_id + 448 < DIM) {
                p15 = d_data[p_id * DIM + lane_id + 448];
            }
            #endif
            #if DIM > 480
            float p16 = 0;
            if (lane_id + 480 < DIM) {
                p16 = d_data[p_id * DIM + lane_id + 480];
            }
            #endif
            #if DIM > 512
            float p17 = 0;
            if (lane_id + 512 < DIM) {
                p17 = d_data[p_id * DIM + lane_id + 512];
            }
            #endif
            #if DIM > 544
            float p18 = 0;
            if (lane_id + 544 < DIM) {
                p18 = d_data[p_id * DIM + lane_id + 544];
            }
            #endif
            #if DIM > 576
            float p19 = 0;
            if (lane_id + 576 < DIM) {
                p19 = d_data[p_id * DIM + lane_id + 576];
            }
            #endif
            #if DIM > 608
            float p20 = 0;
            if (lane_id + 608 < DIM) {
                p20 = d_data[p_id * DIM + lane_id + 608];
            }
            #endif
            #if DIM > 640
            float p21 = 0;
            if (lane_id + 640 < DIM) {
                p21 = d_data[p_id * DIM + lane_id + 640];
            }
            #endif
            #if DIM > 672
            float p22 = 0;
            if (lane_id + 672 < DIM) {
                p22 = d_data[p_id * DIM + lane_id + 672];
            }
            #endif
            #if DIM > 704
            float p23 = 0;
            if (lane_id + 704 < DIM) {
                p23 = d_data[p_id * DIM + lane_id + 704];
            }
            #endif
            #if DIM > 736
            float p24 = 0;
            if (lane_id + 736 < DIM) {
                p24 = d_data[p_id * DIM + lane_id + 736];
            }
            #endif
            #if DIM > 768
            float p25 = 0;
            if (lane_id + 768 < DIM) {
                p25 = d_data[p_id * DIM + lane_id + 768];
            }
            #endif
            #if DIM > 800
            float p26 = 0;
            if (lane_id + 800 < DIM) {
                p26 = d_data[p_id * DIM + lane_id + 800];
            }
            #endif
            #if DIM > 832
            float p27 = 0;
            if (lane_id + 832 < DIM) {
                p27 = d_data[p_id * DIM + lane_id + 832];
            }
            #endif
            #if DIM > 864
            float p28 = 0;
            if (lane_id + 864 < DIM) {
                p28 = d_data[p_id * DIM + lane_id + 864];
            }
            #endif
            #if DIM > 896
            float p29 = 0;
            if (lane_id + 896 < DIM) {
                p29 = d_data[p_id * DIM + lane_id + 896];
            }
            #endif
            #if DIM > 928
            float p30 = 0;
            if (lane_id + 928 < DIM) {
                p30 = d_data[p_id * DIM + lane_id + 928];
            }
            #endif

        //compute distance
            #ifdef USE_L2_DIST_
            #if DIM > 0
                float delta1 = (p1 - q1) * (p1 - q1);
            #endif
            #if DIM > 32
                float delta2 = (p2 - q2) * (p2 - q2);
            #endif
            #if DIM > 64
                float delta3 = (p3 - q3) * (p3 - q3);
            #endif
            #if DIM > 96
                float delta4 = (p4 - q4) * (p4 - q4);
            #endif
            #if DIM > 128
                float delta5 = (p5 - q5) * (p5 - q5);
            #endif
            #if DIM > 160
                float delta6 = (p6 - q6) * (p6 - q6);
            #endif
            #if DIM > 192
                float delta7 = (p7 - q7) * (p7 - q7);
            #endif
            #if DIM > 224
                float delta8 = (p8 - q8) * (p8 - q8);
            #endif
            #if DIM > 256
                float delta9 = (p9 - q9) * (p9 - q9);
            #endif
            #if DIM > 288
                float delta10 = (p10 - q10) * (p10 - q10);
            #endif
            #if DIM > 320
                float delta11 = (p11 - q11) * (p11 - q11);
            #endif
            #if DIM > 352
                float delta12 = (p12 - q12) * (p12 - q12);
            #endif
            #if DIM > 384
                float delta13 = (p13 - q13) * (p13 - q13);
            #endif
            #if DIM > 416
                float delta14 = (p14 - q14) * (p14 - q14);
            #endif
            #if DIM > 448
                float delta15 = (p15 - q15) * (p15 - q15);
            #endif
            #if DIM > 480
                float delta16 = (p16 - q16) * (p16 - q16);
            #endif
            #if DIM > 512
                float delta17 = (p17 - q17) * (p17 - q17);
            #endif
            #if DIM > 544
                float delta18 = (p18 - q18) * (p18 - q18);
            #endif
            #if DIM > 576
                float delta19 = (p19 - q19) * (p19 - q19);
            #endif
            #if DIM > 608
                float delta20 = (p20 - q20) * (p20 - q20);
            #endif
            #if DIM > 640
                float delta21 = (p21 - q21) * (p21 - q21);
            #endif
            #if DIM > 672
                float delta22 = (p22 - q22) * (p22 - q22);
            #endif
            #if DIM > 704
                float delta23 = (p23 - q23) * (p23 - q23);
            #endif
            #if DIM > 736
                float delta24 = (p24 - q24) * (p24 - q24);
            #endif
            #if DIM > 768
                float delta25 = (p25 - q25) * (p25 - q25);
            #endif
            #if DIM > 800
                float delta26 = (p26 - q26) * (p26 - q26);
            #endif
            #if DIM > 832
                float delta27 = (p27 - q27) * (p27 - q27);
            #endif
            #if DIM > 864
                float delta28 = (p28 - q28) * (p28 - q28);
            #endif
            #if DIM > 896
                float delta29 = (p29 - q29) * (p29 - q29);
            #endif
            #if DIM > 928
                float delta30 = (p30 - q30) * (p30 - q30);
            #endif
            #endif           
            #ifdef USE_L2_DIST_
                float dist = 0;
            #if DIM > 0
                dist += delta1;
            #endif
            #if DIM > 32
                dist += delta2;
            #endif
            #if DIM > 64
                dist += delta3;
            #endif
            #if DIM > 96
                dist += delta4;
            #endif
            #if DIM > 128
                dist += delta5;
            #endif
            #if DIM > 160
                dist += delta6;
            #endif
            #if DIM > 192
                dist += delta7;
            #endif
            #if DIM > 224
                dist += delta8;
            #endif
            #if DIM > 256
                dist += delta9;
            #endif
            #if DIM > 288
                dist += delta10;
            #endif
            #if DIM > 320
                dist += delta11;
            #endif
            #if DIM > 352
                dist += delta12;
            #endif
            #if DIM > 384
                dist += delta13;
            #endif
            #if DIM > 416
                dist += delta14;
            #endif
            #if DIM > 448
                dist += delta15;
            #endif
            #if DIM > 480
                dist += delta16;
            #endif
            #if DIM > 512
                dist += delta17;
            #endif
            #if DIM > 544
                dist += delta18;
            #endif
            #if DIM > 576
                dist += delta19;
            #endif
            #if DIM > 608
                dist += delta20;
            #endif
            #if DIM > 640
                dist += delta21;
            #endif
            #if DIM > 672
                dist += delta22;
            #endif
            #if DIM > 704
                dist += delta23;
            #endif
            #if DIM > 736
                dist += delta24;
            #endif
            #if DIM > 768
                dist += delta25;
            #endif
            #if DIM > 800
                dist += delta26;
            #endif
            #if DIM > 832
                dist += delta27;
            #endif
            #if DIM > 864
                dist += delta28;
            #endif
            #if DIM > 896
                dist += delta29;
            #endif
            #if DIM > 928
                dist += delta30;
            #endif
            #endif
            #ifdef USE_L2_DIST_
            dist += __shfl_down_sync(FULL_MASK, dist, 16);
            dist += __shfl_down_sync(FULL_MASK, dist, 8);
            dist += __shfl_down_sync(FULL_MASK, dist, 4);
            dist += __shfl_down_sync(FULL_MASK, dist, 2);
            dist += __shfl_down_sync(FULL_MASK, dist, 1);
            #endif
        // insert
            if(lane_id == 0){
            neighbors_array[i].first = dist;
            }
        }
        __syncthreads();
        //bitonic sort
        step_id = 1;
        substep_id = 1;

        for(; step_id <= n_candidates/2; step_id *= 2){
        substep_id = step_id;
        for(; substep_id >= 1; substep_id /= 2){
            for(int i = t_id; i < n_candidates; i += blockSize){
            int unrollt_id = (i/substep_id) * 2 * substep_id + (i & (substep_id - 1));
            if(unrollt_id < n_candidates && unrollt_id + substep_id < n_candidates){
                if((i/step_id) % 2 == 0){
                if(neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first){
                    tmp_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                    neighbors_array[unrollt_id + substep_id] = tmp_neighbor;
                }
                }
                else{
                if(neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first){
                    tmp_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                    neighbors_array[unrollt_id + substep_id] = tmp_neighbor;
                }
                }
            }
            }
        }
        __syncthreads();
        }
        __syncthreads();
    }
  #endif*/

  #ifdef REORDER
    for(int i=t_id; i<n_candidates; i+=blockSize){
      d_candidates[q_id * n_candidates + i] = _abs(neighbors_array[i].second);
    }
  #else
    for(int i=t_id; i<topk; i+=blockSize){
      crt_results[i] = _abs(neighbors_array[i].second);
    }
  #endif
}

/*template<typename IdType, typename FloatType, int WARP_SIZE>
__global__ void ReorderKernel(float* d_data, float* d_query, int* d_results, int* d_map, int* d_candidates, int n_candidates, int topk){
  int q_id = blockIdx.x;
  int b_id = blockIdx.x;
  int t_id = threadIdx.x;
  int lane_id = t_id % WARP_SIZE;
  int warp_id = t_id / WARP_SIZE;
  int n_warp = blockDim.x / WARP_SIZE;
  int blockSize = blockDim.x;

  extern __shared__ KernelPair<float, int> shared_memory_[];
  KernelPair<float, int>* neighbors_array = shared_memory_;

  // read d_query
    #if DIM > 0
        float q1 = 0;
        if (lane_id < DIM) {
            q1 = d_query[b_id * DIM + lane_id];
        }
    #endif
    #if DIM > 32
        float q2 = 0;
        if (lane_id + 32 < DIM) {
            q2 = d_query[b_id * DIM + lane_id + 32];
        }
    #endif
    #if DIM > 64
        float q3 = 0;
        if (lane_id + 64 < DIM) {
            q3 = d_query[b_id * DIM + lane_id + 64];
        }
    #endif
    #if DIM > 96
        float q4 = 0;
        if (lane_id + 96 < DIM) {
            q4 = d_query[b_id * DIM + lane_id + 96];
        }
    #endif
    #if DIM > 128
        float q5 = 0;
        if (lane_id + 128 < DIM) {
            q5 = d_query[b_id * DIM + lane_id + 128];
        }
    #endif
    #if DIM > 160
        float q6 = 0;
        if (lane_id + 160 < DIM) {
            q6 = d_query[b_id * DIM + lane_id + 160];
        }
    #endif
    #if DIM > 192
        float q7 = 0;
        if (lane_id + 192 < DIM) {
            q7 = d_query[b_id * DIM + lane_id + 192];
        }
    #endif
    #if DIM > 224
        float q8 = 0;
        if (lane_id + 224 < DIM) {
            q8 = d_query[b_id * DIM + lane_id + 224];
        }
    #endif
    #if DIM > 256
        float q9 = 0;
        if (lane_id + 256 < DIM) {
            q9 = d_query[b_id * DIM + lane_id + 256];
        }
    #endif
    #if DIM > 288
        float q10 = 0;
        if (lane_id + 288 < DIM) {
            q10 = d_query[b_id * DIM + lane_id + 288];
        }
    #endif
    #if DIM > 320
        float q11 = 0;
        if (lane_id + 320 < DIM) {
            q11 = d_query[b_id * DIM + lane_id + 320];
        }
    #endif
    #if DIM > 352
        float q12 = 0;
        if (lane_id + 352 < DIM) {
            q12 = d_query[b_id * DIM + lane_id + 352];
        }
    #endif
    #if DIM > 384
        float q13 = 0;
        if (lane_id + 384 < DIM) {
            q13 = d_query[b_id * DIM + lane_id + 384];
        }
    #endif
    #if DIM > 416
        float q14 = 0;
        if (lane_id + 416 < DIM) {
            q14 = d_query[b_id * DIM + lane_id + 416];
        }
    #endif
    #if DIM > 448
        float q15 = 0;
        if (lane_id + 448 < DIM) {
            q15 = d_query[b_id * DIM + lane_id + 448];
        }
    #endif
    #if DIM > 480
        float q16 = 0;
        if (lane_id + 480 < DIM) {
            q16 = d_query[b_id * DIM + lane_id + 480];
        }
    #endif
    #if DIM > 512
        float q17 = 0;
        if (lane_id + 512 < DIM) {
            q17 = d_query[b_id * DIM + lane_id + 512];
        }
    #endif
    #if DIM > 544
        float q18 = 0;
        if (lane_id + 544 < DIM) {
            q18 = d_query[b_id * DIM + lane_id + 544];
        }
    #endif
    #if DIM > 576
        float q19 = 0;
        if (lane_id + 576 < DIM) {
            q19 = d_query[b_id * DIM + lane_id + 576];
        }
    #endif
    #if DIM > 608
        float q20 = 0;
        if (lane_id + 608 < DIM) {
            q20 = d_query[b_id * DIM + lane_id + 608];
        }
    #endif
    #if DIM > 640
        float q21 = 0;
        if (lane_id + 640 < DIM) {
            q21 = d_query[b_id * DIM + lane_id + 640];
        }
    #endif
    #if DIM > 672
        float q22 = 0;
        if (lane_id + 672 < DIM) {
            q22 = d_query[b_id * DIM + lane_id + 672];
        }
    #endif
    #if DIM > 704
        float q23 = 0;
        if (lane_id + 704 < DIM) {
            q23 = d_query[b_id * DIM + lane_id + 704];
        }
    #endif
    #if DIM > 736
        float q24 = 0;
        if (lane_id + 736 < DIM) {
            q24 = d_query[b_id * DIM + lane_id + 736];
        }
    #endif
    #if DIM > 768
        float q25 = 0;
        if (lane_id + 768 < DIM) {
            q25 = d_query[b_id * DIM + lane_id + 768];
        }
    #endif
    #if DIM > 800
        float q26 = 0;
        if (lane_id + 800 < DIM) {
            q26 = d_query[b_id * DIM + lane_id + 800];
        }
    #endif
    #if DIM > 832
        float q27 = 0;
        if (lane_id + 832 < DIM) {
            q27 = d_query[b_id * DIM + lane_id + 832];
        }
    #endif
    #if DIM > 864
        float q28 = 0;
        if (lane_id + 864 < DIM) {
            q28 = d_query[b_id * DIM + lane_id + 864];
        }
    #endif
    #if DIM > 896
        float q29 = 0;
        if (lane_id + 896 < DIM) {
            q29 = d_query[b_id * DIM + lane_id + 896];
        }
    #endif
    #if DIM > 928
        float q30 = 0;
        if (lane_id + 928 < DIM) {
            q30 = d_query[b_id * DIM + lane_id + 928];
        }
    #endif

  //初始化neighbors_array
  for(int i = t_id; i < n_candidates; i += blockSize){
    neighbors_array[i].first = MAX;
    neighbors_array[i].second = d_candidates[q_id * n_candidates + i];
  }
  __syncthreads();

  for(int i = warp_id; i < n_candidates; i += n_warp){
    int p_id = d_map[neighbors_array[i].second];
    // read point
      #if DIM > 0
      float p1 = 0;
      if (lane_id < DIM) {
          p1 = d_data[p_id * DIM + lane_id];
      }
      #endif
      #if DIM > 32
      float p2 = 0;
      if (lane_id + 32 < DIM) {
          p2 = d_data[p_id * DIM + lane_id + 32];
      }
      #endif
      #if DIM > 64
      float p3 = 0;
      if (lane_id + 64 < DIM) {
          p3 = d_data[p_id * DIM + lane_id + 64];
      }
      #endif
      #if DIM > 96
      float p4 = 0;
      if (lane_id + 96 < DIM) {
          p4 = d_data[p_id * DIM + lane_id + 96];
      }
      #endif
      #if DIM > 128
      float p5 = 0;
      if (lane_id + 128 < DIM) {
          p5 = d_data[p_id * DIM + lane_id + 128];
      }
      #endif
      #if DIM > 160
      float p6 = 0;
      if (lane_id + 160 < DIM) {
          p6 = d_data[p_id * DIM + lane_id + 160];
      }
      #endif
      #if DIM > 192
      float p7 = 0;
      if (lane_id + 192 < DIM) {
          p7 = d_data[p_id * DIM + lane_id + 192];
      }
      #endif
      #if DIM > 224
      float p8 = 0;
      if (lane_id + 224 < DIM) {
          p8 = d_data[p_id * DIM + lane_id + 224];
      }
      #endif
      #if DIM > 256
      float p9 = 0;
      if (lane_id + 256 < DIM) {
          p9 = d_data[p_id * DIM + lane_id + 256];
      }
      #endif
      #if DIM > 288
      float p10 = 0;
      if (lane_id + 288 < DIM) {
          p10 = d_data[p_id * DIM + lane_id + 288];
      }
      #endif
      #if DIM > 320
      float p11 = 0;
      if (lane_id + 320 < DIM) {
          p11 = d_data[p_id * DIM + lane_id + 320];
      }
      #endif
      #if DIM > 352
      float p12 = 0;
      if (lane_id + 352 < DIM) {
          p12 = d_data[p_id * DIM + lane_id + 352];
      }
      #endif
      #if DIM > 384
      float p13 = 0;
      if (lane_id + 384 < DIM) {
          p13 = d_data[p_id * DIM + lane_id + 384];
      }
      #endif
      #if DIM > 416
      float p14 = 0;
      if (lane_id + 416 < DIM) {
          p14 = d_data[p_id * DIM + lane_id + 416];
      }
      #endif
      #if DIM > 448
      float p15 = 0;
      if (lane_id + 448 < DIM) {
          p15 = d_data[p_id * DIM + lane_id + 448];
      }
      #endif
      #if DIM > 480
      float p16 = 0;
      if (lane_id + 480 < DIM) {
          p16 = d_data[p_id * DIM + lane_id + 480];
      }
      #endif
      #if DIM > 512
      float p17 = 0;
      if (lane_id + 512 < DIM) {
          p17 = d_data[p_id * DIM + lane_id + 512];
      }
      #endif
      #if DIM > 544
      float p18 = 0;
      if (lane_id + 544 < DIM) {
          p18 = d_data[p_id * DIM + lane_id + 544];
      }
      #endif
      #if DIM > 576
      float p19 = 0;
      if (lane_id + 576 < DIM) {
          p19 = d_data[p_id * DIM + lane_id + 576];
      }
      #endif
      #if DIM > 608
      float p20 = 0;
      if (lane_id + 608 < DIM) {
          p20 = d_data[p_id * DIM + lane_id + 608];
      }
      #endif
      #if DIM > 640
      float p21 = 0;
      if (lane_id + 640 < DIM) {
          p21 = d_data[p_id * DIM + lane_id + 640];
      }
      #endif
      #if DIM > 672
      float p22 = 0;
      if (lane_id + 672 < DIM) {
          p22 = d_data[p_id * DIM + lane_id + 672];
      }
      #endif
      #if DIM > 704
      float p23 = 0;
      if (lane_id + 704 < DIM) {
          p23 = d_data[p_id * DIM + lane_id + 704];
      }
      #endif
      #if DIM > 736
      float p24 = 0;
      if (lane_id + 736 < DIM) {
          p24 = d_data[p_id * DIM + lane_id + 736];
      }
      #endif
      #if DIM > 768
      float p25 = 0;
      if (lane_id + 768 < DIM) {
          p25 = d_data[p_id * DIM + lane_id + 768];
      }
      #endif
      #if DIM > 800
      float p26 = 0;
      if (lane_id + 800 < DIM) {
          p26 = d_data[p_id * DIM + lane_id + 800];
      }
      #endif
      #if DIM > 832
      float p27 = 0;
      if (lane_id + 832 < DIM) {
          p27 = d_data[p_id * DIM + lane_id + 832];
      }
      #endif
      #if DIM > 864
      float p28 = 0;
      if (lane_id + 864 < DIM) {
          p28 = d_data[p_id * DIM + lane_id + 864];
      }
      #endif
      #if DIM > 896
      float p29 = 0;
      if (lane_id + 896 < DIM) {
          p29 = d_data[p_id * DIM + lane_id + 896];
      }
      #endif
      #if DIM > 928
      float p30 = 0;
      if (lane_id + 928 < DIM) {
          p30 = d_data[p_id * DIM + lane_id + 928];
      }
      #endif

    // compute distance
      #ifdef USE_L2_DIST_
      #if DIM > 0
          float delta1 = (p1 - q1) * (p1 - q1);
      #endif
      #if DIM > 32
          float delta2 = (p2 - q2) * (p2 - q2);
      #endif
      #if DIM > 64
          float delta3 = (p3 - q3) * (p3 - q3);
      #endif
      #if DIM > 96
          float delta4 = (p4 - q4) * (p4 - q4);
      #endif
      #if DIM > 128
          float delta5 = (p5 - q5) * (p5 - q5);
      #endif
      #if DIM > 160
          float delta6 = (p6 - q6) * (p6 - q6);
      #endif
      #if DIM > 192
          float delta7 = (p7 - q7) * (p7 - q7);
      #endif
      #if DIM > 224
          float delta8 = (p8 - q8) * (p8 - q8);
      #endif
      #if DIM > 256
          float delta9 = (p9 - q9) * (p9 - q9);
      #endif
      #if DIM > 288
          float delta10 = (p10 - q10) * (p10 - q10);
      #endif
      #if DIM > 320
          float delta11 = (p11 - q11) * (p11 - q11);
      #endif
      #if DIM > 352
          float delta12 = (p12 - q12) * (p12 - q12);
      #endif
      #if DIM > 384
          float delta13 = (p13 - q13) * (p13 - q13);
      #endif
      #if DIM > 416
          float delta14 = (p14 - q14) * (p14 - q14);
      #endif
      #if DIM > 448
          float delta15 = (p15 - q15) * (p15 - q15);
      #endif
      #if DIM > 480
          float delta16 = (p16 - q16) * (p16 - q16);
      #endif
      #if DIM > 512
          float delta17 = (p17 - q17) * (p17 - q17);
      #endif
      #if DIM > 544
          float delta18 = (p18 - q18) * (p18 - q18);
      #endif
      #if DIM > 576
          float delta19 = (p19 - q19) * (p19 - q19);
      #endif
      #if DIM > 608
          float delta20 = (p20 - q20) * (p20 - q20);
      #endif
      #if DIM > 640
          float delta21 = (p21 - q21) * (p21 - q21);
      #endif
      #if DIM > 672
          float delta22 = (p22 - q22) * (p22 - q22);
      #endif
      #if DIM > 704
          float delta23 = (p23 - q23) * (p23 - q23);
      #endif
      #if DIM > 736
          float delta24 = (p24 - q24) * (p24 - q24);
      #endif
      #if DIM > 768
          float delta25 = (p25 - q25) * (p25 - q25);
      #endif
      #if DIM > 800
          float delta26 = (p26 - q26) * (p26 - q26);
      #endif
      #if DIM > 832
          float delta27 = (p27 - q27) * (p27 - q27);
      #endif
      #if DIM > 864
          float delta28 = (p28 - q28) * (p28 - q28);
      #endif
      #if DIM > 896
          float delta29 = (p29 - q29) * (p29 - q29);
      #endif
      #if DIM > 928
          float delta30 = (p30 - q30) * (p30 - q30);
      #endif
      #endif           

    // reduce
      #ifdef USE_L2_DIST_
          float dist = 0;
      #if DIM > 0
          dist += delta1;
      #endif
      #if DIM > 32
          dist += delta2;
      #endif
      #if DIM > 64
          dist += delta3;
      #endif
      #if DIM > 96
          dist += delta4;
      #endif
      #if DIM > 128
          dist += delta5;
      #endif
      #if DIM > 160
          dist += delta6;
      #endif
      #if DIM > 192
          dist += delta7;
      #endif
      #if DIM > 224
          dist += delta8;
      #endif
      #if DIM > 256
          dist += delta9;
      #endif
      #if DIM > 288
          dist += delta10;
      #endif
      #if DIM > 320
          dist += delta11;
      #endif
      #if DIM > 352
          dist += delta12;
      #endif
      #if DIM > 384
          dist += delta13;
      #endif
      #if DIM > 416
          dist += delta14;
      #endif
      #if DIM > 448
          dist += delta15;
      #endif
      #if DIM > 480
          dist += delta16;
      #endif
      #if DIM > 512
          dist += delta17;
      #endif
      #if DIM > 544
          dist += delta18;
      #endif
      #if DIM > 576
          dist += delta19;
      #endif
      #if DIM > 608
          dist += delta20;
      #endif
      #if DIM > 640
          dist += delta21;
      #endif
      #if DIM > 672
          dist += delta22;
      #endif
      #if DIM > 704
          dist += delta23;
      #endif
      #if DIM > 736
          dist += delta24;
      #endif
      #if DIM > 768
          dist += delta25;
      #endif
      #if DIM > 800
          dist += delta26;
      #endif
      #if DIM > 832
          dist += delta27;
      #endif
      #if DIM > 864
          dist += delta28;
      #endif
      #if DIM > 896
          dist += delta29;
      #endif
      #if DIM > 928
          dist += delta30;
      #endif
      #endif
      #ifdef USE_L2_DIST_
      dist += __shfl_down_sync(FULL_MASK, dist, 16);
      dist += __shfl_down_sync(FULL_MASK, dist, 8);
      dist += __shfl_down_sync(FULL_MASK, dist, 4);
      dist += __shfl_down_sync(FULL_MASK, dist, 2);
      dist += __shfl_down_sync(FULL_MASK, dist, 1);
      #endif

    // insert
      if(lane_id == 0){
        neighbors_array[i].first = finalize_distance(dist,
          static_cast<int>(p_id), q_id,
          query_top_clusters, cluster_top_t,
          point_infos, cluster_to_page,
          cache_data, page_size, dim_partial, dim_total,
          query_full, linear_w, linear_b, linear_dim);
      }
    }
    __syncthreads();

  // bitonic sort
  int step_id = 1;
  int substep_id = 1;

  KernelPair<float, int> tmp_neighbor;

  for(; step_id <= n_candidates/2; step_id *= 2){
    substep_id = step_id;

    for(; substep_id >= 1; substep_id /= 2){
      // for(int temparory_id = 0; temparory_id < (n_points_per_batch/2 + blockSize - 1)/blockSize; )
      for(int i=t_id; i<n_candidates; i+=blockSize){
        int unrollt_id = (i/substep_id) * 2 * substep_id + (i&(substep_id-1));
        if(unrollt_id < n_candidates && unrollt_id + substep_id < n_candidates){
          if((i/step_id)%2 == 0){
            if(neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first){
              tmp_neighbor = neighbors_array[unrollt_id];
              neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
              neighbors_array[unrollt_id + substep_id] = tmp_neighbor;
            }
          }
          else{
            if(neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first){
              tmp_neighbor = neighbors_array[unrollt_id];
              neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
              neighbors_array[unrollt_id + substep_id] = tmp_neighbor;
            }
          }
        }
      }
    }
    __syncthreads();
  }
  __syncthreads();

  for(int i=t_id; i<topk; i+=blockSize){
    d_results[q_id * topk + i] = neighbors_array[i].second;
  }
}*/