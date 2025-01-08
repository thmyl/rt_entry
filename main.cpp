#include <iostream>
#include <cstdio>
#include "graph.h"
#include "matrix.h"

void SetDevice(int device_id=0){
    int device_count=0;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop,device_id);
    #ifdef DETAIL
      printf("Maximum dimensions of grid size: (%d, %d, %d)\n",
            device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
      printf("Maximum dimensions of block size: (%d, %d, %d)\n",
            device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
    #endif
    size_t available_memory,total_memory;
    cudaMemGetInfo(&available_memory,&total_memory);
    // std::cout<<"==========================================================\n";
    std::cout<<"Total GPUs visible: "<<device_count;
    std::cout<<", using ["<<device_id<<"]: "<<device_prop.name<<std::endl;
    std::cout<<"Available Memory: "<<int(available_memory/1024/1024)<<" MB, ";
    std::cout<<"Total Memory: "<<int(total_memory/1024/1024)<<" MB\n";
}

void check_gpu_memory() {
  size_t free_memory, total_memory;
  cudaMemGetInfo(&free_memory, &total_memory);
  
  std::cout << "Free memory: " << free_memory / 1024 / 1024 << " MB" << std::endl;
  std::cout << "Total memory: " << total_memory / 1024 / 1024 << " MB" << std::endl;
  std::cout << "Used memory: " << (total_memory - free_memory) / 1024 / 1024 << " MB" << std::endl;

  std::ofstream outfile;
  outfile.open(OUTFILE, std::ios_base::app);
  outfile <<  "Used memory: " << (total_memory - free_memory) / 1024 / 1024 << " MB\n" << std::flush;
  outfile.close();
}

int main(int argc, char **argv){
  // freopen("out.txt", "w", stdout);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  #ifdef DETAIL
    std::cout<<"buffer_size = "<<FLAGS_buffer_size<<std::endl;
    std::cout<<"n_subspaces = "<<FLAGS_n_subspaces<<std::endl;
    std::cout<<"data_name = "<<FLAGS_data_name<<std::endl;
    std::cout<<"data_path = "<<FLAGS_data_path<<std::endl;
    std::cout<<"query_path = "<<FLAGS_query_path<<std::endl;
    std::cout<<"gt_path = "<<FLAGS_gt_path<<std::endl;
    std::cout<<"graph_path = "<<FLAGS_graph_path<<std::endl;
    std::cout<<"n_candidates = "<<FLAGS_n_candidates<<std::endl;
    std::cout<<"max_hits = "<<FLAGS_max_hits<<std::endl;
    std::cout<<"expand_ratio = "<<FLAGS_expand_ratio<<std::endl;
    std::cout<<"point_ratio = "<<FLAGS_point_ratio<<std::endl;
    std::cout<<"topk = "<<FLAGS_topk<<std::endl;
  #endif

  std::ofstream outfile;
  outfile.open(OUTFILE, std::ios_base::app);
  // outfile <<  "\n---------- expand_ratio = " << FLAGS_expand_ratio << "\t" << "point_ratio = " << FLAGS_point_ratio <<" ----------\n\n" << std::flush;
  int reorder = 0;
  #ifdef REORDER
    reorder = 1;
  #endif
  outfile <<  "\n---------- n_candidates = " << FLAGS_n_candidates << "\t" 
                        << "graph_path = " << FLAGS_graph_path << "\t" 
                        << "DIM = " << DIM << "\t" 
                        << "REORDER = " << reorder <<" ----------\n\n" << std::flush;
  outfile.close();

  char* datafile = (char*)FLAGS_data_path.c_str();
  char* queryfile = (char*)FLAGS_query_path.c_str();
  char* gtfile = (char*)FLAGS_gt_path.c_str();

  SetDevice(3);

  Graph graph(FLAGS_n_subspaces, FLAGS_buffer_size, FLAGS_n_candidates, FLAGS_max_hits, FLAGS_expand_ratio, FLAGS_point_ratio,
              FLAGS_data_name, FLAGS_data_path, FLAGS_query_path, FLAGS_gt_path, FLAGS_graph_path, FLAGS_ALGO, FLAGS_search_width, FLAGS_topk);
  graph.Input();
  graph.RB_Graph();
  if(FLAGS_ALGO!=0) {
    graph.Projection();
  }
  if(FLAGS_ALGO==1) {
    graph.Init_entry();
  }
  graph.Search();
  check_gpu_memory();
  graph.CleanUp();

  return 0;
}