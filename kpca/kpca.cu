#include <kpca.h>

KPCA::KPCA(){}

KPCA::KPCA(float* data_, uint _nb, uint _nq, uint _dim, uint _n_clusters, uint _n_components, uint _n_iteration) {
  n_clusters = _n_clusters;
  n_components = _n_components;
  nb = _nb;
  nq = _nq;
  dim = _dim;
  n_iteration = _n_iteration;
  pcas.resize(n_clusters);
  data = data_;
  lists.resize(n_clusters);
  B.resize(nb, dim);
  B_pca.resize(nb, n_components);
  B_reconstructed.resize(nb, dim);

  for(int i=0; i<nb; i++){
    for(int j=0; j<dim; j++){
      B(i, j) = data[i*dim + j];
    }
  }
}

float find_min(const Eigen::MatrixXd& _B, uint j){
  float res = _B(0, j);
  for(int i=1; i<_B.rows(); i++){
    res = min(res, _B(i, j));
  }
  return res;
}

float find_max(const Eigen::MatrixXd& _B, uint j){
  float res = _B(0, j);
  for(int i=1; i<_B.rows(); i++){
    res = max(res, _B(i, j));
  }
  return res;
}

void KPCA::pca_split(PCA& pca1, PCA& pca2, std::vector<uint>& list1, std::vector<uint>& list2){
  float min_ = find_min(pca1.B_res, n_components);
  float max_ = find_max(pca1.B_res, n_components);
  float mid_ = (min_ + max_) / 2.0;
  
  std::vector<uint> tmp_list(list1);
  list1.resize(0);
  list2.resize(0);
  for(uint i=0; i<tmp_list.size(); i++){
    uint id = tmp_list[i];
    if(pca1.B_res(i, n_components) <= mid_) list1.push_back(id);
    else list2.push_back(id);
  }

  pca1.reset(B, dim, list1);
  pca1.calc_result(n_components + 1);
  pca2.reset(B, dim, list2);
  pca2.calc_result(n_components + 1);
}

void KPCA::pca_tree(){
  std::vector<float> ratio_list(n_clusters);
  uint cluster_count = 1;

  //初始化根节点
  lists[0].resize(nb);
  for(int i=0; i<nb; i++) lists[0][i] = i;
  pcas[0].reset(B, dim, lists[0]);
  ratio_list[0] = pcas[0].Ratio(n_components);
  pcas[0].calc_result(n_components + 1);

  while(cluster_count < n_clusters){
    //找需要split的cluster
    float min_ratio = ratio_list[0];
    uint split_cluster_id = 0;
    for(int i=1; i<cluster_count; i++){
      if(ratio_list[i] < min_ratio){
        min_ratio = ratio_list[i];
        split_cluster_id = i;
      }
    }
    //进行split
    pca_split(pcas[split_cluster_id], pcas[cluster_count], lists[split_cluster_id], lists[cluster_count]);
    ratio_list[split_cluster_id] = pcas[split_cluster_id].Ratio(n_components);
    ratio_list[cluster_count] = pcas[cluster_count].Ratio(n_components);

    cluster_count ++;

    for(int i=0; i<cluster_count; i++)printf("%d ",lists[i].size());
    printf("\n");
    for(int i=0; i<cluster_count; i++)printf("%f ",ratio_list[i]);
    printf("\n----------------\n");
  }
  printf("finish init\n");
}

void KPCA::InitLists(uint method){
  printf("InitLists\n");
  if(method == 0){
    uint n_points_per_cluster = nb/n_clusters;
    uint rest = nb % n_clusters;
    uint count = 0;
    for(int i=0; i<n_clusters; i++){
      if(i < rest) lists[i].resize(n_points_per_cluster + 1);
      else lists[i].resize(n_points_per_cluster);
      for(int j=0; j<lists[i].size(); j++){
        lists[i][j] = count + j;
      }
      count += lists[i].size();
    }
  }
  else if(method == 1){
    pca_tree();
  }
}

void KPCA::SetPcas(){
  for (int i = 0; i < n_clusters; i++) {
    // if(lists[i].size() > 1) 
    pcas[i].reset(B, dim, lists[i]);
  }
}

void KPCA::Output(){
  for(int i=0; i<n_clusters; i++){
    printf("%d ", lists[i].size());
  }
  printf("\n");
  for(int i=0; i<n_clusters; i++){
    printf("%f ", pcas[i].Ratio(n_components));
  }
  printf("\n");
}

void KPCA::Learn(){
  SetPcas();
  Output();

  uint iter = 0;
  float *min_dist = new float[nb];
  uint *min_id = new uint[nb];
  while(iter < n_iteration){
    iter++;
    printf("第 %d 次迭代\n", iter);
    for(int i=0; i<n_clusters; i++){
      lists[i].resize(0);
    }
    // for(uint cluster_id = 0; cluster_id < n_clusters; cluster_id++){
    for(int cluster_id = n_clusters-1; cluster_id >= 0; cluster_id--){
      //pca变换
      pcas[cluster_id].pca_transform(B, B_pca, n_components);
      //pca逆变换
      pcas[cluster_id].pca_inverse_transform(B_pca, B_reconstructed, n_components);
      
      //计算重构误差
      for(int i=0; i<nb; i++){
        float dist = (B.row(i) - B_reconstructed.row(i)).squaredNorm();
        if(cluster_id==n_clusters-1 || dist < min_dist[i]){
          min_id[i] = cluster_id;
          min_dist[i] = dist;
        }
      }
    }
    for(int i=0; i<nb; i++){
      lists[min_id[i]].push_back(i);
    }
    SetPcas();
    Output();
  }
  printf("迭代完成\n");
}

void KPCA::output_lists(const char* filename){
  uint* out_list = new uint[nb];
  for(int i=0; i<n_clusters; i++){
    for(int j=0; j<lists[i].size(); j++){
      out_list[lists[i][j]] = i;
    }
  }
  FILE* fp = fopen(filename, "w");
  for(int i=0; i<nb; i++){
    fprintf(fp, "%d\n", out_list[i]);
  }
  fclose(fp);
  delete[] out_list;
}

void KPCA::SaveLearnedData(){
  printf("Saving learned data...\n");
  for(int i=0; i<n_clusters; i++){
    pcas[i].calc_result(n_components);
  }
  h_cluster_lists.resize(n_clusters);
  d_cluster_lists.resize(n_clusters);
  h_rotation.resize(n_clusters);
  d_rotation.resize(n_clusters);
  h_means.resize(n_clusters);
  d_means.resize(n_clusters);
  h_transforms.resize(n_clusters);
  d_transforms.resize(n_clusters);
  h_reconstructed.resize(n_clusters);
  d_reconstructed.resize(n_clusters);
  h_t.resize(n_clusters);
  d_t.resize(n_clusters);
  h_pca_bases.resize(n_clusters);
  d_pca_bases.resize(n_clusters);

  std::vector<thrust::device_vector<float> > d_tmp(n_clusters);

  for(int cluster_id=0; cluster_id<n_clusters; cluster_id++){
    h_cluster_lists[cluster_id].resize(lists[cluster_id].size());
    for(int i=0; i<lists[cluster_id].size(); i++){
      h_cluster_lists[cluster_id][i] = lists[cluster_id][i];
    }
    d_cluster_lists[cluster_id].resize(lists[cluster_id].size());
    thrust::copy(h_cluster_lists[cluster_id].begin(), h_cluster_lists[cluster_id].end(), d_cluster_lists[cluster_id].begin());

    h_rotation[cluster_id].resize(dim*n_components);
    for(int i=0; i<dim; i++){
      for(int j=0; j<n_components; j++){
        h_rotation[cluster_id][i*n_components+j] = pcas[cluster_id].vec(i,j);
      }
    }
    d_rotation[cluster_id].resize(dim*n_components);
    thrust::copy(h_rotation[cluster_id].begin(), h_rotation[cluster_id].end(), d_rotation[cluster_id].begin());

    h_means[cluster_id].resize(dim);
    for(int i=0; i<dim; i++){
      h_means[cluster_id][i] = pcas[cluster_id].meanvecRow(i);
    }
    d_means[cluster_id].resize(dim);
    thrust::copy(h_means[cluster_id].begin(), h_means[cluster_id].end(), d_means[cluster_id].begin());

    h_t[cluster_id].resize(n_components*dim);
    for(int i=0; i<n_components; i++){
      for(int j=0; j<dim; j++){
        h_t[cluster_id][i*dim+j] = pcas[cluster_id].vec(j,i);
      }
    }
    d_t[cluster_id].resize(n_components*dim);
    thrust::copy(h_t[cluster_id].begin(), h_t[cluster_id].end(), d_t[cluster_id].begin());

    h_pca_bases[cluster_id].resize(lists[cluster_id].size()*n_components);
    for(int i=0; i<lists[cluster_id].size(); i++){
      for(int j=0; j<n_components; j++){
        h_pca_bases[cluster_id][i*n_components+j] = pcas[cluster_id].B_res(i,j);
      }
    }
    d_pca_bases[cluster_id].resize(lists[cluster_id].size()*n_components);
    thrust::copy(h_pca_bases[cluster_id].begin(), h_pca_bases[cluster_id].end(), d_pca_bases[cluster_id].begin());

    // h_transforms[cluster_id].resize(nq*dim);
    // for(int i=0; i<nq; i++){
    //   for(int j=0; j<dim; j++){
    //     h_transforms[cluster_id][i*dim + j] = h_means[cluster_id][j];
    //   }
    // }
    // d_transforms[cluster_id].resize(dim*n_components);
    // // thrust::copy(h_transforms[cluster_id].begin(), h_transforms[cluster_id].end(), d_transforms[cluster_id].begin());
    // d_tmp[cluster_id].resize(dim*n_components);
    // thrust::copy(h_transforms[cluster_id].begin(), h_transforms[cluster_id].end(), d_tmp[cluster_id].begin());

    // h_reconstructed[cluster_id].resize(nq*dim);
    // d_reconstructed[cluster_id].resize(nq*dim);
    // thrust::copy(h_transforms[cluster_id].begin(), h_transforms[cluster_id].end(), h_reconstructed[cluster_id].begin());
    // thrust::copy(d_tmp[cluster_id].begin(), d_tmp[cluster_id].end(), d_reconstructed[cluster_id].begin());

    h_reconstructed[cluster_id].resize(nq*dim);
    for(int i=0; i<nq; i++){
      for(int j=0; j<dim; j++){
        h_reconstructed[cluster_id][i*dim + j] = h_means[cluster_id][j];
      }
    }
    d_reconstructed[cluster_id].resize(nq * dim);
    thrust::copy(h_reconstructed[cluster_id].begin(), h_reconstructed[cluster_id].end(), d_reconstructed[cluster_id].begin());
  }

  //mr = mean * rotation
  float alpha = 1.0, beta = 0.0;
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "!!!! CUBLAS initialization error\n";
    return;
  }
  for(int cluster_id=0; cluster_id<n_clusters; cluster_id++){
    d_transforms[cluster_id].resize(nq*n_components);
    thrust::fill(d_transforms[cluster_id].begin(), d_transforms[cluster_id].end(), 0.0f);
    matrixMultiply(handle, d_reconstructed[cluster_id], d_rotation[cluster_id], d_transforms[cluster_id], nq, n_components, dim, alpha, beta);
  }
  cublasDestroy(handle);

  pcas.resize(0);
  lists.resize(0);
  B.resize(0,0);
  B_pca.resize(0,0);
  B_reconstructed.resize(0,0);
}

void KPCA::WriteCluster(const char* filename){
  printf("Writing cluster to %s...\n", filename);
  FILE* fp = fopen(filename, "wb");
  fwrite(&n_clusters, sizeof(uint), 1, fp);
  fwrite(&dim, sizeof(uint), 1, fp);
  fwrite(&n_components, sizeof(uint), 1, fp);
  for(int cluster_id=0; cluster_id < n_clusters; cluster_id++){
    uint list_size = h_cluster_lists[cluster_id].size();
    fwrite(&list_size, sizeof(uint), 1, fp);
    uint* list_ptr = thrust::raw_pointer_cast(h_cluster_lists[cluster_id].data());
    fwrite(list_ptr, sizeof(uint), list_size, fp);
  }
  for(int cluster_id=0; cluster_id < n_clusters; cluster_id++){
    float* mean_ptr = thrust::raw_pointer_cast(h_means[cluster_id].data());
    fwrite(mean_ptr, sizeof(float), dim, fp);
  }
  for(int cluster_id=0; cluster_id < n_clusters; cluster_id++){
    float* rotation_ptr = thrust::raw_pointer_cast(h_rotation[cluster_id].data());
    fwrite(rotation_ptr, sizeof(float), dim*n_components, fp);
  }
  for(int cluster_id=0; cluster_id < n_clusters; cluster_id++){
    float* t_ptr = thrust::raw_pointer_cast(h_t[cluster_id].data());
    fwrite(t_ptr, sizeof(float), n_components*dim, fp);
  }
  for(int cluster_id=0; cluster_id < n_clusters; cluster_id++){
    float* pca_base_ptr = thrust::raw_pointer_cast(h_pca_bases[cluster_id].data());
    fwrite(pca_base_ptr, sizeof(float), h_pca_bases[cluster_id].size(), fp);
  }
  fclose(fp);
}

/*void KPCA::ReadCluster(const char* filename){
  FILE* fp = fopen(filename, "rb");
  fread(&n_clusters, sizeof(uint), 1, fp);
  fread(&dim, sizeof(uint), 1, fp);
  fread(&n_components, sizeof(uint), 1, fp);
  h_cluster_lists.resize(n_clusters);
  d_cluster_lists.resize(n_clusters);
  h_rotation.resize(n_clusters);
  d_rotation.resize(n_clusters);
  h_means.resize(n_clusters);
  d_means.resize(n_clusters);
  h_transforms.resize(n_clusters);
  d_transforms.resize(n_clusters);
  h_reconstructed.resize(n_clusters);
  d_reconstructed.resize(n_clusters);
  h_t.resize(n_clusters);
  d_t.resize(n_clusters);
  h_pca_bases.resize(n_clusters);
  d_pca_bases.resize(n_clusters);

  std::vector<thrust::device_vector<float> > d_tmp(n_clusters);

  for(int cluster_id=0; cluster_id < n_clusters; cluster_id++){
    uint list_size;
    fread(&list_size, sizeof(uint), 1, fp);
    h_cluster_lists[cluster_id].resize(list_size);
    d_cluster_lists[cluster_id].resize(list_size);
    fread(thrust::raw_pointer_cast(h_cluster_lists[cluster_id].data()), sizeof(uint), list_size, fp);
    thrust::copy(h_cluster_lists[cluster_id].begin(), h_cluster_lists[cluster_id].end(), d_cluster_lists[cluster_id].begin());
  }
  for(int cluster_id=0; cluster_id < n_clusters; cluster_id++){
    h_means[cluster_id].resize(dim);
    d_means[cluster_id].resize(dim);
    fread(thrust::raw_pointer_cast(h_means[cluster_id].data()), sizeof(float), dim, fp);
    thrust::copy(h_means[cluster_id].begin(), h_means[cluster_id].end(), d_means[cluster_id].begin());

    h_transforms[cluster_id].resize(nq*dim);
    for(int i=0; i<nq; i++){
      for(int j=0; j<dim; j++){
        h_transforms[cluster_id][i*dim + j] = h_means[cluster_id][j];
      }
    }
    d_transforms[cluster_id].resize(nq*dim);
    // thrust::copy(h_transforms[cluster_id].begin(), h_transforms[cluster_id].end(), d_transforms[cluster_id].begin());
    d_tmp[cluster_id].resize(nq*dim);
    thrust::copy(h_transforms[cluster_id].begin(), h_transforms[cluster_id].end(), d_tmp[cluster_id].begin());

    h_reconstructed[cluster_id].resize(nq*dim);
    d_reconstructed[cluster_id].resize(nq*dim);
    thrust::copy(h_transforms[cluster_id].begin(), h_transforms[cluster_id].end(), h_reconstructed[cluster_id].begin());
    thrust::copy(d_tmp[cluster_id].begin(), d_tmp[cluster_id].end(), d_reconstructed[cluster_id].begin());
  }
  for(int cluster_id=0; cluster_id < n_clusters; cluster_id++){
    h_rotation[cluster_id].resize(dim*n_components);
    d_rotation[cluster_id].resize(dim*n_components);
    fread(thrust::raw_pointer_cast(h_rotation[cluster_id].data()), sizeof(float), dim*n_components, fp);
    thrust::copy(h_rotation[cluster_id].begin(), h_rotation[cluster_id].end(), d_transforms[cluster_id].begin());
  }
  for(int cluster_id=0; cluster_id < n_clusters; cluster_id++){
    h_t[cluster_id].resize(n_components*dim);
    d_t[cluster_id].resize(n_components*dim);
    fread(thrust::raw_pointer_cast(h_t[cluster_id].data()), sizeof(float), n_components*dim, fp);
    thrust::copy(h_t[cluster_id].begin(), h_t[cluster_id].end(), d_t[cluster_id].begin());
  }
  for(int cluster_id=0; cluster_id < n_clusters; cluster_id++){
    h_pca_bases[cluster_id].resize(h_cluster_lists[cluster_id].size()*n_components);
    d_pca_bases[cluster_id].resize(h_cluster_lists[cluster_id].size()*n_components);
    fread(thrust::raw_pointer_cast(h_pca_bases[cluster_id].data()), sizeof(float), h_pca_bases[cluster_id].size(), fp);
    thrust::copy(h_pca_bases[cluster_id].begin(), h_pca_bases[cluster_id].end(), d_pca_bases[cluster_id].begin());
  }
  fclose(fp);

  //mr = mean * rotation
  float alpha = 1.0, beta = 0.0;
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "!!!! CUBLAS initialization error\n";
    return;
  }
  for(int cluster_id=0; cluster_id<n_clusters; cluster_id++){
    matrixMultiply(handle, d_means[cluster_id], d_rotation[cluster_id], d_transforms[cluster_id], nq, n_components, dim, alpha, beta);
  }
  cublasDestroy(handle);

  for(int i=0; i<n_clusters; i++){
    printf("%d ", h_cluster_lists[i].size());
  }
  printf("\n");
}*/

void KPCA::ReadCluster(const char* filename){
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("Failed to open file");
        throw std::runtime_error("Failed to open cluster file.");
    }

    // Initialize header information
    if (fread(&n_clusters, sizeof(uint), 1, fp) != 1 ||
        fread(&dim, sizeof(uint), 1, fp) != 1 ||
        fread(&n_components, sizeof(uint), 1, fp) != 1) { // Assuming nq is also part of the header
        fclose(fp);
        throw std::runtime_error("Failed to read header from file.");
    }

    // Resize vectors based on header info
    h_cluster_lists.resize(n_clusters);
    d_cluster_lists.resize(n_clusters);
    h_means.resize(n_clusters);
    d_means.resize(n_clusters);
    h_rotation.resize(n_clusters);
    d_rotation.resize(n_clusters);
    h_transforms.resize(n_clusters);
    d_transforms.resize(n_clusters);
    h_reconstructed.resize(n_clusters);
    d_reconstructed.resize(n_clusters);
    h_t.resize(n_clusters);
    d_t.resize(n_clusters);
    h_pca_bases.resize(n_clusters);
    d_pca_bases.resize(n_clusters);

    // Temporary buffers for reading data into host memory first
    std::vector<uint> temp_list;
    std::vector<float> temp_data;

    try {
        // Read cluster lists
        for(int cluster_id = 0; cluster_id < n_clusters; cluster_id++) {
            uint list_size;
            if (fread(&list_size, sizeof(uint), 1, fp) != 1) {
                throw std::runtime_error("Failed to read list size.");
            }
            h_cluster_lists[cluster_id].resize(list_size);
            d_cluster_lists[cluster_id].resize(list_size);
            temp_list.resize(list_size);
            if (fread(temp_list.data(), sizeof(uint), list_size, fp) != list_size) {
                throw std::runtime_error("Failed to read cluster list.");
            }
            thrust::copy(temp_list.begin(), temp_list.end(), h_cluster_lists[cluster_id].begin());
            thrust::copy(h_cluster_lists[cluster_id].begin(), h_cluster_lists[cluster_id].end(), d_cluster_lists[cluster_id].begin());
        }

        // Read means
        for(int cluster_id = 0; cluster_id < n_clusters; cluster_id++) {
            h_means[cluster_id].resize(dim);
            d_means[cluster_id].resize(dim);
            temp_data.resize(dim);
            if (fread(temp_data.data(), sizeof(float), dim, fp) != dim) {
                throw std::runtime_error("Failed to read means.");
            }
            thrust::copy(temp_data.begin(), temp_data.end(), h_means[cluster_id].begin());
            thrust::copy(h_means[cluster_id].begin(), h_means[cluster_id].end(), d_means[cluster_id].begin());

            //Initialize reconstructed with means
            h_reconstructed[cluster_id].resize(nq*dim);
            for(int i=0; i<nq; i++){
              for(int j=0; j<dim; j++){
                h_reconstructed[cluster_id][i*dim + j] = h_means[cluster_id][j];
              }
            }
            d_reconstructed[cluster_id].resize(nq * dim);
            thrust::copy(h_reconstructed[cluster_id].begin(), h_reconstructed[cluster_id].end(), d_reconstructed[cluster_id].begin());
        }

        // Read rotations
        for(int cluster_id = 0; cluster_id < n_clusters; cluster_id++) {
            h_rotation[cluster_id].resize(dim * n_components);
            d_rotation[cluster_id].resize(dim * n_components);
            temp_data.resize(dim * n_components);
            if (fread(temp_data.data(), sizeof(float), dim * n_components, fp) != dim * n_components) {
                throw std::runtime_error("Failed to read rotation.");
            }
            thrust::copy(temp_data.begin(), temp_data.end(), h_rotation[cluster_id].begin());
            thrust::copy(h_rotation[cluster_id].begin(), h_rotation[cluster_id].end(), d_rotation[cluster_id].begin());
        }

        // Read t matrices
        for(int cluster_id = 0; cluster_id < n_clusters; cluster_id++) {
            h_t[cluster_id].resize(n_components * dim);
            d_t[cluster_id].resize(n_components * dim);
            temp_data.resize(n_components * dim);
            if (fread(temp_data.data(), sizeof(float), n_components * dim, fp) != n_components * dim) {
                throw std::runtime_error("Failed to read t matrix.");
            }
            thrust::copy(temp_data.begin(), temp_data.end(), h_t[cluster_id].begin());
            thrust::copy(h_t[cluster_id].begin(), h_t[cluster_id].end(), d_t[cluster_id].begin());
        }

        // Read pca bases
        for(int cluster_id = 0; cluster_id < n_clusters; cluster_id++) {
            h_pca_bases[cluster_id].resize(h_cluster_lists[cluster_id].size() * n_components);
            d_pca_bases[cluster_id].resize(h_cluster_lists[cluster_id].size() * n_components);
            temp_data.resize(h_pca_bases[cluster_id].size());
            if (fread(temp_data.data(), sizeof(float), h_pca_bases[cluster_id].size(), fp) != h_pca_bases[cluster_id].size()) {
                throw std::runtime_error("Failed to read pca bases.");
            }
            thrust::copy(temp_data.begin(), temp_data.end(), h_pca_bases[cluster_id].begin());
            thrust::copy(h_pca_bases[cluster_id].begin(), h_pca_bases[cluster_id].end(), d_pca_bases[cluster_id].begin());
        }

        fclose(fp); // Close the file before proceeding with any CUDA operations

        // d_transforms = temp_means * d_rotation
        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("CUBLAS initialization error.");
        }

        float alpha = 1.0f, beta = 0.0f;
        for(int cluster_id = 0; cluster_id < n_clusters; cluster_id++) {
          d_transforms[cluster_id].resize(nq * n_components);
          thrust::fill(d_transforms[cluster_id].begin(), d_transforms[cluster_id].end(), 0.0f);
          matrixMultiply(handle, d_reconstructed[cluster_id], d_rotation[cluster_id], d_transforms[cluster_id], nq, n_components, dim, alpha, beta);
        }

        cublasDestroy(handle);

        // Print sizes of cluster lists as a final check
        for(int i = 0; i < n_clusters; i++){
            printf("%d ", h_cluster_lists[i].size());
        }
        printf("\n");

    } catch (const std::exception& e) {
        fprintf(stderr, "Error in ReadCluster: %s\n", e.what());
        if (fp) fclose(fp); // Ensure file is closed in case of exception
        throw; // Rethrow the exception after cleanup
    }
}