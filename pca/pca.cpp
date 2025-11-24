#include <bits/stdc++.h>
#include "pca.h"

void PCA::computeCov(Eigen::MatrixXd &X, Eigen::MatrixXd &C)
{
	printf("computeCov\n");
	//计算协方差矩阵C = XTX / n-1;
	C = X.adjoint() * X;
	C = C.array() / (X.rows() - 1);
}
void PCA::computeEig(Eigen::MatrixXd &C, Eigen::MatrixXd &vec, Eigen::MatrixXd &val)
{
	printf("computeEig\n");
	//计算特征值和特征向量，使用selfadjont按照对阵矩阵的算法去计算，可以让产生的vec和val按照有序排列
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C);
 
	vec = eig.eigenvectors();//求自伴随矩阵的特征向量
	val = eig.eigenvalues();//求自伴随矩阵的特征值

	//按照特征值从大到小顺序
	for(int i=0; i<dim/2; i++){
		Eigen::MatrixXd tmp = vec.col(i);
		vec.col(i) = vec.col(dim-1-i);
		vec.col(dim-1-i) = tmp;
		tmp = val.row(i);
		val.row(i) = val.row(dim-1-i);
		val.row(dim-1-i) = tmp;
	}
}
double PCA::Ratio(uint ratio_dim){
	double sum = 0;
	for(int i=0; i<ratio_dim; i++){
		sum += val(i,0);
		printf("%f ",sum/val.sum());
	}
	printf("\n");
	sum=sum/val.sum();
	return sum;
}

void PCA::calc_eigenvalues(){
  //均值零化
	printf("calc_eigenvalues\n");
	Eigen::MatrixXd meanval = B.colwise().mean();
	meanvecRow = meanval;
	for(int i=0; i<dim; i++) printf("%f ", meanvecRow(i));
	//样本均值化为0
	B.rowwise() -= meanvecRow;

	//计算协方差
	computeCov(B, C);
	//计算特征值和特征向量
	computeEig(C, vec, val);

  //降低的维数以及特征值占比
  // printf("dim = %d\n", dim);
	// printf("ratio = %.5f\n", Ratio(dim, val));
}

void PCA::save_mean_rotation(const char *mean_path, const char *rotation_path){
	printf("save_mean_rotation\n");
	FILE *mean_file = fopen(mean_path, "wb");
	FILE *rotation_file = fopen(rotation_path, "wb");
// write mean file
	if (mean_file == nullptr) {
		perror("Failed to open file");
		return;
	}
	printf("dim = %d, meanvecRow.size() = %d\n", dim, meanvecRow.size());
	uint row = 1;
	fwrite(&row, sizeof(uint), 1, mean_file);
	fwrite(&dim, sizeof(uint), 1, mean_file);
	for(int i=0; i<dim; i++){
		float tmp = meanvecRow(i);
		fwrite(&tmp, sizeof(float), 1, mean_file);
	}
	fclose(mean_file);
// write rotation file
	printf("dim = %d, vec.cols() = %d, vec.rows() = %d\n", dim, vec.cols(), vec.rows());
	fwrite(&dim, sizeof(uint), 1, rotation_file);
	fwrite(&dim, sizeof(uint), 1, rotation_file);
	for(int i=0; i<dim; i++){
		for(int j=0; j<dim; j++){
			float tmp = vec(i,j);
			fwrite(&tmp, sizeof(float), 1, rotation_file);
		}
		// fwrite(vec.col(i).data(), sizeof(float), dim, rotation_file);
	}
	fclose(rotation_file);
}

void PCA::read_mean_rotation(const char *mean_path, const char *rotation_path){
	FILE *mean_file = fopen(mean_path, "rb");
	FILE *rotation_file = fopen(rotation_path, "rb");
// read mean file
	uint row;
	fread(&row, sizeof(uint), 1, mean_file);
	fread(&dim, sizeof(uint), 1, mean_file);
	meanvecRow.resize(dim);
	for(int i=0; i<dim; i++){
		float tmp;
		fread(&tmp, sizeof(float), 1, mean_file);
		meanvecRow(i) = tmp;
	}
	fclose(mean_file);
// read rotation file
	uint dim2;
	fread(&dim, sizeof(uint), 1, rotation_file);
	fread(&dim2, sizeof(uint), 1, rotation_file);
	vec.resize(dim, dim2);
	for(int i=0; i<dim; i++){
		for(int j=0; j<dim2; j++){
			float tmp;
			fread(&tmp, sizeof(float), 1, rotation_file);
			vec(i,j) = tmp;
		}
	}
	fclose(rotation_file);
}

void PCA::calc_result(uint pj_dim){
	for(int i=0; i<dim; i++) printf("%f ", val(i,0));
	printf("\n");
  	B_res = B * vec.leftCols(pj_dim);
}

void PCA::save_result(uint pj_dim, const char *pca_base_path){
	FILE *pca_base_file = fopen(pca_base_path, "wb");
	fwrite(&nb, sizeof(uint), 1, pca_base_file);
	fwrite(&pj_dim, sizeof(uint), 1, pca_base_file);
	for(int i=0; i<nb; i++){
		for(int j=0; j<pj_dim; j++){
			float tmp = B_res(i,j);
			fwrite(&tmp, sizeof(float), 1, pca_base_file);
		}
	}
	fclose(pca_base_file);
}

void PCA::save_linear_params(const char *linear_params_path){
	FILE *linear_file = fopen(linear_params_path, "wb");
	if (linear_file == nullptr) {
		perror("Failed to open linear params file for writing");
		return;
	}
	uint model_count = w.size();
	fwrite(&model_count, sizeof(uint), 1, linear_file);
	for(int i=0; i<model_count; i++){
		fwrite(&w[i], sizeof(float), 1, linear_file);
		fwrite(&b[i], sizeof(float), 1, linear_file);
	}
	fclose(linear_file);
}

void PCA::read_linear_params(const char *linear_params_path){
	FILE *linear_file = fopen(linear_params_path, "rb");
	if (linear_file == nullptr) {
		perror("Failed to open linear params file for reading");
		return;
	}
	uint model_count;
	fread(&model_count, sizeof(uint), 1, linear_file);
	w.resize(model_count);
	b.resize(model_count);
	for(int i=0; i<model_count; i++){
		fread(&w[i], sizeof(float), 1, linear_file);
		fread(&b[i], sizeof(float), 1, linear_file);
	}
	fclose(linear_file);
}

float naive_l2_dist_calc(float* p, float* q, int D){
	float dis = 0;
	for(int i=0; i<D; i++){
		dis += (p[i] - q[i]) * (p[i] - q[i]);
	}
	return dis;
}

// 传入的数据都是pca映射后的数据
void PCA::linear(float* data, float* query, int* groundtruth, int np, int nq, int test_nq, int topk, int gt_k, int D, int delta_d){
    std::vector<float> Dis, thresh;
    std::vector<std::vector<float> > Dis_;
    int model_count = D / delta_d;
    if (D % delta_d) model_count++;
    Dis_.resize(model_count);
	int count_bound = std::min(test_nq, nq);
	for(int i=0; i<count_bound; i++){
		float *q = query + i*D;
		int *gt = groundtruth + i*gt_k;
		float *p = data + gt[topk-1]*D;
		float thresh_dis = naive_l2_dist_calc(p, q, D);
		for(int j=0; j<topk; j++){
			p = data + gt[j]*D;
			float dis_ = 0;
			float dis = naive_l2_dist_calc(p, q, D);
			unsigned dis_count = 0;
			for(int k=0; k<D; k+=delta_d){
				if(k+delta_d > D) dis_ += naive_l2_dist_calc(q+k, p+k, D%delta_d);
				else dis_ += naive_l2_dist_calc(q+k, p+k, delta_d);
				Dis_[dis_count].push_back(dis_);
				dis_count++;
			}
			Dis.push_back(dis);
			thresh.push_back(thresh_dis);
		}
	}
	w.resize(model_count);
	b.resize(model_count);
	Eigen::VectorXf Y = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(Dis.data(), Dis.size());
	float y_mean = Y.mean();
	for(int i=0; i<model_count; i++){
		Eigen::VectorXf X = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(Dis_[i].data(), Dis_[i].size());
		float x_mean = X.mean();
		Eigen::VectorXf X_centered = X.array() - x_mean;
		Eigen::VectorXf Y_centered = Y.array() - y_mean;
		float w_i = (X_centered.cwiseProduct(Y_centered).sum()) / (X_centered.cwiseProduct(X_centered).sum());
		float b_i = y_mean - w_i * x_mean;
		w[i] = w_i;
		b[i] = b_i;
	}
}