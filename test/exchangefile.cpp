#include <bits/stdc++.h>
using namespace std;
int main(){
  int d, n;
  const char* infilename = "/home/myl/rt_entry/bin/sift1M/pca_base.fbin";
  FILE* infile = fopen(infilename, "rb");
  if(infile == NULL){
    printf("inFile open failed\n");
    return 0;
  }
  assert(fread(&n, sizeof(n), 1, infile) == 1);
  assert(fread(&d, sizeof(d), 1, infile) == 1);
  printf("n: %d, d: %d\n", n, d);
  float* data = new float[n*d];
  fread(data, sizeof(float), 1LL*n*d, infile);
  fclose(infile);

  const char* outfilename = "/home/myl/rt_entry/test/sift1M_pca_last_64.fbin";
  FILE* outfile = fopen(outfilename, "wb");
  if(outfile == NULL){
    printf("outFile open failed\n");
    return 0;
  }
  fwrite(&n, sizeof(n), 1, outfile);
  int last_d = 64;
  fwrite(&last_d, sizeof(last_d), 1, outfile);
  for(int i=0; i<n; i++){
    for(int j=d-last_d; j<d; j++){
      fwrite(&data[i*d+j], sizeof(float), 1, outfile);
    }
  }
  fclose(outfile);
  return 0;
}