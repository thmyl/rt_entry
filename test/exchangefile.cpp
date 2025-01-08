#include <bits/stdc++.h>
using namespace std;
int main(){
  int d, n;
  const char* infilename = "/data/myl/sift10M/sift10M_base.hh";
  FILE* infile = fopen(infilename, "rb");
  if(infile == NULL){
    printf("inFile open failed\n");
    return 0;
  }
  n=10000000;
  // assert(fread(&n, sizeof(n), 1, infile) == 1);
  assert(fread(&d, sizeof(d), 1, infile) == 1);
  printf("n: %d, d: %d\n", n, d);
  float* data = new float[1LL*n*d];
  fread(data, sizeof(float), 1LL*n*d, infile);
  fclose(infile);

  const char* outfilename = "/data/myl/sift10M/sift10M_base.fbin";
  FILE* outfile = fopen(outfilename, "wb");
  if(outfile == NULL){
    printf("outFile open failed\n");
    return 0;
  }
  fwrite(&n, sizeof(n), 1, outfile);
  fwrite(&d, sizeof(d), 1, outfile);
  fwrite(data, sizeof(float), 1LL*n*d, outfile);
  fclose(outfile);
  return 0;
}