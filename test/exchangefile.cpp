#include <bits/stdc++.h>
using namespace std;
int main(){
  int d;
  const char* infilename = "/data/myl/sift10M/sift10M_base.hh";
  FILE* infile = fopen(infilename, "rb");
  if(infile == NULL){
    printf("inFile open failed\n");
    return 0;
  }
  assert(fread(&d, sizeof(d), 1, infile) == 1);
  long long infilelength = 0;
  fseek(infile, 0, SEEK_END);
  infilelength = ftell(infile);
  fseek(infile, 0, SEEK_SET);
  cout<<infilelength<<endl;
  int n = (infilelength - 4) / (d * 4);
  printf("n: %d, d: %d\n", n, d);
  fread(&d, sizeof(d), 1, infile);
  float* data = new float[n*d];
  fread(data, sizeof(float), 1LL*n*d, infile);
  fclose(infile);

  const char* outfilename = "/data/myl/sift10M/sift10M_base_.fbin";
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