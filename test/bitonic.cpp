#include <bits/stdc++.h>
int main(){
  uint n_points_per_batch = 8;
  uint step_id = 1;
  uint substep_id = 1;

  for(; step_id <= n_points_per_batch/2; step_id *= 2){
    substep_id = step_id;

    for(; substep_id >= 1; substep_id /= 2){
      printf("%d\t%d\t", step_id, substep_id);
      // for(int temparory_id = 0; temparory_id < (n_points_per_batch/2 + blockSize - 1)/blockSize; )
      for(int i=0; i<n_points_per_batch; i++){
        int unrollt_id = (i/substep_id) * 2 * substep_id + (i&(substep_id-1));
        printf("(%d, %d) ", unrollt_id, unrollt_id + substep_id);
      }
      printf("\n");
    }
    printf("\n");
  }
  return 0;
}