nsys profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -o double_buffer \
  --force-overwrite=true \
  --stats=true \
./bin/test --ALGO=3 --topk=0 --n_candidates=1024 --expand_ratio=0.2 --point_ratio=0.00001024 --search_width=4 \
          --t=600 --n_cluster=10000 --page_size=1000 --n_page=15000 \
          --centroids_path="/mnt/IntelP5520_8T_1/myl/cache_search/data/sift100M/centroids_10000_hilbert" \
          --max_iter=100


./bin/test --ALGO=1 --topk=0 --n_candidates=128 --expand_ratio=0.2 --point_ratio=0.000128 --search_width=4 \
          --t=6 --n_cluster=100 --page_size=1000 --n_page=1500 \
          --max_iter=100