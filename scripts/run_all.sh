set -e

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$ROOT_DIR"

echo "=== Cache Search - 完整运行流程 ==="

if [ $# -lt 3 ]; then
    echo "用法: $0 <K> <test_nq> <t> [delta_d] [m_partial] [nq_eval] [tmp_topk] [topk]"
    echo "  K: 聚类数量"
    echo "  test_nq: 用于线性拟合的query数量"
    echo "  t: 每个query查找的最近聚类中心数量"
    echo "  delta_d: 线性/验证的分块大小(默认 D/4)"
    echo "  m_partial: 验证阶段使用的块数m(默认 ceil((D/4)/delta_d))"
    echo "  nq_eval: 验证时参与计算的query数量(默认全部)"
    echo "  tmp_topk: 两阶段方法中先用估计距离选择的候选数量(默认200，仅模式3和4需要)"
    echo "  topk: 最终选择的最近邻数量(默认100，仅模式3和4需要)"
    echo ""
    echo "示例: bash run_all.sh 100 10000 3 1 64 500 100 100"
    exit 1
fi

K=$1
test_nq=$2
t=$3
delta_d=${4:-0}
m_partial=${5:-0}
nq_eval=${6:-0}
tmp_topk=${7:-200}
topk=${8:-100}

echo "参数: K=$K, test_nq=$test_nq, t=$t, delta_d=${delta_d}, m_partial=${m_partial}, nq_eval=${nq_eval}, tmp_topk=${tmp_topk}, topk=${topk}"

# DATASET_PATH="/data/myl/deep1M/deep1M_base.fvecs" # TODO: change dataset path
DATASET_PATH="/data/myl/sift1M/sift1M_base.fvecs" # TODO: change dataset path
# DATASET_PATH="/data/myl/sift100M/sift100M_base.fbin"
DATASET_NAME=$(basename "$DATASET_PATH" | cut -d'_' -f1)
DATA_ROOT="/data/myl/cache_search/data/${DATASET_NAME}"
mkdir -p "$DATA_ROOT"
CENTROIDS_FILE="$DATA_ROOT/centroids_${K}"

LOG_FILE="$DATA_ROOT/run_all.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "无论如何我也要构建..."
bash scripts/build.sh

if [ ! -f "bin/compute_pca" ] || [ ! -f "bin/test" ]; then
  echo "未发现可执行文件，先构建..."
  bash scripts/build.sh
fi

echo "=== 第一步：聚类 ==="
if [ ! -f "$CENTROIDS_FILE" ]; then
  echo "运行 python scripts/cluster.py $K ..."
  python3 scripts/cluster.py $K
else
  echo "centroids_$K 已存在，跳过"
fi

echo "=== 第二步：PCA与线性参数 ==="
if [ "$delta_d" -gt 0 ]; then
  ./bin/compute_pca "$test_nq" "$delta_d" "$topk"
else
  ./bin/compute_pca "$test_nq" "0" "$topk"
fi

echo "=== 第三步：运行cache_search ==="
# compute-sanitizer --print-limit 3 --show-backtrace=yes ./bin/test --ALGO=1 --topk=0 --n_candidates=128 --expand_ratio=0.2 --point_ratio=0.000128 --search_width=1 \
#           --t=$t --n_cluster=$K --page_size=1000 --n_page=500 --centroids_path="$CENTROIDS_FILE"
# ./bin/test --ALGO=1 --topk=0 --n_candidates=64 --expand_ratio=0.2 --point_ratio=0.000064 --search_width=4 \
#           --t=$t --n_cluster=$K --page_size=1000 --n_page=2000 --centroids_path="$CENTROIDS_FILE" --max_iter=100

# 用ncu分析kernel
# ncu --set full --target-processes all -o profile_output \
# ./bin/test --ALGO=1 --topk=0 --n_candidates=64 --expand_ratio=0.2 --point_ratio=0.000064 --search_width=4 \
#           --t=$t --n_cluster=$K --page_size=1000 --n_page=2000 --centroids_path="$CENTROIDS_FILE" --max_iter=100

nsys profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -o my_report \
  --force-overwrite=true \
  --stats=true \
  ./bin/test --ALGO=2 --topk=0 --n_candidates=128 --expand_ratio=0.2 --point_ratio=0.000128 --search_width=4 \
          --t=$t --n_cluster=$K --page_size=1000 --n_page=300 --centroids_path="$CENTROIDS_FILE" --max_iter=12

# ./bin/test --ALGO=1 --topk=0 --n_candidates=128 --expand_ratio=0.2 --point_ratio=0.000128 --search_width=4 \
#           --t=$t --n_cluster=$K --page_size=1000 --n_page=2000 --centroids_path="$CENTROIDS_FILE" --max_iter=100

# ./bin/test --ALGO=1 --topk=0 --n_candidates=32 --expand_ratio=0.2 --point_ratio=0.00000032 --search_width=8 \
#           --t=$t --n_cluster=$K --page_size=1000 --n_page=1000 --centroids_path="$CENTROIDS_FILE" --max_iter=100

# ./bin/test --ALGO=1 --topk=0 --n_candidates=256 --expand_ratio=0.2 --point_ratio=0.00000256 --search_width=4 \
#           --t=$t --n_cluster=$K --page_size=1000 --n_page=6461 --centroids_path="$CENTROIDS_FILE" --max_iter=100

# ./bin/test --ALGO=2 --topk=0 --n_candidates=512 --expand_ratio=0.2 --point_ratio=0.00000512 --search_width=4 \
#           --t=$t --n_cluster=$K --page_size=1000 --n_page=1000 --centroids_path="$CENTROIDS_FILE" --max_iter=100