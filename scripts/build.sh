echo "Building cache search..."

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$ROOT_DIR"

mkdir -p build
cd build

cmake ..
make -j$(nproc)

echo "Build finished! "