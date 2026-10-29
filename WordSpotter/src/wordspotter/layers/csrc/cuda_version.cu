#include <cuda_runtime_api.h>

namespace wordspotter {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace wordspotter
