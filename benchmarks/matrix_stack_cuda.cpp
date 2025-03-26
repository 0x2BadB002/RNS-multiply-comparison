#include "../src/matrix_stack.hpp"
#include <benchmark/benchmark.h>

template <size_t N> void MatrixStackMultiplyCudaImpl(benchmark::State &state) {
  auto a = MatrixStack<N>::generate_random();
  auto b = MatrixStack<N>::generate_random();
  for (auto _ : state) {
    auto c = a.multiply_cuda(b);
    benchmark::DoNotOptimize(c);
  }
  state.SetComplexityN(N);
}

static void BM_MatrixStackMultiplyCUDA(benchmark::State &state) {
  const auto N = state.range(0);
  switch (N) {
  case 10:
    MatrixStackMultiplyCudaImpl<10>(state);
    break;
  case 64:
    MatrixStackMultiplyCudaImpl<64>(state);
    break;
  case 100:
    MatrixStackMultiplyCudaImpl<100>(state);
    break;
  case 128:
    MatrixStackMultiplyCudaImpl<128>(state);
    break;
  case 256:
    MatrixStackMultiplyCudaImpl<256>(state);
    break;
  case 512:
    MatrixStackMultiplyCudaImpl<512>(state);
    break;
  default:
    state.SkipWithError("Unsupported matrix size");
  }
}

BENCHMARK(BM_MatrixStackMultiplyCUDA)
    ->Unit(benchmark::kMillisecond)
    ->Complexity(benchmark::oNCubed)
    ->Arg(10)
    ->Arg(64)
    ->Arg(100)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512);
