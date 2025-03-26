#include "../src/matrix_stack.hpp"
#include <benchmark/benchmark.h>

template <size_t N> void MatrixStackMultiplyImpl(benchmark::State &state) {
  auto a = MatrixStack<N>::generate_random();
  auto b = MatrixStack<N>::generate_random();
  for (auto _ : state) {
    auto c = a * b;
    benchmark::DoNotOptimize(c);
  }
  state.SetComplexityN(N);
}

static void BM_MatrixStackMultiply(benchmark::State &state) {
  const auto N = state.range(0);
  switch (N) {
  case 10:
    MatrixStackMultiplyImpl<10>(state);
    break;
  case 64:
    MatrixStackMultiplyImpl<64>(state);
    break;
  case 100:
    MatrixStackMultiplyImpl<100>(state);
    break;
  case 128:
    MatrixStackMultiplyImpl<128>(state);
    break;
  case 256:
    MatrixStackMultiplyImpl<256>(state);
    break;
  case 512:
    MatrixStackMultiplyImpl<512>(state);
    break;
  default:
    state.SkipWithError("Unsupported matrix size");
  }
}

BENCHMARK(BM_MatrixStackMultiply)
    ->Unit(benchmark::kMillisecond)
    ->Complexity(benchmark::oNCubed)
    ->Arg(10)
    ->Arg(64)
    ->Arg(100)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512);
