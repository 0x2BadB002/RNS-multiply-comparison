#include "../src/matrix_heap.hpp"
#include <benchmark/benchmark.h>

static void BM_MatrixHeapMultiplication(benchmark::State &state) {
  const size_t size = static_cast<size_t>(state.range(0));
  const auto A = MatrixHeap<int>::generate_random(size, -100, 100);
  const auto B = MatrixHeap<int>::generate_random(size, -100, 100);

  for (auto _ : state) {
    auto result = A * B;
    benchmark::DoNotOptimize(result);
  }

  state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_MatrixHeapMultiplication)
    ->Arg(10)
    ->Arg(64)
    ->Arg(100)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond)
    ->Complexity(benchmark::oNCubed);
