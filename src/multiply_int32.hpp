#ifndef MULTIPLY_INT32
#define MULTIPLY_INT32

#include <cstddef>
#include <cstdint>

void matrixMultiplyCUDA(const int32_t *A, const int32_t *B, int32_t *C,
                        size_t N);

void rnsMatrixMultiply(const int32_t *h_A, const int32_t *h_B, int32_t *h_C,
                       size_t N);

#endif // !MULTIPLY_INT32
