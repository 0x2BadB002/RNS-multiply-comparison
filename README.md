# RNS matrix multiplication

В программе реализовано множество алгоритмов для умножения матриц. Также 
сравнивается производительность этих алгоритмов.

## Используемые библиотеки/программы

- `g++`
- `cmake`
- `cuda`
- `googletests` 
- `googlebenchmark`

## Сборка программы

```sh
just build
```

## Запуск тестов

```sh
just test
```

## Запуск бенчмарка

```sh
just benchmark
```

## Реализованы алгоритмы

- Умножение за N³ с хранением матрицы в стеке
- Умножение за N³ с хранением матрицы в куче
- Распределенное умножение матрицы с помощью CUDA
- Распределенное умножение матрицы с помощью CUDA в системе RNS

## Вывод программы

| Benchmark                                             | Time       | CPU        | Iterations |
|-------------------------------------------------------|------------|------------|------------|
| BM\_MatrixStackMultiplyCUDA\_RNS/10                   | 0.101 ms   | 0.101 ms   | 5078       |
| BM\_MatrixStackMultiplyCUDA\_RNS/64                   | 0.101 ms   | 0.101 ms   | 6896       |
| BM\_MatrixStackMultiplyCUDA\_RNS/100                  | 0.116 ms   | 0.116 ms   | 5954       |
| BM\_MatrixStackMultiplyCUDA\_RNS/128                  | 0.123 ms   | 0.123 ms   | 5605       |
| BM\_MatrixStackMultiplyCUDA\_RNS/256                  | 0.273 ms   | 0.272 ms   | 2547       |
| BM\_MatrixStackMultiplyCUDA\_RNS/512                  | 1.20 ms    | 1.20 ms    | 571        |
| BM\_MatrixStackMultiplyCUDA\_RNS\_BigO                | 0.01 N³    | 0.01 N³    |            |
| BM\_MatrixStackMultiplyCUDA\_RNS\_RMS                 | 31 %       | 31 %       |            |
| BM\_MatrixStackMultiplyCUDA/10                        | 0.089 ms   | 0.089 ms   | 7932       |
| BM\_MatrixStackMultiplyCUDA/64                        | 0.101 ms   | 0.101 ms   | 6962       |
| BM\_MatrixStackMultiplyCUDA/100                       | 0.115 ms   | 0.115 ms   | 6058       |
| BM\_MatrixStackMultiplyCUDA/128                       | 0.123 ms   | 0.123 ms   | 5677       |
| BM\_MatrixStackMultiplyCUDA/256                       | 0.273 ms   | 0.273 ms   | 2546       |
| BM\_MatrixStackMultiplyCUDA/512                       | 1.20 ms    | 1.20 ms    | 567        |
| BM\_MatrixStackMultiplyCUDA\_BigO                     | 0.01 N³    | 0.01 N³    |            |
| BM\_MatrixStackMultiplyCUDA\_RMS                      | 30 %       | 30 %       |            |
| BM\_MatrixStackMultiply/10                            | 0.000 ms   | 0.000 ms   | 4140677    |
| BM\_MatrixStackMultiply/64                            | 0.047 ms   | 0.047 ms   | 15057      |
| BM\_MatrixStackMultiply/100                           | 0.177 ms   | 0.177 ms   | 3949       |
| BM\_MatrixStackMultiply/128                           | 0.391 ms   | 0.390 ms   | 1797       |
| BM\_MatrixStackMultiply/256                           | 3.33 ms    | 3.33 ms    | 210        |
| BM\_MatrixStackMultiply/512                           | 36.0 ms    | 35.9 ms    | 19         |
| BM\_MatrixStackMultiply\_BigO                         | 0.27 N³    | 0.27 N³    |            |
| BM\_MatrixStackMultiply\_RMS                          | 7 %        | 7 %        |            |
| BM\_MatrixHeapMultiplication/10                       | 0.000 ms   | 0.000 ms   | 1786376    |
| BM\_MatrixHeapMultiplication/64                       | 0.072 ms   | 0.072 ms   | 9438       |
| BM\_MatrixHeapMultiplication/100                      | 0.317 ms   | 0.317 ms   | 2214       |
| BM\_MatrixHeapMultiplication/128                      | 1.01 ms    | 1.01 ms    | 690        |
| BM\_MatrixHeapMultiplication/256                      | 8.72 ms    | 8.70 ms    | 80         |
| BM\_MatrixHeapMultiplication/512                      | 99.1 ms    | 98.8 ms    | 7          |
| BM\_MatrixHeapMultiplication\_BigO                    | 0.73 N³    | 0.73 N³    |            |
| BM\_MatrixHeapMultiplication\_RMS                     | 8 %        | 8 %        |            |

## Принцип работы 

1. Для сравнения реализованы алгоритмы умножения матриц с временной сложностью N³, умножение происходит на CPU;
2. Реализовано распределенное умножение матриц с помощью CUDA;
3. Реализовано распределенное умножение матриц в системе RNS с помощью CUDA:
    - используется базис `{ 2, 3, 5, 7, 11, 13, 17, 19 }`;
    - для подсчета промежуточных значений написана программа помошник `misc/moduli.py`.
