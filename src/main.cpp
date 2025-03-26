#include <iostream>
#include <ostream>

#include "matrix_stack.hpp"

int main(int argc, char *argv[]) {
  MatrixStack<2> m{{2, 2}};

  for (auto const &el : m) {
    std::cout << el << ' ';
  }
  std::cout << std::endl;

  return 0;
}
