# Justfile for RNS-Testing
#
# This Justfile provides common development tasks:
#
#   • build  - Configures and builds the project from a clean (or existing) build directory.
#   • test   - Builds (if needed) and runs tests with output on failure.
#   • clean  - Removes the build directory.
#
# Usage:
#   just build
#   just test
#   just clean

BUILD_DIR := "build"

build:
  @echo "Configuring and building project with Ninja (Release mode)..."
  mkdir -p {{BUILD_DIR}}
  cd {{BUILD_DIR}} && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . && \
    ln -srf compile_commands.json ../ 

test: build
	@echo "Running tests..."
	cd {{BUILD_DIR}} && ctest --output-on-failure

benchmark: build
	@echo "Running benchmarks..."
	cd {{BUILD_DIR}} && ./matrix_benchmark

clean:
  @echo "Cleaning build directory..."
  rm -rf {{BUILD_DIR}}
  rm -f compile_commands.json
