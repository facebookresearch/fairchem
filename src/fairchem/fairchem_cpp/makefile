EXT_NAME=extension_cpp
TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
TORCH_ABI=0

.PHONY: all clean install rebuild

all: install

clean:
		rm -rf build dist *.egg-info $(EXT_NAME)/_C.*.so
			rm -rf ~/.cache/torch_extensions/

install:
		TORCH_CUDA_ARCH_LIST=$(TORCH_CUDA_ARCH_LIST) \
				     	CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI)" \
						uv pip install -v --no-build-isolation -e .

rebuild: clean install
