## Setup

### GPU

To install JAX and the other required packages, we first need to have system-wide installations of:
 - CUDA
 - CUDNN
CUDA should be located (or symlinked) to `/usr/local/`,
as explained on the JAX [GitHub](https://github.com/google/jax#installation).
There are also some version requirements for CUDA, the driver, and CUDNN (by default, I am using CUDA 12, driver 525.85.05, CUDNN v8.9.5)
 - You can do `nvidia-smi` to see the CUDA and driver versions
 - To see if CUDNN is installed, you can try guessing at the library locations (often `usr/local`) or do `find / -name "*libcudnn*”`.
 - You can find the appropriate CUDNN version for your CUDA version on the [NVIDIA developer webpage](https://developer.nvidia.com/rdp/cudnn-download).

Then you can install the virtual environment:
```
cd simple_GNN_JAX
python -m venv venv_gpu
source venv_gpu/bin/activate
pip install -r requirements_gpu.txt
```

The necessary packages are:
```
[Check for (latest instructions)[https://github.com/google/jax#installation] – recommend CUDA 12 via pip shown below]
pip install --upgrade pip setuptools wheel
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --no-cache-dir
pip install flax jraph torch_geometric energyflow tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If you encounter an issue with incompatible CUDNN versions (due to e.g. conflict with a system install), you might need to do:
```
export LD_LIBRARY_PATH=$PWD/venv_gpu/lib/python3.9/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH
```

### CPU

The CPU version of JAX is simpler to install, and may be useful for spinning up quick tests – simply create a virtual environment and install the necessary packages:
```
cd simple_GNN_JAX
python -m venv venv_cpu
source venv_cpu/bin/activate
pip install -r requirements_cpu.txt
```

The necessary packages are:
```
[Check for (latest instructions)[https://github.com/google/jax#installation], to ensure version compatibility with CUDA...here we are running CUDA 12.0]
pip install --upgrade pip setuptools wheel
pip install --upgrade "jax[cpu]"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install flax jraph torch_geometric energyflow tqdm
```