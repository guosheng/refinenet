


export PATH=/usr/local/cuda-7.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=../libs/cudnn5/lib64:$LD_LIBRARY_PATH

matlab -nodisplay -singleCompThread
# matlab
