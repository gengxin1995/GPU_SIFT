使用Wu Changchang同学写的GPU版本的SIFT，[http://www.cs.unc.edu/~ccwu/siftgpu/](http://www.cs.unc.edu/~ccwu/siftgpu/)

需要安装：  
* SiftGPU([参考链接](http://www.cnblogs.com/gaoxiang12/p/5149067.html))
* opencv2.4.13
* Cmake
* CUDA

修改**CMakeLists.txt**中SiftGPU的头文件和库文件位置

运行：
```bash
cmake .
make
./GPU_SIFT
```