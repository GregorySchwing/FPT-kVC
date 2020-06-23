//#include "serial/COO.h"
//#include "serial/CSR.h"
//#include "DCSR.h"
#include "gpu/COO.cuh"

int main(int argc, char *argv[])
{
    /*
    COO coo1(4, 4, 4, true);
    COO coo2(4, 4, 4, true);
    std::cout << coo1.toString();
    std::cout << coo2.toString();
    coo1.insertElements(coo2);
    std::cout << coo1.toString();

    CSR csr1(coo1);
    CSR csr2(coo2);

    std::cout << csr1.toString();
    std::cout << csr2.toString();

    std::cout << "Calling Insert Elements" << std::endl;
    csr1.insertElements(coo1);

    std::cout << csr1.toString();
*/

  int N = 2e7;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) 
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 20M elements on the GPU
  add<<<1, 512>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  // Free memory
  cudaFree(x);
  cudaFree(y); 
  return 0;

 //   std::cout << coo1.toString();
 //   std::cout << coo2.toString();



  //  DCSR dcsr(csr1, 2);
  //  std::cout << dcsr.toString();
  //  dcsr.allocateSegments(csr2);
  //  std::cout << dcsr.toString();

}