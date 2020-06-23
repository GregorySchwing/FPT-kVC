#include <iostream>
#include <stdio.h>
#include "cuda_by_example/common/book.h"


__global__ void kernel(void){

}

__global__ void add(int a, int b, int *c){
    *c = a + b;
}

int main(int argc, char *argv[])
{
    int c;
    int *dev_c;
    HANDLE_ERROR( cudaMalloc((void**)&dev_c, sizeof(int)));

    add<<<1,1>>>( 2, 7, dev_c);

    HANDLE_ERROR( cudaMemcpy(   &c, 
                                dev_c,
                                sizeof(int),
                                cudaMemcpyDeviceToHost
                            ));

    printf( "2 + 7 = %d\n", c);
    cudaFree(dev_c);

    //kernel<<<1,1>>>();
    //printf("Hello, World!\n");
    return 0;

}