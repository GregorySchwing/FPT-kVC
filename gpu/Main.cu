#include <cuda.h>
int main(int argc, char *argv[])
{
    int N = atoi(argv[1]);
    int numEntries = atoi(argv[2]);
    int column_indices_a[numEntries], row_indices_a[numEntries], values_a[numEntries];
    int * column_indices_a_dev, * row_indices_a_dev, * values_a_dev;

    cudaMalloc( (void**)&column_indices_a_dev, numEntries * sizeof(int) );
    cudaMalloc( (void**)&row_indices_a_dev, numEntries * sizeof(int) );
    cudaMalloc( (void**)&values_a_dev, numEntries * sizeof(int) );

    // fill the arrays ‘a’ and ‘b’ on the CPU
    for (int i=0; i<numEntries; i++) {
        column_indices_a[i] = rand() % N;
        row_indices_a[i] = rand() % N;
        values_a[i] = rand() % N;
    }

    cudaMemcpy(column_indices_a_dev, column_indices_a, numEntries * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(row_indices_a_dev, row_indices_a, numEntries * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(values_a_dev, values_a, numEntries * sizeof(int), cudaMemcpyHostToDevice);



}