#ifndef Sequential_DF_Kernelization_H
#define Sequential_DF_Kernelization_H

#include "Graph.h"
#include "SequentialKernelization.h"

#include <iostream>

class SequentialDFKernelization{
    SequentialDFKernelization(Graph & g_arg, SequentialKernelization & sk, int k_arg);
};

#endif