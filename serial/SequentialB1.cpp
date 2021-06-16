#include "SequentialB1.h"

SequentialB1::SequentialB1( Graph & g_arg, 
                            SequentialKernelization & sk_arg,
                            int k_arg, 
                            int k_prime_arg):
                            g(g_arg), sk(sk_arg), k(k_arg), k_prime(k_prime_arg){
    for (auto v : sk.GetS())
        g.GetCSR()->removeVertexEdges(v);
}