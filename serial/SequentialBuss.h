#ifndef SEQUENTIALBUSS_H
#define SEQUENTIALBUSS_H

/* Vertex Cover Using Buss’ algorithm */
#include "Graph.h"
#include <iostream>

/* For : http://rosettacode.org/wiki/Combinations#C.2B.2B */
#include <algorithm>
#include <string>
/* For : http://rosettacode.org/wiki/Combinations#C.2B.2B */
class SequentialBuss {
    public:
        SequentialBuss(int k_arg, Graph & g_arg);
        ~SequentialBuss();
        void FindCover();
        void PrintVCSets();


    private:
		/* http://rosettacode.org/wiki/Combinations#C.2B.2B */
        void PopulateCombinations(int * combinations_arg, int N, int K);
        /* https://stackoverflow.com/questions/1505675/power-of-an-integer-in-c */
        int myPow(int x, unsigned int p);
};
#endif