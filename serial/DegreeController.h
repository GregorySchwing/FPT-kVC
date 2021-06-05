#ifndef DEGREE_CONTROLLER_H
#define DEGREE_CONTROLLER_H

#include <list>
#include <vector>
#include "CSR.h"

class DegreeController{
    
    struct DegreeNode {
        std::list<int> listOfNodes;
        int degree;
    };
    public:
        DegreeController(CSR * compressedSparseMatrix);
        std::string toString();
        std::vector< std::vector<int> > & GetTempDegCont();

    private:
        std::list<DegreeNode> degreeController;
        std::vector< std::vector<int> > temporaryDegCont;
        void CSR2VecofVecs(CSR * compressedSparseMatrix);

};
#endif