#ifndef DEGREE_CONTROLLER_H
#define DEGREE_CONTROLLER_H

#include <list>
#include <vector>
#include "CSR.h"
#include "NeighborsBinaryDataStructure.h"

class DegreeController{
    
    struct DegreeNode {
        std::list<int> listOfNodes;
        int degree;
    };
    public:
        DegreeController(int numVerts, NeighborsBinaryDataStructure * neighBits);
        DegreeController(int numVerts, CSR * compressedSparseMatrix);

        std::string toString();
        std::vector< std::vector<int> > & GetTempDegCont();

    private:
        std::list<DegreeNode> degreeController;
        std::vector< std::vector<int> > degContVecOfVecs;
        void AllocateVecofVecs(int numVerts);
        void CSR2VecofVecs(CSR * compressedSparseMatrix);
        void CreateDegreeController(NeighborsBinaryDataStructure * neighBits);
        void UpdateDegreeController();
        int numVerts;
        CSR * compressedSparseMatrixRef;
        //void CSR2VecofVecs(CSR * compressedSparseMatrix);

};
#endif