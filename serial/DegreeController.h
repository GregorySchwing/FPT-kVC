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
        std::string toString();
        std::vector< std::vector<int> > & GetTempDegCont();

    private:
        std::list<DegreeNode> degreeController;
        std::vector< std::vector<int> > temporaryDegCont;
        void CreateDegreeController(NeighborsBinaryDataStructure * neighBits);


        //void CSR2VecofVecs(CSR * compressedSparseMatrix);

};
#endif