#ifndef DEGREE_CONTROLLER_H
#define DEGREE_CONTROLLER_H

#include <list>
#include <vector>
#include <map>
#include <set>

#include "CSR.h"
#include "NeighborsBinaryDataStructure.h"
#include  <random>
#include  <iterator>

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
        int GetRandomVertex();

    private:
        std::list<DegreeNode> degreeController;
        std::vector< std::vector<int> > degContVecOfVecs;
        std::map< int, std::vector<int> > degMap;
        std::set<int> mapKeys;

        void AllocateVecofVecs(int numVerts);
        void CSR2VecofVecs(CSR * compressedSparseMatrix);
        void CreateDegreeController(NeighborsBinaryDataStructure * neighBits);
        void UpdateDegreeController(NeighborsBinaryDataStructure * neighBits);
        int numVerts;
        CSR * compressedSparseMatrixRef;

        template<typename RandomGenerator>
        int select_random_vertex(std::vector<int>::iterator start, std::vector<int>::iterator end, RandomGenerator& g);
        template<typename RandomGenerator>
        int select_random_degree(RandomGenerator& g);
        int select_random_degree();

};
#endif