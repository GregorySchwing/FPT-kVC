#include "DegreeController.h"

DegreeController::DegreeController(int numVerts_arg, NeighborsBinaryDataStructure * neighBits) : numVerts(numVerts_arg){
    AllocateVecofVecs(numVerts);
    CreateDegreeController(neighBits);
}

DegreeController::DegreeController(int numVerts_arg, CSR * compressedSparseMatrix) : 
numVerts(numVerts_arg), 
compressedSparseMatrixRef(compressedSparseMatrix)
{
    AllocateVecofVecs(numVerts);
    CSR2VecofVecs(compressedSparseMatrix);
}

/* Replace this with N bitsets of N size */
/* This will take N*std::allocator space */
void DegreeController::CSR2VecofVecs(CSR * compressedSparseMatrix){
    degContVecOfVecs.resize(compressedSparseMatrix->numberOfRows);
    // Simple degree arimetic for the first half of the vertices
    for (int i = 0; i < compressedSparseMatrix->numberOfRows; ++i){
        degContVecOfVecs[compressedSparseMatrix->row_offsets[i+1] - compressedSparseMatrix->row_offsets[i]].push_back(i);
    }
}


/* Replace this with N bitsets of N size */
/* This will take N*std::allocator space */
void DegreeController::AllocateVecofVecs(int numVerts){
    degContVecOfVecs.resize(numVerts);
}

std::string DegreeController::toString(){
    std::stringstream ss;
    std::string myMatrix;
    ss << "Degree Controller" << std::endl;
    for(int i = 0; i < degContVecOfVecs.size(); i++){
        ss << "\t" << i << " : ";
        for(int j = 0; j < degContVecOfVecs[i].size(); ++j)
            ss << "\t" << degContVecOfVecs[i][j];
        ss << std::endl;
    }
    ss << std::endl;
    myMatrix = ss.str();
    return myMatrix;
}

std::vector< std::vector<int> > & DegreeController::GetTempDegCont(){
    return degContVecOfVecs;
}

void DegreeController::CreateDegreeController(NeighborsBinaryDataStructure * neighBits){
    for (int i = 0; i < numVerts; ++i)
        degContVecOfVecs[neighBits->GetDegree(i)].push_back(i);
}

void DegreeController::UpdateDegreeController(NeighborsBinaryDataStructure * neighBits){
    degContVecOfVecs.clear();
    for (int i = 0; i < numVerts; ++i)
        degContVecOfVecs[neighBits->GetDegree(i)].push_back(i);
}

int DegreeController::GetRandomVertex(){
    int r;
    return r = select_random_degree(degContVecOfVecs.begin(), degContVecOfVecs.end());
}

template<typename RandomGenerator>
int DegreeController::select_random_vertex(std::vector<int>::iterator start, std::vector<int>::iterator end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return *start;
}

template<typename RandomGenerator>
int DegreeController::select_random_degree(std::vector<std::vector<int>>::iterator start, std::vector<std::vector<int>>::iterator end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return select_random_vertex(start->begin(), start->end(), g);
}

int DegreeController::select_random_degree(std::vector<std::vector<int>>::iterator start, std::vector<std::vector<int>>::iterator end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_random_degree(start, end, gen);
}




