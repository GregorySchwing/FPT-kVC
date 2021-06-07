#include "DegreeController.h"

DegreeController::DegreeController(int numVerts_arg, NeighborsBinaryDataStructure * neighBits) : numVerts(numVerts_arg){
    AllocateVecofVecs(numVerts);
    CreateDegreeController(neighBits);
}

/* Replace this with N bitsets of N size */
/* This will take N*std::allocator space 
void DegreeController::CSR2VecofVecs(CSR * compressedSparseMatrix){
    temporaryDegCont.resize(compressedSparseMatrix->numberOfRows);
    // Simple degree arimetic for the first half of the vertices
    for (int i = 0; i < compressedSparseMatrix->numberOfRows; ++i){
        temporaryDegCont[compressedSparseMatrix->row_offsets[i+1] - compressedSparseMatrix->row_offsets[i]].push_back(i);
    }
}
*/

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




