#include "DegreeController.h"

DegreeController::DegreeController(CSR * compressedSparseMatrix){
    CSR2VecofVecs(compressedSparseMatrix);
}

/* Replace this with N bitsets of N size */
/* This will take N*std::allocator space */
void DegreeController::CSR2VecofVecs(CSR * compressedSparseMatrix){
    temporaryDegCont.resize(compressedSparseMatrix->numberOfRows);
    /* Simple degree arimetic for the first half of the vertices */
    for (int i = 0; i < compressedSparseMatrix->numberOfRows; ++i){
        temporaryDegCont[compressedSparseMatrix->row_offsets[i+1] - compressedSparseMatrix->row_offsets[i]].push_back(i);
    }
    //compressedSparseMatrix->column_indices

}

std::string DegreeController::toString(){
    std::stringstream ss;
    std::string myMatrix;
    ss << "Degree Controller" << std::endl;
    for(int i = 0; i < temporaryDegCont.size(); i++){
        ss << "\t" << i << " : ";
        for(int j = 0; j < temporaryDegCont[i].size(); ++j)
            ss << "\t" << temporaryDegCont[i][j];
        ss << std::endl;
    }
    ss << std::endl;
    myMatrix = ss.str();
    return myMatrix;
}

std::vector< std::vector<int> > & DegreeController::GetTempDegCont(){
    return temporaryDegCont;
}


