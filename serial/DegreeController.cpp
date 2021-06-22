#include "DegreeController.h"

DegreeController::DegreeController(int numVerts_arg, NeighborsBinaryDataStructure * neighBits) : numVerts(numVerts_arg){
    AllocateVecofVecs(numVerts);
    CreateDegreeController(neighBits);
}

DegreeController::DegreeController(int numVerts_arg, CSR * compressedSparseMatrix) : 
numVerts(numVerts_arg)
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
        degMap[compressedSparseMatrix->row_offsets[i+1] - compressedSparseMatrix->row_offsets[i]].push_back(i);
        mapKeys.insert(compressedSparseMatrix->row_offsets[i+1] - compressedSparseMatrix->row_offsets[i]);
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
    return r = select_random_degree();
}

template<typename RandomGenerator>
int DegreeController::select_random_vertex(std::vector<int>::iterator start, std::vector<int>::iterator end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    //std::cout << "random vertex " << *start << std::endl;
    return *start;
}

template<typename RandomGenerator>
int DegreeController::select_random_degree(RandomGenerator& g) {
    std::cout << "Map size " << mapKeys.size() << std::endl;

    //std::uniform_int_distribution<> dis(0, mapKeys.size() - 1);
    /* Skip 0 so we don't select vertices of degree 0 */
    std::uniform_int_distribution<> dis(1, mapKeys.size() - 1);

    auto start = mapKeys.begin();
    std::advance(start, dis(g));
    std::cout << "Deg " << *start << std::endl;
    //return *start;
    return select_random_vertex(degMap[*start].begin(), degMap[*start].end(), g);
}

int DegreeController::select_random_degree() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_random_degree(gen);
}




