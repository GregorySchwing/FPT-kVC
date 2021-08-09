#include "CSR.h"

/* Must be sorted, which our COO constructor does by default */
CSR::CSR():
// Sets values_ref(c.values) in SM constructor
SparseMatrix(),
// Didn't exist in COO or SM so we have a cicular ref
old_row_offsets_ref(&new_row_offsets),
// Reference to COO col inds
old_column_indices_ref(&new_column_indices)
{
    
}


/* Must be sorted, which our COO constructor does by default */
CSR::CSR(COO & c):
vertexCount(c.vertexCount),
// Sets values_ref(c.values) in SM constructor
SparseMatrix(c),
// Didn't exist in COO or SM so we have a cicular ref
old_row_offsets_ref(&new_row_offsets),
// Reference to COO col inds
old_column_indices_ref(&c.new_column_indices)
{
    // Only make a copy of the column indices
    // When going from COO to CSR
    new_column_indices = c.new_column_indices;

    new_row_offsets.resize(numberOfRows + 1);
    int row_size = 0;
    int row = 0;
    int index = 0;
    new_row_offsets[0] = 0;
    while(row < numberOfRows){
        if(c.new_row_indices[index] == row){
            row_size++;
            index++;
        } else {
            new_row_offsets[row+1] = row_size;
            row++;
        }
    }
}

/* Building the first graph */
CSR::CSR(CSR & c):
SparseMatrix(c),
old_row_offsets_ref(&c.new_row_offsets),
old_column_indices_ref(&c.new_column_indices)
{
    new_row_offsets = c.new_row_offsets;
    new_column_indices = c.new_column_indices;
}


/* Copy constructor */
CSR::CSR(const CSR & c):
SparseMatrix(c)
{
    new_row_offsets.reserve(c.new_row_offsets.capacity());
    new_column_indices.reserve(c.new_column_indices.capacity());
    std::cout << "numberOfRows " << c.numberOfRows << std::endl;
    std::cout << "c.new_column_indices.size() " << c.new_column_indices.size()  << std::endl;
    std::cout << "c.new_column_indices.capacity() " << c.new_column_indices.capacity()  << std::endl;

}

void CSR::SetOldRowOffRef(std::vector<int> & old_arg){
    old_row_offsets_ref = &old_arg;
}
void CSR::SetOldColRef(std::vector<int> & old_arg){
    old_column_indices_ref = &old_arg;
}
void CSR::SetOldValRef(std::vector<int> & old_arg){
    old_values_ref = &old_arg;
}

void CSR::PopulateNewVals(int edgesLeftToCover){
    int ONE = 1;
    std::vector<int> & new_vals = GetNewValRef();
    for (int i = 0; i < edgesLeftToCover*2; ++i)
        new_vals.push_back(ONE);
}

void CSR::PopulateNewRefs(int edgesLeftToCover){
    int ZERO = 0;
    int ONE = 1;
    // See comment below
    //std::vector<int> & new_row_offs = GetNewRowOffRef();
    std::vector<int> & new_col_vals = GetNewColRef();
    std::vector<int> & new_vals = GetNewValRef();

    for (int i = 0; i < edgesLeftToCover; ++i){
        //new_row_offs are calculated sequentially
        // in CalculateNewRowOffs()
        // so we dont need to push back zeros
        // If we parallelize this through diffs from
        // old, i.e. an n-array of the number of
        // edges removed from each vertex
        // we can add this line back
        //new_row_offs.push_back(ZERO);
        new_col_vals.push_back(ZERO);
        //new_vals.push_back(ONE);
        // Once we test the CSPRV just push back all ones
        // and no longer write the actual val
        new_vals.push_back(ZERO);
    }
}


std::string CSR::toString(){
    std::stringstream ss;
    std::string myMatrix;

    ss << "\t\tCSR Matrix" << std::endl;

    ss << "Row offsets" << std::endl;
    for(int i = 0; i< old_row_offsets_ref->size(); i++){
        ss << "\t" << (*old_row_offsets_ref)[i];
    }
    ss << std::endl;
    ss << "Column indices" << std::endl;
    for(int i = 0; i< old_column_indices_ref->size(); i++){
        ss << "\t" << (*old_column_indices_ref)[i];
    }
    ss << std::endl;
    ss << "values" << std::endl;
    for(int i = 0; i< old_values_ref->size(); i++){
        ss << "\t" << (*old_values_ref)[i];
    }
    ss << std::endl;
    myMatrix = ss.str();
    return myMatrix;
}

std::vector<int> * CSR::GetOldRowOffRef(){
    return old_row_offsets_ref;
}

std::vector<int> * CSR::GetOldColRef(){
    return old_column_indices_ref;
}

std::vector<int> * CSR::GetOldValRef(){
    return old_values_ref;
}

std::vector<int> & CSR::GetNewRowOffRef(){
    return new_row_offsets;
}

std::vector<int> & CSR::GetNewColRef(){
    return new_column_indices;
}

std::vector<int> & CSR::GetNewValRef(){
    return new_values;
}
