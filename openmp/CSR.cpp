#include "CSR.h"

/* Must be sorted, which our COO constructor does by default */
CSR::CSR(COO & c):
// Sets values_ref(c.values) in SM constructor
SparseMatrix(c),
// Didn't exist in COO or SM so we have a cicular ref
row_offsets_ref(row_offsets),
// Reference to COO col inds
column_indices_ref(c.column_indices)
{
    row_offsets.resize(numberOfRows + 1);
    int row_size = 0;
    int row = 0;
    int index = 0;
    row_offsets[0] = 0;
    while(row < numberOfRows){
        if(c.row_indices[index] == row){
            row_size++;
            index++;
        } else {
            row_offsets[row+1] = row_size;
            row++;
        }
    }
}

CSR::CSR(CSR & c):SparseMatrix(c),
column_indices_ref(column_indices),
row_offsets_ref(row_offsets){
    row_offsets = c.row_offsets;
    column_indices = c.column_indices;
    values = c.values;    
}

/* For K to B1 transition */
CSR::CSR(int numberOfRows,
        std::vector<int> & row_offsets_ref,
        std::vector<int> & column_indices_ref,
        std::vector<int> & values_ref):
// Creates the local copy of values vector of size : values_ref.size() 
SparseMatrix(numberOfRows, values_ref),
row_offsets_ref(row_offsets_ref),
column_indices_ref(column_indices_ref)
{
    // Nothing to do
}

/* For B1 to B1 transition */
CSR::CSR(int numberOfRows,
         CSR & c,
         std::vector<int> & values_ref_arg) :
// Creates the local copy of values vector of size : values_ref.size() 
SparseMatrix(numberOfRows, values_ref_arg),
row_offsets_ref(c.row_offsets_ref),
column_indices_ref(c.column_indices_ref)
{
    // Nothing to do
}


std::string CSR::toString(){
    std::stringstream ss;
    std::string myMatrix;

    ss << "\t\tCSR Matrix" << std::endl;

    ss << "Row offsets" << std::endl;
    for(int i = 0; i< row_offsets_ref.size(); i++){
        ss << "\t" << row_offsets_ref[i];
    }
    ss << std::endl;
    ss << "Column indices" << std::endl;
    for(int i = 0; i< column_indices_ref.size(); i++){
        ss << "\t" << column_indices_ref[i];
    }
    ss << std::endl;
    ss << "values" << std::endl;
    for(int i = 0; i< values_ref.size(); i++){
        ss << "\t" << values_ref[i];
    }
    ss << std::endl;
    myMatrix = ss.str();
    return myMatrix;
}


std::vector<int> & CSR::GetOldRowOffRef(){
    return row_offsets_ref;
}

std::vector<int> & CSR::GetOldColRef(){
    return column_indices_ref;
}

std::vector<int> & CSR::GetOldValRef(){
    return values_ref;
}

std::vector<int> & CSR::GetNewRowOffRef(){
    return row_offsets;
}

std::vector<int> & CSR::GetNewColRef(){
    return column_indices;
}

std::vector<int> & CSR::GetNewValRef(){
    return values;
}
