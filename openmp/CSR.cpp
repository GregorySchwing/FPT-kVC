#include "CSR.h"

/* Must be sorted, which our COO constructor does by default */
CSR::CSR(COO & c):
// Sets values_ref(c.values) in SM constructor
SparseMatrix(c),
// Didn't exist in COO or SM so we have a cicular ref
old_row_offsets_ref(new_row_offsets),
// Reference to COO col inds
old_column_indices_ref(c.new_column_indices)
{
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
    std::cout << "nros" << std::endl;
    for (auto & v : new_row_offsets){
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

/* Building the next graph */
CSR::CSR(CSR & c):
SparseMatrix(numberOfRows, c.new_values),
old_row_offsets_ref(c.new_row_offsets),
old_column_indices_ref(c.new_column_indices)
{
 
}

/* For B1 to B1 transition */
CSR::CSR(int numberOfRows,
         CSR & c,
         std::vector<int> & values_ref_arg) :
// Creates the local copy of values vector of size : values_ref.size() 
SparseMatrix(numberOfRows, values_ref_arg),
old_row_offsets_ref(c.new_row_offsets),
old_column_indices_ref(c.new_column_indices)
{
    // Nothing to do
}


std::string CSR::toString(){
    std::stringstream ss;
    std::string myMatrix;

    ss << "\t\tCSR Matrix" << std::endl;

    ss << "Row offsets" << std::endl;
    for(int i = 0; i< old_row_offsets_ref.size(); i++){
        ss << "\t" << old_row_offsets_ref[i];
    }
    ss << std::endl;
    ss << "Column indices" << std::endl;
    for(int i = 0; i< old_column_indices_ref.size(); i++){
        ss << "\t" << old_column_indices_ref[i];
    }
    ss << std::endl;
    ss << "values" << std::endl;
    for(int i = 0; i< old_values_ref.size(); i++){
        ss << "\t" << old_values_ref[i];
    }
    ss << std::endl;
    myMatrix = ss.str();
    return myMatrix;
}


std::vector<int> & CSR::GetOldRowOffRef(){
    return old_row_offsets_ref;
}

std::vector<int> & CSR::GetOldColRef(){
    return old_column_indices_ref;
}

std::vector<int> & CSR::GetOldValRef(){
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
