#include "COO.cuh"
#include "CSR.cuh"
#include <math.h>       /* floor */
#include <cuda.h>

CSR::CSR(const COO & c){
    row_offset.resize(c.dimensions + 1);
    col_vec.resize(c.numberOfEntries);
    val_vec.resize(c.numberOfEntries);
    int row_size = 0;
    int row = 0;
    int index = 0;
    row_offset[0] = 0;
    while(row < c.dimensions){
        if(c.row_vec[index] == row){
            row_size++;
            index++;
        } else {
            row_offset[row+1] = row_size;
            row++;
        }
    }
    col_vec = c.col_vec;
    val_vec = c.val_vec;    
}

std::string CSR::toString(){
    std::stringstream ss;
    std::string myMatrix;

    ss << "\t\tCSR Matrix" << std::endl;

    ss << "Row offsets" << std::endl;
    for(int i = 0; i< row_offset.size(); i++){
        ss << "\t" << row_offset[i];
    }
    ss << std::endl;
    ss << "Column indices" << std::endl;
    for(int i = 0; i< col_vec.size(); i++){
        ss << "\t" << col_vec[i];
    }
    ss << std::endl;
    ss << "values" << std::endl;
    for(int i = 0; i< val_vec.size(); i++){
        ss << "\t" << val_vec[i];
    }
    ss << std::endl;
    myMatrix = ss.str();
    return myMatrix;
}


  