#include "CSR.h"

CSR::CSR(const COO & c):SparseMatrix(c){
    row_offsets.resize(numberOfRows + 1);
    column_indices.resize(size);
    values.resize(size);
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
    column_indices = c.column_indices;
    values = c.values;    
}

CSR::CSR(const CSR & c):SparseMatrix(c){
    row_offsets = c.row_offsets;
    column_indices = c.column_indices;
    values = c.values;    
}

/* For post-kernelization G' induced subgraph */
CSR::CSR(const CSR & c, int edgesLeftToCover):SparseMatrix(c, edgesLeftToCover){
    row_offsets.reserve(c.numberOfRows + 1);
    column_indices.reserve(edgesLeftToCover);
    int count = 0;
    row_offsets.push_back(count);
    for (int i = 0; i < c.numberOfRows + 1; ++i){
        for (int j = c.row_offsets[i]; j < c.row_offsets[i+1]; ++j)
            if (c.values[j] != 0){
                ++count; 
                column_indices.push_back(c.column_indices[j]);
                values.push_back(c.values[j]);
            }
        row_offsets.push_back(count);
    }
}

/* For branch owned G'' induced subgraph */
CSR::CSR(const CSR & c, std::vector<int> & verticesToDelete, std::vector<int> valuesToModify){
    row_offsets.reserve(c.numberOfRows + 1);
    for (auto & v: verticesToDelete)
        removeVertexEdges(v, valuesToModify);

        
    
    column_indices.reserve(edgesLeftToCover);
    int count = 0;
    row_offsets.push_back(count);
    for (int i = 0; i < c.numberOfRows + 1; ++i){
        for (int j = c.row_offsets[i]; j < c.row_offsets[i+1]; ++j)
            if (valuesToModify[j] != 0){
                ++count; 
                column_indices.push_back(c.column_indices[j]);
                values.push_back(valuesToModify[j]);
            }
        row_offsets.push_back(count);
    }
}

void CSR::removeVertexEdges(int u){
    int v, i, j;
    /* i - out going vertices of u */
    for (i = 0; i < row_offsets[u+1]-row_offsets[u]; ++i){
        /* Get neighbor vertex */
        v = column_indices[row_offsets[u]+i];
        /* Set (u,v) to 0 */
        values[row_offsets[u] + i] = 0;

        j = 0;
        /* Find u in v's list of edges */
        /* j - out going vertices of v */

        while (column_indices[row_offsets[v] + j] != u){
            ++j;
        }
        /* Set (v,u) to 0 */
        values[row_offsets[v] + j] = 0;
    }
}


void CSR::removeVertexEdges(int u, std::vector<int> & valuesToModify){
    int v, i, j;
    /* i - out going vertices of u */
    for (i = 0; i < row_offsets[u+1]-row_offsets[u]; ++i){
        /* Get neighbor vertex */
        v = column_indices[row_offsets[u]+i];
        /* Set (u,v) to 0 */
        valuesToModify[row_offsets[u] + i] = 0;

        j = 0;
        /* Find u in v's list of edges */
        /* j - out going vertices of v */

        while (column_indices[row_offsets[v] + j] != u){
            ++j;
        }
        /* Set (v,u) to 0 */
        valuesToModify[row_offsets[v] + j] = 0;
    }
}

void CSR::insertElements(const SparseMatrix & s){

    const CSR & csr = castSparseMatrix(s);

    std::vector<int> new_row_offsets(numberOfRows + 1);
    std::vector<int> new_column_indices = column_indices;
    std::vector<int> new_values = values;
    
    int row_size = 0;
    int row = 0;
    int index = 0;
    new_row_offsets = row_offsets + csr.row_offsets;

    while(row < numberOfRows){
        new_column_indices.insert(new_column_indices.begin()+new_row_offsets[row], (csr.column_indices.begin()+csr.row_offsets[row]), (csr.column_indices.begin()+csr.row_offsets[row+1]));
        new_values.insert(new_values.begin()+new_row_offsets[row], (csr.values.begin()+csr.row_offsets[row]), (csr.values.begin()+csr.row_offsets[row+1]));
        row++;
    }

    row_offsets = new_row_offsets;
    column_indices = new_column_indices;
    values = new_values;

    size += csr.size;
}

const CSR & CSR::castSparseMatrix(const SparseMatrix & s){
    try {
        const COO & coo = dynamic_cast<const COO&>(s);
        const CSR * csr = new CSR(coo);
        return *csr;
    }
    catch(const std::bad_cast& e) {
        try {
            const CSR & csr = dynamic_cast<const CSR&>(s);
            return csr;
        }
        catch(const std::bad_cast& e) {
            std::cout << "wrong type, expecting type COO or CSR.\n";
            exit(1);
        }
    }
}



std::string CSR::toString(){
    std::stringstream ss;
    std::string myMatrix;

    ss << "\t\tCSR Matrix" << std::endl;

    ss << "Row offsets" << std::endl;
    for(int i = 0; i< row_offsets.size(); i++){
        ss << "\t" << row_offsets[i];
    }
    ss << std::endl;
    ss << "Column indices" << std::endl;
    for(int i = 0; i< column_indices.size(); i++){
        ss << "\t" << column_indices[i];
    }
    ss << std::endl;
    ss << "values" << std::endl;
    for(int i = 0; i< values.size(); i++){
        ss << "\t" << values[i];
    }
    ss << std::endl;
    myMatrix = ss.str();
    return myMatrix;
}