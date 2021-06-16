#include "COO.h"
#include <math.h>       /* floor */

COO::COO(int size, int numberOfRows, int numberOfColumns, bool populate):SparseMatrix(size, numberOfRows, numberOfColumns){
    column_indices.assign(size, -1);
    row_indices.assign(size, -1);
    values.resize(size);
    if (populate){
        int trialRow, trialCol;
        bool empty;
        for (int i = 0; i < size; i++){
            do {
                empty = true;
                trialCol = std::rand() % numberOfColumns;
                trialRow = std::rand() %  numberOfRows;
                for (int j = 0; j < i; j++){
                    if (row_indices[j] == trialRow && column_indices[j] == trialCol)
                        empty = false;
                }
            } while (!empty);
            row_indices[i] = trialRow;
            column_indices[i] = trialCol;
            values[i] =  std::rand() % size + 1;
        }
        toString();
    }
}

COO::COO(int numberOfRows, int numberOfColumns):SparseMatrix(numberOfRows, numberOfColumns){
}

COO::COO(COO & coo_arg):SparseMatrix(coo_arg){
    column_indices = coo_arg.column_indices;
    row_indices = coo_arg.row_indices;
    isSorted = coo_arg.isSorted;
}


void COO::addEdge(int u, int v, int weight, int edgeID){
    if(u <= v){
    row_indices[edgeID] = u;
    column_indices[edgeID] = v;
    values[edgeID] =  weight;
    } else {
        row_indices[edgeID] = v;
        column_indices[edgeID] = u;
        values[edgeID] =  weight;
    }
}

/* Edges are stored once, but I need to do some fancy logic to get accurate degrees of vertices
    v > |V|/2
    One possibility is to use the 2D Bit Matrix to intersect 1 vs all, then call Count()
 */
void COO::addEdgeASymmetric(int u, int v, int weight){
    if(u <= v){
        row_indices.push_back(u);
        column_indices.push_back(v);
        values.push_back(weight);
    } else {
        row_indices.push_back(v);
        column_indices.push_back(u);
        values.push_back(weight);
    }
}


/* Works but the amount of data is 2x */
void COO::addEdgeSymmetric(int u, int v, int weight){
    row_indices.push_back(u);
    column_indices.push_back(v);
    values.push_back(weight);
    
    row_indices.push_back(v);
    column_indices.push_back(u);
    values.push_back(weight);
}


void COO::insertElements(const SparseMatrix & s){
    try {
        const COO &c = dynamic_cast<const COO&>(c);
    }
    catch(const std::bad_cast& e) {
        std::cout << "wrong type, expecting type COO\n";    
        exit(1);
    }
    const COO& c =  dynamic_cast<const COO&> (s);
    row_indices.insert(row_indices.end(), c.row_indices.begin(), c.row_indices.end());
    column_indices.insert(column_indices.end(), c.column_indices.begin(), c.column_indices.end());
    values.insert(values.end(), c.values.begin(), c.values.end());
    size+=c.size;
    sortMyself();
}

std::string COO::toString(){
    //if(!isSorted)
    //    sortMyself();
    std::stringstream ss;
    std::string myMatrix;
    ss << "\t\tCOO Matrix" << std::endl;
    for (int i = 0; i<numberOfColumns; i++){
        ss << "\tcol " << i;
    }
    ss << std::endl;
    int row_index = 0;
    for (int i = 0; i < numberOfRows; i++){
        ss << "row " << i;
        for( int j = 0; j < numberOfColumns; j++){
            if (row_indices[row_index] ==  i){
                if(j==column_indices[row_index]){
                    ss << "\t" << values[row_index];
                    // Skip duplicate entries
                    while(row_indices[row_index] == i && j == column_indices[row_index]){
                        row_index++;
                    }
                } else {
                    ss << "\t" << 0;
                }
            } else {
                ss << "\t" << 0;
            }
        }        
        ss << std::endl;
    }
    ss << "Row indices" << std::endl;
    for(int i = 0; i< row_indices.size(); i++){
        ss << "\t" << row_indices[i];
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

void COO::sortMyself(){
    float min_val, temp_val;
    int temp_row, temp_col, min_row, min_col;
// Sort by rows
    for(int i = 0; i < size; i++){
        min_row = row_indices[i];
        min_col = column_indices[i];
        min_val = values[i];
        for (int j = i; j < size; j++){
            if(row_indices[j] < min_row){
                temp_row = row_indices[j];
                temp_col = column_indices[j];
                temp_val = values[j];
                row_indices[j] = min_row;
                column_indices[j] = min_col;
                values[j] = min_val;
                row_indices[i] = temp_row;
                column_indices[i] = temp_col;
                values[i] = temp_val;

                min_row = row_indices[i];
                min_col = column_indices[i];
                min_val = values[i];
            }
        }
    }
    
//  Sort within rows by col
    for(int i = 0; i < size; i++){
        min_row = row_indices[i];
        min_col = column_indices[i];
        min_val = values[i];
        for (int j = i; j < size; j++){
            if(min_row == row_indices[j] && column_indices[j] < min_col){
                temp_row = row_indices[j];
                temp_col = column_indices[j];
                temp_val = values[j];
                row_indices[j] = min_row;
                column_indices[j] = min_col;
                values[j] = min_val;
                row_indices[i] = temp_row;
                column_indices[i] = temp_col;
                values[i] = temp_val;

                min_row = row_indices[i];
                min_col = column_indices[i];
                min_val = values[i];            
            }
        }
    }
    isSorted = true;
}

//COO& COO::SpMV(COO & c){

//}