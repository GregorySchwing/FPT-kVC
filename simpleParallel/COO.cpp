#include "COO.h"
#include <math.h>       /* floor */

COO::COO(int size, int numberOfRows, int numberOfColumns, bool populate):SparseMatrix(size, numberOfRows, numberOfColumns){
    new_column_indices.assign(size, -1);
    new_row_indices.assign(size, -1);
    new_values.resize(size);
    if (populate){
        int trialRow, trialCol;
        bool empty;
        for (int i = 0; i < size; i++){
            do {
                empty = true;
                trialCol = std::rand() % numberOfColumns;
                trialRow = std::rand() %  numberOfRows;
                for (int j = 0; j < i; j++){
                    if (new_row_indices[j] == trialRow && new_column_indices[j] == trialCol)
                        empty = false;
                }
            } while (!empty);
            new_row_indices[i] = trialRow;
            new_column_indices[i] = trialCol;
            new_values[i] =  std::rand() % size + 1;
        }
        //toString();
    }
}

COO::COO(int numberOfRows, int numberOfColumns):SparseMatrix(numberOfRows, numberOfColumns){
}

COO::COO():SparseMatrix(){
    std::cout << "Building COO from file/online" << std::endl;
}

/* Copy Constructor */
COO::COO(COO & coo_arg):SparseMatrix(coo_arg){
    new_column_indices = coo_arg.new_column_indices;
    new_row_indices = coo_arg.new_row_indices;
    isSorted = coo_arg.isSorted;
}


void COO::BuildCOOFromFile(std::string filename){
    char sep = ' ';
    std::ifstream file(filename);
    std::string::size_type sz;   // alias of size_t
    for(auto& row: CSVRange(file, sep))
    {
        //std::cout << "adding (" << std::stoi(row[0],&sz) 
        //<< ", " << std::stoi(row[1],&sz) << ")" << std::endl; 
        addEdgeSymmetric(std::stoi(row[0],&sz), 
                        std::stoi(row[1],&sz), 1);
        //coordinateFormat->addEdgeSimple(std::stoi(row[0],&sz), 
        //                                    std::stoi(row[1],&sz), 1);
    }

    size = new_values.size();
    // vlog(e)
    sortMyself();
}

void COO::BuildTheExampleCOO(){
    addEdgeSymmetric(0,1,1);
    addEdgeSymmetric(0,4,1);
    addEdgeSymmetric(1,4,1);
    addEdgeSymmetric(1,5,1);
    addEdgeSymmetric(1,6,1);
    addEdgeSymmetric(2,4,1);
    addEdgeSymmetric(2,6,1);
    addEdgeSymmetric(3,5,1);
    addEdgeSymmetric(3,6,1);
    addEdgeSymmetric(4,7,1);
    addEdgeSymmetric(4,8,1);
    addEdgeSymmetric(5,8,1);
    addEdgeSymmetric(6,9,1);

    size = new_values.size();
    // vlog(e)
    sortMyself();
}


void COO::SetVertexCountFromEdges(){
    int min;
    auto it = min_element(std::begin(new_row_indices), std::end(new_row_indices)); // C++11
    min = *it;
    it = max_element(std::begin(new_column_indices), std::end(new_column_indices)); // C++11
    if(min > *it)
        min = *it;
    if(min != 0){
        int scaleToRenumberAtZero = 0 - min;
        for (auto & v : new_row_indices)
            v += scaleToRenumberAtZero;
        for (auto & v : new_column_indices)
            v += scaleToRenumberAtZero;
    }
            
    int max;
    it = max_element(std::begin(new_row_indices), std::end(new_row_indices)); // C++11
    max = *it;
    it = max_element(std::begin(new_column_indices), std::end(new_column_indices)); // C++11
    if(max < *it)
        max = *it;

    SetNumberOfRows(max+1);
    vertexCount = max+1;

}


int COO::GetNumberOfVertices(){
    return vertexCount;
}

int COO::GetNumberOfEdges(){
    return new_column_indices.size();
}

void COO::addEdge(int u, int v, int weight, int edgeID){
    if(u <= v){
    new_row_indices[edgeID] = u;
    new_column_indices[edgeID] = v;
    new_values[edgeID] =  weight;
    } else {
        new_row_indices[edgeID] = v;
        new_column_indices[edgeID] = u;
        new_values[edgeID] =  weight;
    }
}
void COO::addEdgeSimple(int u, int v, int weight){
    new_row_indices.push_back(u);
    new_column_indices.push_back(v);
    new_values.push_back(weight);
}

/* Edges are stored once, but I need to do some fancy logic to get accurate degrees of vertices
    v > |V|/2
    One possibility is to use the 2D Bit Matrix to intersect 1 vs all, then call Count()
 */
void COO::addEdgeASymmetric(int u, int v, int weight){
    if(u <= v){
        new_row_indices.push_back(u);
        new_column_indices.push_back(v);
        new_values.push_back(weight);
    } else {
        new_row_indices.push_back(v);
        new_column_indices.push_back(u);
        new_values.push_back(weight);
    }
}


/* Works but the amount of data is 2x */
void COO::addEdgeSymmetric(int u, int v, int weight){
    new_row_indices.push_back(u);
    new_column_indices.push_back(v);
    new_values.push_back(weight);
    
    new_row_indices.push_back(v);
    new_column_indices.push_back(u);
    new_values.push_back(weight);
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
    new_row_indices.insert(new_row_indices.end(), c.new_row_indices.begin(), c.new_row_indices.end());
    new_column_indices.insert(new_column_indices.end(), c.new_column_indices.begin(), c.new_column_indices.end());
    new_values.insert(new_values.end(), c.new_values.begin(), c.new_values.end());
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
            if (new_row_indices[row_index] ==  i){
                if(j==new_column_indices[row_index]){
                    ss << "\t" << new_values[row_index];
                    // Skip duplicate entries
                    while(new_row_indices[row_index] == i && j == new_column_indices[row_index]){
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
    for(int i = 0; i< new_row_indices.size(); i++){
        ss << "\t" << new_row_indices[i];
    }
    ss << std::endl;
    ss << "Column indices" << std::endl;
    for(int i = 0; i< new_column_indices.size(); i++){
        ss << "\t" << new_column_indices[i];
    }
    ss << std::endl;
    ss << "new_values" << std::endl;
    for(int i = 0; i< new_values.size(); i++){
        ss << "\t" << new_values[i];
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
        min_row = new_row_indices[i];
        min_col = new_column_indices[i];
        min_val = new_values[i];
        for (int j = i; j < size; j++){
            if(new_row_indices[j] < min_row){
                temp_row = new_row_indices[j];
                temp_col = new_column_indices[j];
                temp_val = new_values[j];
                new_row_indices[j] = min_row;
                new_column_indices[j] = min_col;
                new_values[j] = min_val;
                new_row_indices[i] = temp_row;
                new_column_indices[i] = temp_col;
                new_values[i] = temp_val;

                min_row = new_row_indices[i];
                min_col = new_column_indices[i];
                min_val = new_values[i];
            }
        }
    }
    
//  Sort within rows by col
    for(int i = 0; i < size; i++){
        min_row = new_row_indices[i];
        min_col = new_column_indices[i];
        min_val = new_values[i];
        for (int j = i; j < size; j++){
            if(min_row == new_row_indices[j] && new_column_indices[j] < min_col){
                temp_row = new_row_indices[j];
                temp_col = new_column_indices[j];
                temp_val = new_values[j];
                new_row_indices[j] = min_row;
                new_column_indices[j] = min_col;
                new_values[j] = min_val;
                new_row_indices[i] = temp_row;
                new_column_indices[i] = temp_col;
                new_values[i] = temp_val;

                min_row = new_row_indices[i];
                min_col = new_column_indices[i];
                min_val = new_values[i];            
            }
        }
    }
    isSorted = true;
}

//COO& COO::SpMV(COO & c){

//}