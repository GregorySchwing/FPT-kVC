#include "DCSR.h"

DCSR::DCSR(int size, int sparsity_factor, int numberOfSegments, int sizeOfSegments, int numberOfRows, int numberOfColumns):
SparseMatrix(size, numberOfRows, numberOfColumns), pitch(2*numberOfRows), alpha(sizeOfSegments){
    /* a single allocated segment of size μ per row */
    column_indices.assign(sizeOfSegments * numberOfRows, -1);
    values.assign(sizeOfSegments * numberOfRows, -1);
    row_offsets.resize(2 * numberOfRows);
    row_sizes.assign(numberOfRows, 0);

    for (int i = 0; i < row_offsets.size(); i++){
        if (i == 0){
            row_offsets[i] = 0;
        }   else if (i % 2 == 0) {
            row_offsets[i]+= sizeOfSegments;
        }   else {    
            row_offsets[i] = row_offsets[i-1];
        }
    }
}

DCSR::DCSR(const CSR & c, int sizeOfSegments):SparseMatrix(c), sizeOfSegments(sizeOfSegments), pitch(2*numberOfRows), alpha(sizeOfSegments){

    //column_indices.assign(sizeOfSegments * numberOfRows, -1);
    //values.assign(sizeOfSegments * numberOfRows, -1);    
    row_offsets.reserve(2 * numberOfRows);
    row_sizes.assign(numberOfRows, 0);

    allocateSegments(   getRowSizes(c),
                        CSRRowOffsetsToDCSRFormat(c),
                        c.column_indices,
                        c.values
                    );
}

void DCSR::allocateSegments(CSR & c){
    allocateSegments(   getRowSizes(c),
                        CSRRowOffsetsToDCSRFormat(c),
                        c.column_indices,
                        c.values
                    );
}
// TEST: If this is called from the constructor, it should allocate 1 segment per row in each array
// Serial
// A is expanded
// B is compressed
void DCSR::allocateSegments(    const std::vector<int> & B_sizes,
                                const std::vector<int> & B_offsets, 
                                const std::vector<int> & B_cols, 
                                const std::vector<int> & B_vals)
{
    // int row = vid; // vector ID // vectorized version

    //std::cout << toString();
    int row = 0;
    int addr;
    while(row < numberOfRows){
        //std::cout << "A_offsets.size() :" << A_offsets.size() << std::endl;
        if (row_offsets.size() < pitch){
            addr = row_offsets.back();
            // row start
            row_offsets.push_back(0);
            // row end
            row_offsets.push_back(0);
        }

        int sid = 0; // segment index
        int B_idx = 0; // Compressed index
        int A_start = row_offsets[row*2]; // Expanded starting segment offset
        int A_end = row_offsets[row*2 + 1]; // Expanded ending segment offset
        int free_mem = 0; // This should be equal to sizeOfSegment*2 at end of method
        int B_start = B_offsets[row]; // Compressed row index start
        int B_end = B_offsets[row*2 + 1]; // Compressed row index end
        int rlB = B_sizes[row]; // Compressed row size

        int rlA = row_sizes[row]; // Expanded row size
        int A_idx = 0; // thread row index, 0 since this is serial version
        // If row has entries
        if (rlA > 0) {
            // While index is less than length of row 
            while(A_idx < rlA){
                //Set index to end of current segment
                A_idx =+ (A_end - A_start);
                // If current segment doesn't contain the length
                if (A_idx < rlA){
                    // skip to next segment
                    sid++;
                    A_start = row_offsets[sid*pitch + row*2];
                    A_end = row_offsets[sid*pitch + row*2 + 1];
                }
            }
            A_idx = A_end + rlA - A_idx;
        } else {
            A_idx = A_start;
        }

        free_mem = A_end - A_idx;
        if(row_offsets.size() <= pitch || (free_mem < rlB && rlB > 0)){
            // allocate new space
            int new_mem_size = rlB - free_mem + alpha;
            if (row_offsets.size() > pitch){
                
                if(sid + 2 > row_offsets.size()/pitch){
                    int oldEnd =  row_offsets.size();
                    row_offsets.resize(2*numberOfRows + row_offsets.size());
                    std::fill(row_offsets.begin()+oldEnd, row_offsets.end(), -1);
                }

                addr = getInitializedBack(row_offsets);

                //allocate new row segment
                row_offsets[(sid+1)*pitch + row*2] = addr;

                // We hold on to addr here because back() will return -1 occasionally from now on
                row_offsets[(sid+1)*pitch + row*2 + 1] = addr + new_mem_size;       
                addr = addr + new_mem_size;          

            } else if (row_offsets.size() == pitch){

                //allocate final row segment of pitch 0
                row_offsets[row*2] = addr;
                row_offsets[row*2 + 1] = addr + new_mem_size;
                
                // Add my second logical rowOffsets row
                int oldEnd =  row_offsets.size();
                row_offsets.resize(2*numberOfRows + row_offsets.size());
                std::fill(row_offsets.begin()+oldEnd, row_offsets.end(), -1);

                // Allocate in col array
                oldEnd =  column_indices.size();
                column_indices.resize(addr + new_mem_size);
                std::fill(column_indices.begin()+addr, column_indices.end(), -1);

                // Allocate in values array
                oldEnd = values.size();
                values.resize(addr + new_mem_size);
                std::fill(values.begin()+addr, values.end(), -1); 

            } else {
                //allocate new row segment
                row_offsets[row*2] = addr;
                row_offsets[row*2 + 1] = addr + new_mem_size;

                // Allocate in col array
                int oldEnd =  column_indices.size();
                column_indices.resize(addr + new_mem_size);
                std::fill(column_indices.begin()+addr, column_indices.end(), -1);

                // Allocate in values array
                oldEnd = values.size();
                values.resize(addr + new_mem_size);
                std::fill(values.begin()+addr, values.end(), -1); 

            }
        }
        // allocate new entries (Alg 2)
        insertElements(row, B_offsets, B_cols, B_vals, B_sizes);
        //row = row + num_vectors;
        row++;
    }
    
}

void DCSR::insertElements(  int row,
                            const std::vector<int> & B_offsets,
                            const std::vector<int> & B_cols, 
                            const std::vector<int> & B_vals,
                            const std::vector<int> & B_sizes
                        ){
    int A_start = row_offsets[row*2]; // Expanded starting segment offset
    int A_end = row_offsets[row*2 + 1]; // Expanded ending segment offset
    int A_idx = row_offsets[row*2]; // Expanded starting segment offset
    int B_start = B_offsets[2*row];
    int B_end = B_offsets[row*2 + 1];
    int B_idx = B_start;
    int sid = 0;

    while(B_idx < B_end){
        if(A_idx >= A_end){
            int pos = A_idx - A_end;
            int sid = sid + 1;
            A_start = row_offsets[sid*pitch + row*2];
            A_end = row_offsets[sid*pitch + row*2 + 1];
            A_idx = A_start + pos;
        }   
        column_indices[A_idx] = B_cols[B_idx];
        values[A_idx] = B_vals[B_idx];

        B_idx++;
        A_idx++;
    }
    row_sizes[row] += B_sizes[row];
}

std::vector<int> DCSR::getRowSizes(const CSR & c){
    std::vector<int> row_sizes;
    row_sizes.resize(numberOfRows);
    for(int i = 0; i < numberOfRows; i++){
        row_sizes[i] = c.row_offsets[i+1] - c.row_offsets[i];
    }
    return row_sizes;
}

std::vector<int> DCSR::CSRRowOffsetsToDCSRFormat(const CSR & c){
    std::vector<int> dcsrOffs;
    dcsrOffs.resize(2 * numberOfRows);
        for(int i = 0; i < dcsrOffs.size(); i++){
        if (i % 2 == 0){
            dcsrOffs[i] = c.row_offsets[i/2];
        } else {
            dcsrOffs[i] = c.row_offsets[i/2 + 1];
        }
    }
    return dcsrOffs;
}

std::string DCSR::toString(){
    std::stringstream ss;
    std::string myMatrix;

    ss << "\t\tDCSR Matrix" << std::endl;

    ss << "Row offsets" << std::endl;
    for(int i = 0; i< row_offsets.size(); i++){
        if ( i != 0 && i % (numberOfRows*2) == 0)
            ss << std::endl;
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
    ss << "Row sizes" << std::endl;
    for(int i = 0; i< row_sizes.size(); i++){
        ss << "\t" << row_sizes[i];
    }
    ss << std::endl;
    myMatrix = ss.str();
    return myMatrix;
}

int DCSR::getInitializedBack(std::vector<int> offsets){
    for (std::vector<int>::reverse_iterator i = offsets.rbegin(); i != offsets.rend(); ++i ) {
        if ( *i != -1)
            return *i;
    } 
}
