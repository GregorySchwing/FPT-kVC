/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.70
Copyright (C) 2018  GOMC Group
A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/

/* Courtesy of https://www.softwaretestinghelp.com/graph-implementation-cpp/ */

#include "ConnectednessTest.cuh"
#include "Graph.cuh"

#define debugAdjList 0 

//return if we fail to read anything
const int READERROR = -1;

// insert new nodes into adjacency list from given graph
adjNode* ConnectednessTest::getAdjListNode(int value, int weight, adjNode* head)   {
    adjNode* newNode = new adjNode;
    newNode->val = value;
    newNode->cost = weight;
    newNode->next = head;   // point new node to current head
    return newNode;
}

ConnectednessTest::ConnectednessTest(COO & coo, std::vector< std::vector<int> > & vectorOfConnectedComponents){
    
    this->nNodes = coo.vertexCount;
    this->nEdges = coo.GetNumberOfEdges();


    unsigned int atom0, atom1;
    int dummy;
    // Loads all the bonds into edges array
    
    std::vector<int> & rows_ref = coo.new_row_indices;
    std::vector<int> & cols_ref = coo.new_column_indices;
    
    // allocate new node
    head = new adjNode*[nNodes]();
    // initialize head pointer for all vertices
    for (int i = 0; i < nNodes; i++)
        head[i] = nullptr;
    // construct directed graph by adding edges to it
    for (int i = 0; i < nEdges; i++)  {
        int start_ver = rows_ref[i];
        int end_ver = cols_ref[i];
        int weight = 1;
        // insert in the beginning
        adjNode* newNode = getAdjListNode(end_ver, weight, head[start_ver]);
           
        // point head pointer to new node
        head[start_ver] = newNode;

        start_ver = cols_ref[i];
        end_ver = rows_ref[i];
        weight = 1;
        // insert in the beginning
        newNode = getAdjListNode(end_ver, weight, head[start_ver]);
           
        // point head pointer to new node
        head[start_ver] = newNode;
    
    }
    /* Generate connected components by DFS of adjacency list.  
        Stored into vectorOfConnectedComponents which is passed by reference */
        
    
    /* Sort Atom Indices in N connected components, then sort N connected components by first atom index*/
    /*for (std::vector< std::vector<int> >::iterator it = vectorOfConnectedComponents.begin();
        it != vectorOfConnectedComponents.end(); it++){
        std::sort(it->begin(), it->end());
    }
    std::sort(vectorOfConnectedComponents.begin(), vectorOfConnectedComponents.end());
    */
    connectedComponents(vectorOfConnectedComponents);

#if debugAdjList
#ifndef NDEBUG
    std::cout << "Adjacency List" << std::endl;
    for (int i = 0; i < nNodes; i++)
        display_AdjList(head[i], i);

    std::cout << "Connected Components" << std::endl;
    for (std::vector< std::vector<int> >::iterator it = vectorOfConnectedComponents.begin();
        it != vectorOfConnectedComponents.end(); it++){
        for (std::vector<int>::iterator it2 = it->begin();
            it2 != it->end(); it2++){
            std::cout << *it2 << ", ";
        }
        std::cout << std::endl;
    }
#endif    
#endif
}

// Destructor
ConnectednessTest::~ConnectednessTest() {
    for (int i = 0; i < nNodes; i++){
      while (head[i] != nullptr) {
        adjNode *curr = head[i];
        head[i] = head[i]->next;
        delete curr;
      }
    }
    delete[] head;
}

// print all adjacent vertices of given vertex
void ConnectednessTest::display_AdjList(adjNode* ptr, int i)
{
    while (ptr != nullptr) {
        std::cout << "(" << i << ", " << ptr->val
            << ", " << ptr->cost << ") ";
        ptr = ptr->next;
    }
    std::cout << std::endl;
}

// Method to print connected components in an 
// undirected graph 
void ConnectednessTest::connectedComponents(std::vector< std::vector<int> > & vectorOfConnectedComponents) 
{ 
    // Mark all the vertices as not visited 
    bool *visited = new bool[nNodes]; 
    for(int v = 0; v < nNodes; v++) 
        visited[v] = false; 
  
    for (int v=0; v < nNodes; v++) 
    { 
        if (visited[v] == false) 
        { 
            // print all reachable vertices
            // from v 
            /* For debugging
            std::cout << "Calling DFSUtil" << std::endl; */
            std::vector<int> moleculeX;
            DFSUtil(v, this->head[v], this->head, visited, moleculeX); 
            vectorOfConnectedComponents.push_back(moleculeX);
            /* For debugging std::cout << "\n"; */
        } 
    } 
    delete[] visited; 
} 
  
void ConnectednessTest::DFSUtil(int v, adjNode * node, adjNode ** head, bool * visited, std::vector<int> & moleculeX) 
{ 
    // Mark the current node as visited and print it 
    visited[v] = true; 
    /* For debugging std::cout << v << " "; */
    moleculeX.push_back(v);
    // Recur for all the vertices 
    // adjacent to this vertex
    while (node != nullptr){
        // outgoing edge : node->val
        v = node->val;
        if (visited[v]==false){
            visited[v] = true; 
            // Evaluate adjacency list of outgoing edge for prev visited
            DFSUtil(v, head[v], head, visited, moleculeX);
        }
        // Go to next node original node's adjacency list
        node = node->next;
    }
}
    
