/*******************************************************************************
GPU OPTIMIZED MONTE CARLO (GOMC) 2.70
Copyright (C) 2018  GOMC Group
A copy of the GNU General Public License can be found in the COPYRIGHT.txt
along with this program, also can be found at <http://www.gnu.org/licenses/>.
********************************************************************************/

/* Courtesy of https://www.softwaretestinghelp.com/graph-implementation-cpp/ */

#ifndef CONNECTEDNESSTEST_H
#define CONNECTEDNESSTEST_H

#include <vector>
#include <cstdio>
#include <iostream>
#include <limits.h>
#include <algorithm>

class Graph;

// stores adjacency list items
struct adjNode {
    int val;
    int cost;
    adjNode* next;
};
// structure to store edges
struct graphEdge {
    int start_ver, end_ver, weight;
};

class ConnectednessTest{
public:

adjNode **head;                //adjacency list as array of pointers
int nNodes;  // number of nodes in the graph
int nEdges; // number of edges in the graph


ConnectednessTest(Graph * g, std::vector< std::vector<int> > & vectorOfConnectedComponents);
~ConnectednessTest();
adjNode* getAdjListNode(int value, int weight, adjNode* head);
void display_AdjList(adjNode* ptr, int i);
void connectedComponents(std::vector< std::vector<int> > & vectorOfConnectedComponents);
void DFSUtil(int v, adjNode * node, adjNode ** head, bool * visited, std::vector<int> & moleculeX);


};
#endif
