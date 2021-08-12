#include <gtest/gtest.h>
#include "Graph.h"
TEST(GraphTest, GraphTest) {
	COO cycle;
	cycle.BuildCycleCOO();
	CSR csr(cycle);
	Graph g(csr);
	std::vector<int> oddVertices;
	for(int i = 0; i < g.GetVertexCount(); ++i)
		if(i%2==1)
			oddVertices.push_back(i);

	g.InitG(g, oddVertices);
	int remainingEdges = g.GetEdgesLeftToCover();	
	std::cout << remainingEdges << std::endl;
	EXPECT_EQ(remainingEdges, -1);	
}
