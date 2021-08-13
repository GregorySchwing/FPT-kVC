#include <gtest/gtest.h>
#include "Graph.h"
TEST(GraphTest, GraphTest) {
	COO cycle;
	cycle.BuildCycleCOO();
	cycle.SetVertexCountFromEdges();
	CSR csr(cycle);
	Graph g(csr);
	std::vector<int> evenVertices;
	for(int i = 0; i < g.GetVertexCount(); ++i)
		if(i % 2 == 0){
			evenVertices.push_back(i);
		}

	g.InitG(g, evenVertices);
	int remainingEdges = g.GetEdgesLeftToCover();	
	EXPECT_EQ(remainingEdges, 0);	
}
