#include <gtest/gtest.h>
#include "Graph.h"
TEST(InitGraphTest, InitGraphTest) {
	COO cycle;
	cycle.BuildCycleCOO();
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

TEST(InduceGraphTest, InduceGraphTest) {
	COO example;
	example.BuildTheExampleCOO();
	CSR csr(example);
	Graph g(csr);
	std::vector<int> S;
	S.push_back(4);
	g.InitG(g, S);
	Graph gPrime(g);
	std::vector<int> mpt;
	gPrime.InitGPrime(g, mpt);
	std::vector<int> newRowOff = gPrime.GetCSR().GetNewRowOffRef();
	std::vector<int> newCols = gPrime.GetCSR().GetNewColRef();
	std::vector<int> newVals = gPrime.GetCSR().GetNewValRef();
	
	static const int rowOff[] = {0,1,4,5,7,7,10,14,14,15,16};
	static const int colInd[] = {1,0,5,6,6,5,6,1,3,8,1,2,3,9,5,6};
	static const int vals[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

	std::vector<int> testRowOff(rowOff, rowOff + sizeof(rowOff) / sizeof(rowOff[0]) );
	std::vector<int> testCols(colInd, colInd + sizeof(colInd) / sizeof(colInd[0]) );
	std::vector<int> testVals(vals, vals + sizeof(vals) / sizeof(vals[0]) );

	EXPECT_EQ(newRowOff, testRowOff);	
	EXPECT_EQ(newCols, testCols);	
	EXPECT_EQ(newVals, testVals);	
}
