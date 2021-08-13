#include <gtest/gtest.h>
#include "LinearTimeDegreeSort.h"
TEST(LTDSTest, DegreeSortTester) {
	std::vector<int> reverseDegrees;
	for (int i = 99; i > -1; --i) 
		reverseDegrees.push_back(i);
	LinearTimeDegreeSort ltds(reverseDegrees.size(), reverseDegrees);
	std::vector<int> forwardDegrees(100);
	std::iota(forwardDegrees.begin(), forwardDegrees.end(), 0);
	EXPECT_EQ(ltds.GetDegreeRef(), forwardDegrees);	

}
