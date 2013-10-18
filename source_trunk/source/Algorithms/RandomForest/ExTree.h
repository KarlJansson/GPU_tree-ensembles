#pragma once
#include "ExTreeNode.h"

namespace DataMiner{
	class ExTree{
	public:
		ExTree(unsigned int numInst){
			m_instIndices = std::vector<std::vector<int>>(2,std::vector<int>(numInst,0));
			for(unsigned int i=0; i<numInst; ++i){
				m_instIndices[0][i] = i;
			}
			m_bufferId = 0;
		}

		void seed(unsigned int seed){
			m_rng.seed(seed);
		}

		std::vector<std::vector<int>> m_instIndices;
		unsigned int m_bufferId;

		boost::random::mt19937 m_rng;
		std::vector<ExTreeNode> m_nodes;
	};
}