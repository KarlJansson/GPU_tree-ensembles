#pragma once

namespace DataMiner{
	class ExTreeNode{
	public:
		ExTreeNode(unsigned int start, unsigned int stop){
			instIndStart = start;
			instIndEnd = stop;
			classProbs.assign(2,0);
		}

		std::vector<double> classProbs;

		unsigned int instIndStart,instIndEnd;
		
		unsigned int m_attribute;
		double m_splitPoint;

		std::vector<ExTreeNode*> m_children;
	};
}