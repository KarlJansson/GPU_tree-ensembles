#pragma once

namespace DataMiner{
	class SerialTreeNode{
	public:
		void clean(){
			sortedIndices.clear();
			classProbs.clear();
			sortedIndices.shrink_to_fit();
			classProbs.shrink_to_fit();
		}

		std::vector<double> m_Prop, m_ClassProbs;

		std::vector<std::vector<int>> sortedIndices;
		std::vector<double> classProbs;

		int m_Attribute;
		double m_SplitPoint;

		std::vector<SerialTreeNodePtr> m_children;
	};
}