#pragma once
#include "RandomTree.h"
#include "SerialTreeNode.h"

namespace DataMiner{
	class SerialRandomTree : public RandomTree{
	public:
		SerialRandomTree(unsigned int seed):RandomTree(seed){}
		void build();

		std::vector<double> distributionForInstance(InstancePtr instance);
		double classifyInstance(InstancePtr instance);
	private:
		void createLeafNode(SerialTreeNodePtr sourceNode);
		void createSplitNodes(SerialTreeNodePtr sourceNode, int &newNodes);
		bool findSplit(SerialTreeNodePtr sourceNode);

		std::vector<SerialTreeNodePtr> m_treeNodes;

		std::vector<int> m_attIndicesWindow;

		std::vector<double> m_vals;
		std::vector<std::vector<std::vector<double>>> m_dists;
		std::vector<std::vector<double>> m_props;
		std::vector<double> m_splits;

		unsigned int m_depth;

		friend class GPURandomForest;
	};
}