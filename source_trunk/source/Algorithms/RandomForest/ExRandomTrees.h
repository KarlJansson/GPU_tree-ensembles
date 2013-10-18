#pragma once
#include "IAlgorithm.h"
#include "ExTree.h"

namespace DataMiner{
	class ExRandomTrees : public IAlgorithm{
	public:
		ExRandomTrees();

	private:
		void run();
		void buildTree();
		void buildTreeCallback(int packId);

		void evaluateTestSet();

		void collectSingleVote();

		float fast_log2(float val);
		double lnFunc(double num);

		int m_maxDepth,
			m_numTrees,
			m_numFeatures,
			m_randomSeed,
			m_minNumInst,
			m_workLeft;

		double m_buildTimer,m_testTimer;
		std::vector<ExTree> m_trees;
		std::vector<int> m_votes;
		std::vector<std::vector<float>> m_dataSetTrain,m_dataSetTest;
		std::vector<unsigned int> m_classSetTrain,m_classSetTest;

		unsigned int m_choiceId;
		BarrierPtr m_barrier;
		MutexPtr m_chooseIdLock;
	};
}