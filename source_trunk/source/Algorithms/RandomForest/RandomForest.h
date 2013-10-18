#pragma once
#include "IAlgorithm.h"
#include "RandomTree.h"
#include "Bagger.h"

namespace DataMiner{
	class RandomForest : public IAlgorithm{
	public:
		RandomForest();

		std::vector<double> distributionForInstance(InstancePtr instance);
		std::vector<double> getFeatureImportances();
	private:
		void run();
		void evaluateTestSet();

		void vote();
		void voteCallback(int packId);

		int m_MaxDepth,
			m_KValue,
			m_numTrees,
			m_numFeatures,
			m_randomSeed,
			m_NumThreads,
			m_numVotesLeft,
			m_voteId;

		double m_buildTimer;

		std::vector<VoteCollectorPtr> m_voteCollectors;
		std::vector<double> m_votes;

		BaggerPtr m_bagger;

		BarrierPtr	m_barrier;
		MutexPtr	m_chooseVote;

		friend class RandomTree;
		friend class RecursiveRandomTree;
		friend class SerialRandomTree;
	};
}