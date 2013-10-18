#pragma once
#include "DataDocument.h"

namespace DataMiner{
	class Bagger{
	public:
		Bagger();

		void build(DataDocumentPtr data, IEvaluationPtr eval, GUIManagerPtr gui, int threads, RandomForestPtr motherForest);

		void setNumIterations(int iterations) { m_NumIterations = iterations; }
		void setCalcOutOfBag(bool val) { m_CalcOutOfBag = val; }
		void setComputeImportances(bool val) { _computeImportances = val; }

		double getOOBError() { return m_OutOfBagError; }
		double getBuildTime() { return m_buildTime;}
		std::vector<RandomTreePtr>& getTrees() { return m_trees; }

		std::vector<double> getFeatureImportances();
		std::vector<double> distributionForInstance(InstancePtr instance);
	private:
		double computeOOBError(DataDocumentPtr data,std::vector<std::vector<bool>> &inBag);

		void buildTree();
		void callback(int packId);
		void vote();
		void voteCallback(int packId);

		std::vector<RandomTreePtr> m_trees;
		std::vector<double> m_FeatureImportances;
		std::vector<std::vector<bool>> m_inBag;
		std::vector<VoteCollectorPtr> m_voteCollectors;
		std::vector<double> m_votes;

		DataDocumentPtr m_data;
		GUIManagerPtr m_gui;
		IEvaluationPtr m_evaluation;
		
		int		m_NumIterations,
				m_BagSizePercent,
				m_bagSize,
				m_numTreesLeft,
				m_numVotesLeft;
		double	m_OutOfBagError,
				m_buildTime;
		bool	m_CalcOutOfBag, 
				_computeImportances;

		MutexPtr	m_chooseTree;
		MutexPtr	m_chooseVote;
		int			m_treeId,
					m_voteId;

		unsigned int m_totalNumNodes,
					 m_seed;

		BarrierPtr m_barrier;
	};
}