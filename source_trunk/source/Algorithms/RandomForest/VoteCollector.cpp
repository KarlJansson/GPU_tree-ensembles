#include "stdafx.h"
#include "VoteCollector.h"
#include "DataDocument.h"
#include "RandomTree.h"
#include "IEvaluation.h"

namespace DataMiner{
	VoteCollector::VoteCollector(std::vector<RandomTreePtr> &trees, int instanceIdx, DataDocumentPtr data, IEvaluationPtr eval, std::vector<std::vector<bool>> &inBag, bool oobe){
		this->m_trees = trees;
		this->m_instanceIdx = instanceIdx;
		this->m_data = data;
		this->m_inBag = inBag;
		this->m_eval = eval;
		m_oobe = oobe;
	}

	double VoteCollector::call(){
		bool regression = m_data->getClassFormat() == Attribute::IF_NUMERIC;

		double regrValue = 0;

		if(!regression)
			classProbs = std::vector<double>(m_data->getNumClassValues(),0);

		int numVotes = 0;
		for(int treeIdx = 0; treeIdx < m_trees.size(); treeIdx++){
			if(m_oobe && m_inBag[treeIdx][m_instanceIdx])
				continue;

			numVotes++;

			if(regression){
				if(m_oobe)
					regrValue += m_trees[treeIdx]->classifyInstance(m_eval->getTrainingInstance(m_instanceIdx));
				else
					regrValue += m_trees[treeIdx]->classifyInstance(m_eval->getTestingInstance(m_instanceIdx));
			}
			else{
				std::vector<double> curDist;
				if(m_oobe)
					curDist = m_trees[treeIdx]->distributionForInstance(m_eval->getTrainingInstance(m_instanceIdx));
				else
					curDist = m_trees[treeIdx]->distributionForInstance(m_eval->getTestingInstance(m_instanceIdx));

				for(int classIdx = 0; classIdx < curDist.size(); classIdx++)
					classProbs[classIdx] += curDist[classIdx];
			}
		}

		double vote;
		if(regression)
			vote = regrValue / numVotes;         // average - for regression
		else
			vote = RandomTree::maxIndex(classProbs);   // consensus - for classification

		return vote;
	}
}