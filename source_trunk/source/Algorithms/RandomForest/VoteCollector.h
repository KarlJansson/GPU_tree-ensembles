#pragma once

namespace DataMiner{
	class VoteCollector{
	public:
		VoteCollector(std::vector<RandomTreePtr> &trees, int instanceIdx, DataDocumentPtr data, IEvaluationPtr eval, std::vector<std::vector<bool>> &inBag, bool oobe);

		double call();
		std::vector<double> getProbs() {return classProbs;}
	private:
		void runThread();
		void callback(int workId);

		std::vector<RandomTreePtr> m_trees;
		int m_instanceIdx;
		DataDocumentPtr m_data;
		IEvaluationPtr m_eval;
		
		std::vector<std::vector<bool>> m_inBag;
		bool m_oobe;
		std::vector<double> classProbs;
	};
}