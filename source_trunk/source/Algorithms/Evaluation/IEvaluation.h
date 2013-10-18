#pragma once
#include "DataDocument.h"
#include "IDataPack.h"

namespace DataMiner{
	class IEvaluation : public boost::enable_shared_from_this<IEvaluation>{
	public:
		struct evaluationMetrics{
			double	AUC,
					accuracy,
					precision,
					recall;
			std::vector<double> enrichmentFactors;
			std::vector<double> maxEnrichments;
		};

		virtual void run(IAlgorithmPtr algo, AlgorithmDataPackPtr data){}

		InstancePtr getTrainingInstance(unsigned int index) { return m_data->getInstance(m_trainingInds[index]); }
		InstancePtr getTestingInstance(unsigned int index) { return m_data->getInstance(m_testingInds[index]); }

		unsigned int getNumTrainingInstances() { return m_trainingInds.size(); }
		unsigned int getNumTestingInstances() { return m_testingInds.size(); }

		virtual bool advance() = 0;
		virtual void init() = 0;
		virtual bool isFinalStage() = 0;

		void setStage(unsigned int stage) { m_stage = stage; }
		unsigned int getStage() { return m_stage; }
		unsigned int getNumStages() { return m_numStages; }

		void setData(DataDocumentPtr data, IDataPackPtr dataPack);
		std::vector<unsigned int>& getTrainingInds() { return m_trainingInds; }
		std::vector<unsigned int>& getTestingInds() { return m_testingInds; }

		double getClC(unsigned int id) { return m_clC[id]; }

		void calculateCost(int cl1,int cl2);
		double calculateStandardDeviation(std::vector<double> &values);
		std::pair<double,double> calculateEnrichment(std::map<double,std::vector<bool>,std::greater<double>> &ranking, unsigned int classCount);
		double calculateAUC(std::map<double,std::vector<bool>,std::greater<double>> &ranking);
	protected:
		void scrambleInstances();

		IDataPackPtr m_dataPack;
		DataDocumentPtr m_data;
		IAlgorithmPtr m_agorithm;

		std::vector<unsigned int>	m_trainingInds,
									m_testingInds;

		std::vector<std::vector<InstancePtr>>	m_clInstances;

		evaluationMetrics m_measurements;

		unsigned int m_stage;
		unsigned int m_numStages;

		std::vector<double> m_clC;

		bool m_randomSelection;
	};
}