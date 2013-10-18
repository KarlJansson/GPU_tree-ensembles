#include "stdafx.h"
#include "IEvaluation.h"

namespace DataMiner{
	void IEvaluation::setData(DataDocumentPtr data, IDataPackPtr dataPack){
		m_dataPack = dataPack; 
		m_data = data; 
		m_randomSelection = true;
		m_clInstances.assign(m_data->getNumClassValues(),std::vector<InstancePtr>());
		init();
	}

	void IEvaluation::calculateCost(int cl1,int cl2){
		m_clC.assign(2,0);
		if(cl2 < cl1){
			m_clC[0] = double(double(cl2)/double(cl1));
			m_clC[1] = 1.0;
		}
		else{
			m_clC[1] = double(double(cl1)/double(cl2));
			m_clC[0] = 1.0;
		}
		/*m_C_cl2 = 1.0;
		m_C_cl1 = 1.0;*/
	}

	double IEvaluation::calculateStandardDeviation(std::vector<double> &values){
		double average = 0;
		for(unsigned int i=0; i<values.size(); ++i){
			average += values[i];
		}
		average /= values.size();

		double standardDev = 0;
		for(unsigned int i=0; i<values.size(); ++i){
			standardDev += pow(values[i]-average,2);
		}
		standardDev /= values.size();

		return sqrt(standardDev);
	}

	std::pair<double,double> IEvaluation::calculateEnrichment(std::map<double,std::vector<bool>,std::greater<double>> &ranking, unsigned int classCount){
		unsigned int topTenPercentCount = unsigned int(float(m_testingInds.size())*0.1f);
		unsigned int count = 0, correctClass = 0;

		// Enrichment factor
		std::map<double,std::vector<bool>,std::greater<double>>::iterator enrichItr = ranking.begin();
		while(count < topTenPercentCount && enrichItr != ranking.end()){
			for(unsigned int i=0; i<enrichItr->second.size(); ++i){
				if(count >= topTenPercentCount)
					break;

				if(enrichItr->second[i])
					++correctClass;

				++count;
			}
			enrichItr++;
		}
		m_measurements.enrichmentFactors.push_back(double(double(correctClass)/double(topTenPercentCount))/(double(classCount)/double(m_testingInds.size())));
		correctClass = classCount < topTenPercentCount ? classCount : topTenPercentCount;
		m_measurements.maxEnrichments.push_back(double(double(correctClass)/double(topTenPercentCount))/(double(classCount)/double(m_testingInds.size())));

		return std::pair<double,double>(m_measurements.enrichmentFactors.back(),m_measurements.maxEnrichments.back());
	}

	double IEvaluation::calculateAUC(std::map<double,std::vector<bool>,std::greater<double>> &ranking){
		unsigned int tpAccum = 0, fpAccum = 0;
		std::vector<unsigned int> truePositive,falsePositive;
		std::map<double,std::vector<bool>,std::greater<double>>::iterator aucIter = ranking.begin();
		while(aucIter != ranking.end()){
			for(unsigned int i=0; i<aucIter->second.size(); ++i){
				if(aucIter->second[i])
					++tpAccum;
				else
					++fpAccum;

				truePositive.push_back(tpAccum);
				falsePositive.push_back(fpAccum);
			}
			aucIter++;
		}

		m_measurements.AUC = 0;
		if(truePositive[truePositive.size()-1] == 0)
			m_measurements.AUC = 0;
		else if(falsePositive[falsePositive.size()-1] == 0)
			m_measurements.AUC = 1;
		else{
			double cumNeg = 0.0,cip,cin;
			for(int i=truePositive.size()-1; i>=0; --i){
				if (i > 0) {
					cip = truePositive[i] - truePositive[i-1];
					cin = falsePositive[i] - falsePositive[i-1];
				}
				else{
					cip = truePositive[0];
					cin = falsePositive[0];
				}
				m_measurements.AUC += cip * (cumNeg + (0.5 * cin));
				cumNeg += cin;
			}
			m_measurements.AUC /= double(truePositive[truePositive.size()-1]*falsePositive[falsePositive.size()-1]);
		}

		return m_measurements.AUC;
	}

	void IEvaluation::scrambleInstances(){
		std::vector<unsigned int> newVec;

		unsigned int rngNr;
		boost::random::mt19937 rng;
		rng.seed(123);
		newVec.reserve(m_trainingInds.size());
		for(unsigned int i=0; i<m_trainingInds.size(); ++i){
			boost::random::uniform_int_distribution<> indRand(0,m_trainingInds.size()-1-i);
			rngNr = indRand(rng);
			newVec.push_back(m_trainingInds[rngNr]);
			m_trainingInds[rngNr] = m_trainingInds[m_trainingInds.size()-1-i];
		}
		m_trainingInds = newVec;

		newVec.clear();
		newVec.reserve(m_testingInds.size());
		for(unsigned int i=0; i<m_testingInds.size(); ++i){
			boost::random::uniform_int_distribution<> indRand(0,m_testingInds.size()-1-i);
			rngNr = indRand(rng);
			newVec.push_back(m_testingInds[rngNr]);
			m_testingInds[rngNr] = m_testingInds[m_testingInds.size()-1-i];
		}
		m_testingInds = newVec;
	}
}