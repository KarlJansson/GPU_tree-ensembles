#include "stdafx.h"
#include "RandomForest.h"
#include "MinerGUI.h"
#include "VoteCollector.h"

namespace DataMiner{
	RandomForest::RandomForest(){
		m_chooseVote = MutexPtr(new boost::mutex);
	}

	void RandomForest::run(){
		std::string setting = m_data->m_gui->getEditText(IDC_RANDFOREST_NUMFEATURES);
		
		try{
			m_numFeatures = boost::lexical_cast<int,std::string>(setting);
		}
		catch(...){
			m_numFeatures = (int)log((float)m_document->getNumAttributes())+1;
		}

		setting = m_data->m_gui->getEditText(IDC_RANDFOREST_NUMTREES);
		try{
			m_numTrees = boost::lexical_cast<int,std::string>(setting);
		}
		catch(...){
			m_numTrees = 10;
		}

		setting = m_data->m_gui->getEditText(IDC_RANDFOREST_TREEDEPTH);
		try{
			m_MaxDepth = boost::lexical_cast<int,std::string>(setting);
		}
		catch(...){
			m_MaxDepth = 100;
		}
		m_bagger = BaggerPtr(new Bagger());

		// Set up the tree options which are held in the motherForest.
		m_KValue = m_numFeatures;

		// set up the bagger and build the forest
		m_bagger->setNumIterations(m_numTrees);
		m_bagger->setCalcOutOfBag(true);
		m_bagger->setComputeImportances(true); //change

		int timerId = ConfigManager::startTimer();
		m_bagger->build(m_document, m_evaluation, m_data->m_gui, m_NumThreads, boost::static_pointer_cast<RandomForest>(shared_from_this()));
		m_buildTimer = ConfigManager::getTime(timerId);
		ConfigManager::removeTimer(timerId);
		evaluateTestSet();

		m_bagger.reset();
		m_voteCollectors.clear();
		m_votes.clear();
	}

	void RandomForest::evaluateTestSet(){
		m_numVotesLeft = m_evaluation->getNumTestingInstances();
		m_votes = std::vector<double>(m_numVotesLeft,0);
		m_voteCollectors = std::vector<VoteCollectorPtr>(m_numVotesLeft,VoteCollectorPtr());
		m_voteId = 0;

		int timerId = ConfigManager::startTimer();
		m_barrier = BarrierPtr(new boost::barrier(2));

		TM_runFunctionPtr runFunc = TM_runFunctionPtr(new boost::function<void(void)>(std::bind(std::mem_fun(&RandomForest::vote),this)));
		TM_callbackFunctionPtr callbackFunc = TM_callbackFunctionPtr(new boost::function<void(int)>(std::bind1st(std::mem_fun(&RandomForest::voteCallback),this)));
		for(int i = 0; i < m_evaluation->getNumTestingInstances(); i++){
			m_voteCollectors[i] = VoteCollectorPtr(new VoteCollector(m_bagger->getTrees(), i, m_document, m_evaluation, std::vector<std::vector<bool>>(), false));
			ThreadManager::launchWorkPackage(runFunc,callbackFunc);
		}

		m_barrier->wait();
		double timerBuild = ConfigManager::getTime(timerId);
		ConfigManager::removeTimer(timerId);

		unsigned int correct = 0,wrong = 0;

		std::map<double,std::vector<bool>,std::greater<double>> rankingcl1,rankingcl2;
		bool truePositive;
		std::vector<int> clCorrect,clWrong;
		std::vector<double> probs;
		for(size_t i = 0; i < m_evaluation->getNumTestingInstances(); i++) {
			double vote = m_votes[i];
			probs = m_voteCollectors[i]->getProbs();

			if(vote == m_evaluation->getTestingInstance(i)->classValue()){
				++clCorrect[m_evaluation->getTestingInstance(i)->classValue()];
				truePositive;
				++correct;
			}
			else{
				++clWrong[m_evaluation->getTestingInstance(i)->classValue()];
				++wrong;
			}

			rankingcl1[double(probs[0])/double(probs[0]+probs[1])].push_back((0 == m_evaluation->getTestingInstance(i)->classValue()));
			rankingcl2[double(probs[1])/double(probs[0]+probs[1])].push_back((1 == m_evaluation->getTestingInstance(i)->classValue()));
		}

		double auc = (m_evaluation->calculateAUC(rankingcl1)+m_evaluation->calculateAUC(rankingcl2))/2.0;

		m_outputStream <<
			"	Time used for building trees: " << m_buildTimer << "s\r\n" <<
			"	Time used for evaluation of trees: " << timerBuild << "s\r\n";

		m_outputStream << 
			"\r\n	OOBError: " << m_bagger->getOOBError() << "\r\n";

		m_outputStream	<< "\r\n	";
		
		for(unsigned int i=0; i<m_document->getNumClassValues(); ++i){
			m_outputStream << i << "	";
		}
		for(unsigned int i=0; i<m_document->getNumClassValues(); ++i){
			m_outputStream << "\r\n	" << clCorrect[i] << "	" << clWrong[m_document->getNumClassValues()-1-i] << "	" << i; 
		}
		m_outputStream << "\r\n";

		double accuracy = double(double(correct)/double(correct+wrong))*100;
		m_outputStream << 
			"\r\n	" << "Accuracy: " << accuracy << "%\r\n";

		PROCESS_MEMORY_COUNTERS pmc;
		GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
		size_t memUsageCPU = pmc.WorkingSetSize;

		m_resultWriter->addKeyValue("accuracy",accuracy);
		m_resultWriter->addKeyValue("cl1Correct",(clCorrect[0]));
		m_resultWriter->addKeyValue("cl1Wrong",(clWrong[0]));
		m_resultWriter->addKeyValue("cl2Correct",(clCorrect[1]));
		m_resultWriter->addKeyValue("cl2Wrong",(clWrong[1]));
		m_resultWriter->addKeyValue("trainingTime",(m_buildTimer));
		m_resultWriter->addKeyValue("testingTime",(timerBuild));
		m_resultWriter->addKeyValue("totalTime",(timerBuild+m_buildTimer));
		m_resultWriter->addKeyValue("testingInstances",(m_evaluation->getNumTestingInstances()));
		m_resultWriter->addKeyValue("trainingInstances",(m_evaluation->getNumTrainingInstances()));
		m_resultWriter->addKeyValue("auc",(auc));
		m_resultWriter->addKeyValue("memUsageCPU",(double(memUsageCPU)/1024.0/1024.0));

		m_voteCollectors.clear();
		m_votes.clear();
	}

	void RandomForest::vote(){
		int voteId = 0;
		m_chooseVote->lock();
		voteId = m_voteId++;
		m_chooseVote->unlock();

		m_votes[voteId] = m_voteCollectors[voteId]->call();
	}
	
	void RandomForest::voteCallback(int packId){
		m_chooseVote->lock();
		m_numVotesLeft--;

		m_data->m_gui->setProgressBar(IDC_PROGRESSBAR_PROGRESS2,m_votes.size(),m_votes.size()-m_numVotesLeft);

		if(m_numVotesLeft <= 0)
			m_barrier->wait();
		m_chooseVote->unlock();
	}

	std::vector<double> RandomForest::distributionForInstance(InstancePtr instance){
		return m_bagger->distributionForInstance(instance);
	}

	std::vector<double> RandomForest::getFeatureImportances(){
		return m_bagger->getFeatureImportances();
	}
}