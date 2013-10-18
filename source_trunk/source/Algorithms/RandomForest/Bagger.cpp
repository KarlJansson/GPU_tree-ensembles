#include "stdafx.h"
#include "Bagger.h"
#include "RandomTree.h"
#include "DataCache.h"
#include "Instance.h"
#include "VoteCollector.h"
#include "ThreadManager.h"
#include "SerialRandomTree.h"
#include "GUIManager.h"
#include "IEvaluation.h"

namespace DataMiner{
	Bagger::Bagger(){
		m_CalcOutOfBag = false;
		_computeImportances = true;
		m_chooseTree = MutexPtr(new boost::mutex);
		m_chooseVote = MutexPtr(new boost::mutex);
		m_BagSizePercent = 100;
	}

	void Bagger::build(DataDocumentPtr data, IEvaluationPtr eval, GUIManagerPtr gui, int threads, RandomForestPtr motherForest){
		m_data = data;
		m_gui = gui;
		m_evaluation = eval;
		m_totalNumNodes = 0;
		m_buildTime = 0;

		std::string setting = m_gui->getEditText(IDC_RANDFOREST_SEED);
		try{
			m_seed = boost::lexical_cast<unsigned int,std::string>(setting);
			m_seed += 362436069;
		}
		catch(...){
			m_seed = 362436069;
		}

		boost::random::uniform_int_distribution<> indRand(0,INT_MAX);
		boost::random::mt19937 rng;
		rng.seed(m_seed);
		m_trees = std::vector<RandomTreePtr>(m_NumIterations,RandomTreePtr());
		for (size_t i = 0; i < m_trees.size(); i++) {
			m_trees[i] = RandomTreePtr(new SerialRandomTree(indRand(rng)));
			m_trees[i]->m_MotherForest = motherForest;
		}

		if (m_CalcOutOfBag && (m_BagSizePercent != 100)) {
		  //throw new IllegalArgumentException("Bag size needs to be 100% if " +
			//"out-of-bag error is to be calculated!");
		}

		m_bagSize = m_evaluation->getNumTrainingInstances() * m_BagSizePercent / 100;
		m_inBag = std::vector<std::vector<bool>>(m_trees.size(),std::vector<bool>());

		// thread management
		m_barrier = BarrierPtr(new boost::barrier(2));
		
		m_treeId = 0;
		m_numTreesLeft = m_trees.size();

		TM_runFunctionPtr runFunc = TM_runFunctionPtr(new boost::function<void(void)>(std::bind(std::mem_fun(&Bagger::buildTree),this)));
		TM_callbackFunctionPtr callbackFunc = TM_callbackFunctionPtr(new boost::function<void(int)>(std::bind1st(std::mem_fun(&Bagger::callback),this)));
		for(size_t treeIdx = 0; treeIdx < m_trees.size(); treeIdx++) {
			ThreadManager::launchWorkPackage(runFunc,callbackFunc);
		}

		// make sure all trees have been trained before proceeding
		m_barrier->wait();

		std::wstringstream msg;
		msg << L"Total number of nodes: " << m_totalNumNodes << "\r\n";
		m_gui->postDebugMessage(msg.str());

		// calc OOB error?
		if(m_CalcOutOfBag || _computeImportances)
			m_OutOfBagError = computeOOBError(m_data, m_inBag);
		else
			m_OutOfBagError = 0;

		//calc feature importances
		if(_computeImportances) {
		//	m_FeatureImportances = std::vector<double>(data->getNumAttributes(),0);
		//	Instances dataCopy = new Instances(data); //To scramble
		//	std::vector<int> permutation = FastRfUtils.randomPermutation(data->getNumInstances(), random);
		//	for (int j = 0; j < data->getNumAttributes(); j++){
		//		if (j != data->m_classAttributeId) {
		//			double sError = computeOOBError(FastRfUtils.scramble(data, dataCopy, j, permutation), inBag);
		//			m_FeatureImportances[j] = sError - m_OutOfBagError;
		//		}
		//	}
		}

		m_gui->setProgressBar(IDC_PROGRESSBAR_PROGRESS,1,0);
		m_gui->setProgressBar(IDC_PROGRESSBAR_PROGRESS2,1,0);
	}

	double Bagger::computeOOBError(DataDocumentPtr data,std::vector<std::vector<bool>> &inBag){
		bool numeric = data->getClassFormat() == Attribute::IF_NUMERIC;

		m_numVotesLeft = m_evaluation->getNumTrainingInstances();
		m_voteId = 0;
		m_barrier = BarrierPtr(new boost::barrier(2));

		m_votes = std::vector<double>(m_numVotesLeft,0);
		m_voteCollectors = std::vector<VoteCollectorPtr>(m_numVotesLeft,VoteCollectorPtr());

		TM_runFunctionPtr runFunc = TM_runFunctionPtr(new boost::function<void(void)>(std::bind(std::mem_fun(&Bagger::vote),this)));
		TM_callbackFunctionPtr callbackFunc = TM_callbackFunctionPtr(new boost::function<void(int)>(std::bind1st(std::mem_fun(&Bagger::voteCallback),this)));
		for(int i = 0; i < m_evaluation->getNumTrainingInstances(); i++){
			m_voteCollectors[i] = VoteCollectorPtr(new VoteCollector(m_trees, i, data, m_evaluation, inBag, true));
			ThreadManager::launchWorkPackage(runFunc,callbackFunc);
		}

		m_barrier->wait();
		m_voteCollectors.clear();

		double outOfBagCount = 0.0;
		double errorSum = 0.0;

		for (size_t i = 0; i < m_evaluation->getNumTrainingInstances(); i++) {
			double vote = m_votes[i];

			// error for instance
			outOfBagCount += data->getInstance(i)->weight();
			if(numeric){
				errorSum += abs(vote - m_evaluation->getTrainingInstance(i)->classValue()) * m_evaluation->getTrainingInstance(i)->weight();
			} 
			else{
				if (vote != m_evaluation->getTrainingInstance(i)->classValue())
					errorSum += m_evaluation->getTrainingInstance(i)->weight();
			}
		}

		return errorSum / outOfBagCount;
	}

	std::vector<double> Bagger::getFeatureImportances(){
		return m_FeatureImportances;
	}

	std::vector<double> Bagger::distributionForInstance(InstancePtr instance){
		std::vector<double> sums(m_data->getNumClassValues(),0), newProbs;

		for(int i = 0; i < m_NumIterations; i++){
			if(m_data->getClassFormat() == Attribute::IF_NUMERIC) {		
				sums[0] += m_trees[i]->classifyInstance(instance);
			} 
			else{
				newProbs = m_trees[i]->distributionForInstance(instance);
				for (int j = 0; j < newProbs.size(); j++)
					sums[j] += newProbs[j];
			}
		}

		if(m_data->getClassFormat() == Attribute::IF_NUMERIC) {
			sums[0] /= (double) m_NumIterations;
			return sums;
		}
		else if(RandomTree::sum(sums) < 1e-6) {
			return sums;
		}
		else{
			RandomTree::normalize(sums);
			return sums;
		}
	}

	void Bagger::buildTree(){
		int treeId;
		
		// Choose tree id
		m_chooseTree->lock();
		treeId = m_treeId++;
		m_chooseTree->unlock();

		// sorting is performed inside this constructor
		DataCachePtr myData = DataCachePtr(new DataCache(m_data,m_evaluation,m_seed));

		// create the in-bag dataset (and be sure to remember what's in bag)
		// for computing the out-of-bag error later
		DataCachePtr bagData = myData->resample(m_bagSize);
		/*bagData.reusableRandomGenerator = bagData.getRandomNumberGenerator(
				random.nextInt());*/
		m_inBag[treeId] = bagData->inBag;

		// build the classifier
		RandomTreePtr aTree = m_trees[treeId];
		aTree->data = bagData;
		TimerPtr timer = TimerPtr(new boost::timer);
		aTree->build();
		double time = timer->elapsed();

		m_chooseTree->lock();
		m_buildTime += time;
		m_totalNumNodes += aTree->getNumNodes();
		m_chooseTree->unlock();
	}
	
	void Bagger::callback(int packId){
		m_chooseTree->lock();
		m_numTreesLeft--;

		m_gui->setProgressBar(IDC_PROGRESSBAR_PROGRESS,m_trees.size(),m_trees.size()-m_numTreesLeft);

		if(m_numTreesLeft <= 0)
			m_barrier->wait();
		m_chooseTree->unlock();
	}

	void Bagger::vote(){
		int voteId = 0;
		m_chooseVote->lock();
		voteId = m_voteId++;
		m_chooseVote->unlock();

		m_votes[voteId] = m_voteCollectors[voteId]->call();
		m_voteCollectors[voteId].reset();
	}

	void Bagger::voteCallback(int packId){
		m_chooseVote->lock();
		m_numVotesLeft--;

		m_gui->setProgressBar(IDC_PROGRESSBAR_PROGRESS2,m_votes.size(),m_votes.size()-m_numVotesLeft);

		if(m_numVotesLeft <= 0)
			m_barrier->wait();
		m_chooseVote->unlock();
	}
}