#include "stdafx.h"
#include "ExRandomTrees.h"

namespace DataMiner{
	ExRandomTrees::ExRandomTrees(){
	}

	void ExRandomTrees::run(){
		std::string setting = m_data->m_gui->getEditText(IDC_RANDFOREST_NUMFEATURES);
		
		try{
			m_numFeatures = boost::lexical_cast<int,std::string>(setting);
		}
		catch(...){
			m_numFeatures = (int)log((float)m_document->getNumAttributes())+1;
		}

		if(m_numFeatures == 0)
			m_numFeatures = (int)log((float)m_document->getNumAttributes())+1;

		setting = m_data->m_gui->getEditText(IDC_RANDFOREST_NUMTREES);
		try{
			m_numTrees = boost::lexical_cast<int,std::string>(setting);
		}
		catch(...){
			m_numTrees = 10;
		}

		setting = m_data->m_gui->getEditText(IDC_RANDFOREST_TREEDEPTH);
		try{
			m_maxDepth = boost::lexical_cast<int,std::string>(setting);
		}
		catch(...){
			m_maxDepth = 100;
		}

		setting = m_data->m_gui->getEditText(IDC_RANDFOREST_SEED);
		try{
			m_randomSeed = boost::lexical_cast<int,std::string>(setting);
		}
		catch(...){
			m_randomSeed = 100;
		}

		setting = m_data->m_gui->getEditText(IDC_RANDFOREST_MAXINST);
		try{
			m_minNumInst = boost::lexical_cast<int,std::string>(setting);
		}
		catch(...){
			m_minNumInst = 10;
		}

		m_dataSetTrain.assign(m_document->getNumAttributes(),std::vector<float>(m_evaluation->getNumTrainingInstances(),0));
		m_classSetTrain.assign(m_document->getNumInstances(),0);
		m_dataSetTest.assign(m_document->getNumAttributes(),std::vector<float>(m_evaluation->getNumTestingInstances(),0));
		m_classSetTest.assign(m_evaluation->getNumTestingInstances(),0);
		for(unsigned int a=0; a<m_document->getNumAttributes(); ++a){
			for(unsigned int i=0; i<m_evaluation->getNumTrainingInstances(); ++i){
				m_dataSetTrain[a][i] = m_evaluation->getTrainingInstance(i)->getValue(a);

				if(a == 0)
					m_classSetTrain[i] = m_evaluation->getTrainingInstance(i)->classValue();
			}

			for(unsigned int i=0; i<m_evaluation->getNumTestingInstances(); ++i){
				m_dataSetTest[a][i] = m_evaluation->getTestingInstance(i)->getValue(a);

				if(a == 0)
					m_classSetTest[i] = m_evaluation->getTestingInstance(i)->classValue();
			}
		}

		// Create trees
		ExTree tree = ExTree(m_evaluation->getNumTrainingInstances());
		m_trees.assign(m_numTrees,tree);
		
		// Seed treees
		boost::random::uniform_int_distribution<> indRand(0,INT_MAX);
		boost::random::mt19937 rng;
		rng.seed(m_randomSeed);
		for(unsigned int i=0; i<m_trees.size(); ++i){
			m_trees[i].seed(indRand(rng));
		}

		m_choiceId = 0;
		m_workLeft = m_numTrees;
		m_chooseIdLock = MutexPtr(new boost::mutex);
		m_barrier = BarrierPtr(new boost::barrier(2));

		TM_runFunctionPtr runFunc = TM_runFunctionPtr(new boost::function<void(void)>(std::bind(std::mem_fun(&ExRandomTrees::buildTree),this)));
		TM_callbackFunctionPtr callbackFunc = TM_callbackFunctionPtr(new boost::function<void(int)>(std::bind1st(std::mem_fun(&ExRandomTrees::buildTreeCallback),this)));
		for(size_t treeIdx = 0; treeIdx < m_trees.size(); ++treeIdx) {
			ThreadManager::queueWorkPackage(runFunc,callbackFunc);
		}

		// Start tree building threads
		TimerPtr timer = TimerPtr(new boost::timer);
		ThreadManager::executeWorkQueue();
		// Make sure all trees have been trained before proceeding
		m_barrier->wait();
		m_buildTimer = timer->elapsed();

		evaluateTestSet();
	}

	void ExRandomTrees::buildTree(){
		unsigned int treeId;
		m_chooseIdLock->lock();
			treeId = m_choiceId++;
		m_chooseIdLock->unlock();

		ExTree& tree = m_trees[treeId];
		tree.m_nodes.reserve(m_evaluation->getNumTrainingInstances()*2);
		tree.m_nodes.push_back(ExTreeNode(0,m_evaluation->getNumTrainingInstances()));
		for(unsigned int i=0; i<m_evaluation->getNumTrainingInstances(); ++i){
			++tree.m_nodes[0].classProbs[m_classSetTrain[i]];
		}

		unsigned int depth = 0;
		unsigned int unprocessedNodes = 1, newNodes = 0;
		unsigned int numInstances;
		boost::random::uniform_int_distribution<> attRand(0,m_document->getNumAttributes()-1);
		double split;
		std::vector<std::vector<unsigned int>> dist,bestDist;
		int k;
		bool sensibleSplit = false, priorDone;
		double prior, posterior;
		double bestVal = -1000;
		unsigned int attribute;
		double splitPoint;

		while(unprocessedNodes != 0){
			// Iterate through unprocessed nodes
			while(unprocessedNodes != 0){
				ExTreeNode& currentNode = tree.m_nodes[tree.m_nodes.size()-(unprocessedNodes+newNodes)];

				// Check if node is a leaf
				numInstances = currentNode.instIndEnd - currentNode.instIndStart;
				if((numInstances > 0  &&  numInstances < max(2, m_minNumInst))  // small
					|| (abs((currentNode.classProbs[0] > currentNode.classProbs[1] ? currentNode.classProbs[0] : currentNode.classProbs[1]) - (currentNode.classProbs[0]+currentNode.classProbs[1])) < 1e-6)      // pure
					|| ((m_maxDepth > 0)  &&  (depth >= m_maxDepth))                           // deep
					){
				}
				else{
					boost::random::uniform_int_distribution<> instRand(0,numInstances-1);
					priorDone = false;
					k = m_numFeatures;
					sensibleSplit = false;
					bestVal = -1000;
					unsigned int numIter = 0;
					while(numIter < m_document->getNumAttributes() && (k-- > 0 || !sensibleSplit)){
						++numIter;
						attribute = attRand(tree.m_rng);

						split = 0;
						for(unsigned int i=0; i<10; ++i){
							split += m_evaluation->getTrainingInstance(tree.m_instIndices[tree.m_bufferId][currentNode.instIndStart+instRand(tree.m_rng)])->getValue(attribute);
						}
						splitPoint = split/10;

						// Calculate distribution
						dist = std::vector<std::vector<unsigned int>>(2,std::vector<unsigned int>(2,0));
						unsigned int instId;
						for(unsigned int i=0; i<numInstances; ++i){
							instId = tree.m_instIndices[tree.m_bufferId][currentNode.instIndStart+i];
							++dist[m_dataSetTrain[attribute][instId] < splitPoint ? 0 : 1][m_classSetTrain[instId]];
						}

						if(!priorDone){ // needs to be computed only once per branch
							// Entropy over collumns
							prior = 0;
							double sumForColumn, total = 0;
							for (size_t j = 0; j < dist[0].size(); j++) {
								sumForColumn = 0;
								for (size_t i = 0; i < dist.size(); i++) {
									sumForColumn += dist[i][j];
								}
								prior -= lnFunc(sumForColumn);
								total += sumForColumn;
							}
							prior = (prior + lnFunc(total)); 

							priorDone = true;
						}
      
						// Entropy over rows
						posterior = 0;
						double sumForBranch;
						for (size_t branchNum = 0; branchNum < dist.size(); branchNum++) {
							sumForBranch = 0;
							for(size_t classNum = 0; classNum < dist[0].size(); classNum++) {
								posterior = posterior + lnFunc(dist[branchNum][classNum]);
								sumForBranch += dist[branchNum][classNum];
							}
							posterior = posterior - lnFunc(sumForBranch);
						}
						posterior = -posterior;

						if(bestVal < prior - posterior){
							bestVal = prior - posterior;
							currentNode.m_attribute = attribute;
							currentNode.m_splitPoint = splitPoint;
							bestDist = dist;
						}

						if(prior - posterior > 1e-2)   // we allow some leeway here to compensate
							sensibleSplit = true;   // for imprecision in entropy computation
					}

					if(sensibleSplit){
						// Split node
						unsigned int cpyBuffer = (tree.m_bufferId == 0 ? 1 : 0);
						unsigned int instId;
						
						tree.m_nodes.push_back(ExTreeNode(currentNode.instIndStart,currentNode.instIndStart+bestDist[0][0]+bestDist[0][1]));
						tree.m_nodes.back().classProbs[0] = bestDist[0][0];
						tree.m_nodes.back().classProbs[1] = bestDist[0][1];
						currentNode.m_children.push_back(&tree.m_nodes.back());

						tree.m_nodes.push_back(ExTreeNode(currentNode.instIndStart+bestDist[0][0]+bestDist[0][1],currentNode.instIndEnd));
						tree.m_nodes.back().classProbs[0] = bestDist[1][0];
						tree.m_nodes.back().classProbs[1] = bestDist[1][1];
						currentNode.m_children.push_back(&tree.m_nodes.back());

						unsigned int left = currentNode.instIndStart, right = currentNode.instIndEnd-1;
						for(unsigned int i=0; i<numInstances; ++i){
							instId = tree.m_instIndices[tree.m_bufferId][currentNode.instIndStart+i];
							if(m_dataSetTrain[currentNode.m_attribute][instId] < currentNode.m_splitPoint){
								tree.m_instIndices[cpyBuffer][left] = instId;
								++left;
							}
							else{
								tree.m_instIndices[cpyBuffer][right] = instId;
								--right;
							}
						}

						newNodes += 2;
					}
				}
				unprocessedNodes--;
			}

			tree.m_bufferId = (tree.m_bufferId == 0 ? 1 : 0);

			depth++;
			unprocessedNodes = newNodes;
			newNodes = 0;
		}
	}

	void ExRandomTrees::buildTreeCallback(int packId){
		m_chooseIdLock->lock();
		m_workLeft--;

		m_data->m_gui->setProgressBar(IDC_PROGRESSBAR_PROGRESS,m_trees.size(),m_trees.size()-m_workLeft);

		if(m_workLeft <= 0)
			m_barrier->wait();
		m_chooseIdLock->unlock();
	}

	void ExRandomTrees::evaluateTestSet(){
		m_votes.assign(m_evaluation->getNumTestingInstances()*2,0);
		m_workLeft = m_trees.size();
		m_choiceId = 0;

		TM_runFunctionPtr runFunc = TM_runFunctionPtr(new boost::function<void(void)>(std::bind(std::mem_fun(&ExRandomTrees::collectSingleVote),this)));
		TM_callbackFunctionPtr callbackFunc = TM_callbackFunctionPtr(new boost::function<void(int)>(std::bind1st(std::mem_fun(&ExRandomTrees::buildTreeCallback),this)));
		for(size_t treeIdx = 0; treeIdx < m_trees.size(); ++treeIdx) {
			ThreadManager::queueWorkPackage(runFunc,callbackFunc);
		}

		// Start tree building threads
		TimerPtr timer = TimerPtr(new boost::timer);
		ThreadManager::executeWorkQueue();
		m_barrier->wait();
		m_testTimer = timer->elapsed();

		unsigned int correct = 0,wrong = 0;

		std::map<double,std::vector<bool>,std::greater<double>> rankingcl1,rankingcl2;
		bool truePositive;
		std::vector<int> clCorrect(m_document->getNumClassValues(),0),clWrong(m_document->getNumClassValues(),0);
		std::vector<double> probs(2,0);
		for(size_t i = 0; i < m_evaluation->getNumTestingInstances(); i++) {
			unsigned int vote = m_votes[2*i] > m_votes[2*i+1] ? 0 : 1;
			probs[0] = m_votes[2*i];
			probs[1] = m_votes[2*i+1];

			if(vote == m_classSetTest[i]){
				++clCorrect[m_classSetTest[i]];
				truePositive;
				++correct;
			}
			else{
				++clWrong[m_classSetTest[i]];
				++wrong;
			}

			rankingcl1[double(probs[0])/double(probs[0]+probs[1])].push_back((0 == m_classSetTest[i]));
			rankingcl2[double(probs[1])/double(probs[0]+probs[1])].push_back((1 == m_classSetTest[i]));
		}

		double auc = (m_evaluation->calculateAUC(rankingcl1)+m_evaluation->calculateAUC(rankingcl2))/2.0;

		m_outputStream <<
			"	Time used for building trees: " << m_buildTimer << "s\r\n" <<
			"	Time used for evaluation of trees: " << m_testTimer << "s\r\n";

		m_outputStream	<< 
			"\r\n	"; 

		std::vector<std::vector<unsigned int>> confusMatrix(m_document->getNumClassValues(),std::vector<unsigned int>(m_document->getNumClassValues(),0));
		for(unsigned int i=0; i<m_document->getNumClassValues(); ++i){
			for(unsigned int j=0; j<m_document->getNumClassValues(); ++j){
				if(i == j)
					confusMatrix[i][j] = clCorrect[i];
				else
					confusMatrix[i][j] = clWrong[i];
			}
		}

		for(unsigned int i=0; i<m_document->getNumClassValues(); ++i){
			m_outputStream << i << "	";
		}
		for(unsigned int i=0; i<m_document->getNumClassValues(); ++i){
			m_outputStream << "\r\n	";
			for(unsigned int j=0; j<m_document->getNumClassValues(); ++j){
				m_outputStream  << confusMatrix[i][j] << "	";
			}
			m_outputStream << i;
		}
		m_outputStream << "\r\n";

		double accuracy = double(double(correct)/double(wrong+correct))*100;

		m_outputStream << 
			"\r\n	" << "Accuracy: " << accuracy << "%\r\n";

		PROCESS_MEMORY_COUNTERS pmc;
		GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
		size_t memUsageCPU = pmc.WorkingSetSize;

		m_resultWriter->addKeyValue("accuracy",accuracy);
		m_resultWriter->addKeyValue("cl1Correct",clCorrect[0]);
		m_resultWriter->addKeyValue("cl1Wrong",clWrong[0]);
		m_resultWriter->addKeyValue("cl2Correct",clCorrect[1]);
		m_resultWriter->addKeyValue("cl2Wrong",clWrong[1]);
		m_resultWriter->addKeyValue("trainingTime",m_buildTimer);
		m_resultWriter->addKeyValue("testingTime",m_testTimer);
		m_resultWriter->addKeyValue("totalTime",m_testTimer+m_buildTimer);
		m_resultWriter->addKeyValue("testingInstances",m_evaluation->getNumTestingInstances());
		m_resultWriter->addKeyValue("trainingInstances",m_evaluation->getNumTrainingInstances());
		m_resultWriter->addKeyValue("auc",auc);
		m_resultWriter->addKeyValue("memUsageCPU",double(memUsageCPU)/1024.0/1024.0);
	}

	void ExRandomTrees::collectSingleVote(){
		unsigned int treeId;
		m_chooseIdLock->lock();
			treeId = m_choiceId++;
		m_chooseIdLock->unlock();

		ExTreeNode *node;
		for(unsigned int i=0; i<m_evaluation->getNumTestingInstances(); ++i){
			node = &m_trees[treeId].m_nodes[0];
			while(node && !node->m_children.empty()){
				if(m_dataSetTest[node->m_attribute][i] < node->m_splitPoint)
					node = node->m_children[0];
				else
					node = node->m_children[1];
			}
			
			m_chooseIdLock->lock();
			int classPred = (node->classProbs[0] > node->classProbs[1] ? 0 : 1);
			++m_votes[2*i + classPred];
			m_chooseIdLock->unlock();
		}
	}

	inline float ExRandomTrees::fast_log2(float val){
		int * const    exp_ptr = reinterpret_cast<int*>(&val);
		int            x = *exp_ptr;
		const int      log_2 = ((x >> 23) & 255) - 128;
		x &= ~(255 << 23);
		x += 127 << 23;
		*exp_ptr = x;

		val = ((-1.0f/3) * val + 2) * val - 2.0f/3;   // (1)

		return (val + log_2);
	}

	double ExRandomTrees::lnFunc(double num){
		if(num <= 1e-6){
			return 0;
		} 
		else{
			return num * fast_log2(float(num));
			//return num * log(num);
		}
	}
}