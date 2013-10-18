#include "stdafx.h"
#include "GPURandomForest.h"

#include "CUDA_RandomForestBagging.h"
#include "CUDA_RandomForestBuild.h"
#include "CUDA_RandomForestUpdateConstants.h"
#include "CUDA_RandomForestSort.h"
#include "CUDA_RandomForestFindSplit.h"
#include "CUDA_RandomForestEvaluateSplit.h"
#include "CUDA_RandomForestSplitData.h"
#include "CUDA_RandomForestClassify.h"
#include "CUDA_RandomForestKeplerBuild.h"
#include "CUDA_RandomForestExCreateNodes.h"
#include "CUDA_RandomForestExMakeSplit.h"
#include "CUDA_RandomForestExFindSplit.h"

#include "SerialRandomTree.h"
#include "GPUERT.h"
#include "GPURF.h"

namespace DataMiner{
	GPURandomForest::GPURandomForest(GraphicsManagerPtr gfxMgr):m_gfxMgr(gfxMgr){
		m_saveModel = false;
		m_kernelSpecificTimings = false;
	}
	
	GPURandomForest::~GPURandomForest(){
		cleanResources();
	}
	
	void GPURandomForest::run(){
		m_numTrees = m_data->m_gui->getSetting<unsigned int>(IDC_RANDFOREST_NUMTREES);
		if(m_numTrees == 0)
			m_numTrees = 1;

		m_numFeatures = m_data->m_gui->getSetting<unsigned int>(IDC_RANDFOREST_NUMFEATURES);
		if(m_numFeatures == 0)
			m_numFeatures = (int)log((float)m_document->getNumAttributes())+1;

		m_MaxDepth = m_data->m_gui->getSetting<unsigned int>(IDC_RANDFOREST_TREEDEPTH);
		if(m_MaxDepth == 0)
			m_MaxDepth = 1;

		m_maxInstInNodes = m_data->m_gui->getSetting<int>(IDC_RANDFOREST_MAXINST);
		if(m_maxInstInNodes < 0)
			m_maxInstInNodes = 1;

		m_seed = m_data->m_gui->getSetting<unsigned int>(IDC_RANDFOREST_SEED);
		m_seed += 362436;

		std::string iterationSelect = m_data->m_gui->getEditText(IDC_RANDFOREST_ITSELECTOR);
		if(iterationSelect.compare("Iteration_1") == 0){
			m_version = 0;
		}
		else if(iterationSelect.compare("Iteration_2") == 0){
			m_version = 1;
		}
		else if(iterationSelect.compare("Iteration_3") == 0){
			m_version = 2;
		}
		else if(iterationSelect.compare("Iteration_4") == 0){
			m_version = 3;
		}
		m_kernelTimes.clear();

		// Get testing instance classes
		for(unsigned int i=0; i<m_evaluation->getNumTestingInstances(); ++i){
			int classValue = m_evaluation->getTestingInstance(i)->classValue();
			m_testSetClassVec.push_back(classValue);
		}

		BarrierPtr barrier = BarrierPtr(new boost::barrier(m_gfxMgr->getNumDevices()+1));
		unsigned int treesPerGPU = m_numTrees/m_gfxMgr->getNumDevices();
		unsigned int restTrees = m_numTrees % m_gfxMgr->getNumDevices();

		switch(m_version){
			case 0:
				break;
			case 1:{
				for(unsigned int i=0; i<m_gfxMgr->getNumDevices(); ++i){
					if(i == m_gfxMgr->getNumDevices()-1)
						treesPerGPU += restTrees;
					GPURFPtr ensemble = GPURFPtr(new GPURF(m_gfxMgr));
					ensemble->m_data = m_data;
					ensemble->m_evaluation = m_evaluation;
					ensemble->m_document = m_document;
					ensemble->m_numFeatures = m_numFeatures;
					ensemble->m_maxInstInNodes = m_maxInstInNodes;
					ensemble->m_MaxDepth = m_MaxDepth;
					ensemble->m_seed = m_seed;
					m_forests.push_back(ensemble);
					ensemble->runBuildProcess(i,treesPerGPU,barrier);
				}
				break;
			}
			case 2:
				break;
			case 3:{
				for(unsigned int i=0; i<m_gfxMgr->getNumDevices(); ++i){
					if(i == m_gfxMgr->getNumDevices()-1)
						treesPerGPU += restTrees;
					GPUERTPtr ensemble = GPUERTPtr(new GPUERT(m_gfxMgr));
					ensemble->m_data = m_data;
					ensemble->m_evaluation = m_evaluation;
					ensemble->m_document = m_document;
					ensemble->m_numFeatures = m_numFeatures;
					ensemble->m_maxInstInNodes = m_maxInstInNodes;
					ensemble->m_MaxDepth = m_MaxDepth;
					ensemble->m_seed = m_seed;
					m_forests.push_back(ensemble);
					ensemble->runBuildProcess(i,treesPerGPU,barrier);
				}
				break;
			}
		}

		barrier->wait();

		evaluateTrees();
		writeOutputStream();
		
		cleanResources();
	}

	void GPURandomForest::calculateMaxNodes(int devId){
		std::wstringstream ss;
		size_t availableMem;
		availableMem = m_gfxMgr->getAvailableMemory(devId);
		unsigned int avaliableMemMb = availableMem/(1024*1024);
		ss << "Available memory on card: " << avaliableMemMb << "mb\r\n";

		// Data set memory consumption
		availableMem -= m_evaluation->getNumTrainingInstances()*m_document->getNumAttributes()*sizeof(Value::v_precision);
		availableMem -= m_evaluation->getNumTrainingInstances()*m_document->getNumAttributes()*sizeof(unsigned int);
		availableMem -= m_evaluation->getNumTestingInstances()*m_document->getNumAttributes()*sizeof(Value::v_precision);
		availableMem -= m_evaluation->getNumTrainingInstances()*sizeof(unsigned int);
		availableMem -= m_evaluation->getNumTestingInstances()*sizeof(unsigned int);
		
		// Save 10% of memory on th GPU
		availableMem -= availableMem*0.10;

		long memPerNode = 0;
		// Random states
		memPerNode += 4*sizeof(unsigned int);
		// Node indices limits
		memPerNode += 2*sizeof(unsigned int);
		// Split points
		memPerNode += sizeof(Value::v_precision);
		// Attribute id
		memPerNode += sizeof(int);
		// Class probabilities
		memPerNode += 2*sizeof(Value::v_precision);
		// Child ids
		memPerNode += sizeof(unsigned int)*2;
		// Tree ids
		memPerNode += sizeof(unsigned int);
		// Dist Buffer
		memPerNode += AttributeMaxNominal*2*sizeof(Value::v_precision);
		// Split value
		memPerNode += sizeof(Value::v_precision);

		long memPerTree = 0;
		// Node indices
		memPerTree += m_evaluation->getNumTrainingInstances()*sizeof(unsigned int)*2;
		// Bags
		memPerTree += m_evaluation->getNumTrainingInstances()*sizeof(Value::v_precision);
		
		// Nodes
		long nodesPerTree = m_evaluation->getNumTrainingInstances()*1.2;
		memPerTree += memPerNode*nodesPerTree;

		// Calculate amount of trees that are reasonable to have on the GPU at a time
		m_maxTreesPerIteration = availableMem/memPerTree;
		if(m_maxTreesPerIteration > m_numTrees)
			m_maxTreesPerIteration = m_numTrees;

		m_maxNodesPerIteration = nodesPerTree*m_maxTreesPerIteration;

		// Temporary cap!
		//if(m_maxTreesPerIteration > 1000)
		//	m_maxTreesPerIteration = 1000;

		ss << "Max possible nodes: " << m_maxNodesPerIteration << "\r\n";
		m_data->m_gui->postDebugMessage(ss.str());
	}

	void GPURandomForest::runBaggingProcess(){
		TimerPtr timer = TimerPtr(new boost::timer);

		m_gfxMgr->setGPUBuffer(m_devId,m_setBufferIdsBagging,m_setResourceTypesBagging);
		m_gfxMgr->setGPUProgram(m_devId,m_gpuFunctionIds["RFP_Bagging"]);

		m_constantsBagging.cb_instanceCount = m_evaluation->getNumTrainingInstances();
		m_constantsBagging.cb_treeOffset = 0;
		m_constantsBagging.cb_nodeBufferEnd = m_maxNodesPerIteration-1;

		int treesLeft = m_maxTreesPerIteration;
		unsigned int maxBlocks = 1024;
		timer->restart();

		while(treesLeft > 0){
			if(int(treesLeft-int(maxBlocks)) < 0)
				maxBlocks = treesLeft;

			m_gfxMgr->copyToGPU(m_devId,m_bufferIds["RFB_constants"],&ConstantUpdate(&m_constantsBagging,KID_Bagging),sizeof(m_constantsBagging));
			m_gfxMgr->launchComputation(m_devId,maxBlocks*thread_group_size,1,1);
			m_constantsBagging.cb_treeOffset += maxBlocks;
			treesLeft -= maxBlocks;
		}
		m_baggingTime += timer->elapsed();
	}

	void GPURandomForest::runClassificationProcess(unsigned int trees){
		TimerPtr timer = TimerPtr(new boost::timer);
		int totalThreads = trees*m_evaluation->getNumTestingInstances();
		int maxThreadsPerLaunch = 1024*thread_group_size;

		std::map<unsigned int,unsigned int,std::greater<unsigned int>> classDist;
		for(unsigned int i=0; i<m_evaluation->getNumTrainingInstances(); ++i){
			classDist[m_evaluation->getTrainingInstance(i)->classValue()]++;
		}

		m_constantsClassify.cb_numTrees = trees;
		m_constantsClassify.cb_treeOffset = 0;
		m_constantsClassify.cb_majorityClass = classDist.begin()->first;
		timer->restart();
		m_gfxMgr->setGPUProgram(m_devId,m_gpuFunctionIds["RFP_Classify"]);
		m_gfxMgr->setGPUBuffer(m_devId,m_setBufferIdsClassify,m_setResourceTypesClassify);

		while(totalThreads > 0){
			m_gfxMgr->copyToGPU(m_devId,m_bufferIds["RFB_constants"],&ConstantUpdate(&m_constantsClassify,KID_Classify),sizeof(m_constantsClassify));
			m_gfxMgr->launchComputation(m_devId,(totalThreads > maxThreadsPerLaunch ? maxThreadsPerLaunch : totalThreads),1,1);

			totalThreads -= maxThreadsPerLaunch;
			m_constantsClassify.cb_treeOffset += maxThreadsPerLaunch;
		}

		if(m_kernelSpecificTimings){
			m_gfxMgr->syncDevice(m_devId);
			m_kernelTimes[L"RFP_Classify"] += timer->elapsed();
		}
	}

	void GPURandomForest::getVotesFromGPU(){
		m_gfxMgr->copyFromGPU(m_devId,m_bufferIds["RFB_votes"],&m_testSetVotes[0],m_testSetVotes.size()*sizeof(unsigned int));
	}

	void GPURandomForest::getResultsFromGPU(){
		m_baseIndVec.push_back(m_attributeVec.size());

		std::vector<Value::v_precision> splitPointsVec(m_maxNodesPerIteration,0);
		m_gfxMgr->copyFromGPU(m_devId,m_bufferIds["RFB_splitPoints"],&splitPointsVec[0],splitPointsVec.size()*sizeof(Value::v_precision));

		std::vector<int> attributeVec(m_maxNodesPerIteration,0);
		m_gfxMgr->copyFromGPU(m_devId,m_bufferIds["RFB_attributeBuffer"],&attributeVec[0],attributeVec.size()*sizeof(int));

		std::vector<unsigned int> classProbVec(2*m_maxNodesPerIteration,0);
		m_gfxMgr->copyFromGPU(m_devId,m_bufferIds["RFB_classProbs"],&classProbVec[0],classProbVec.size()*sizeof(unsigned int));

		std::vector<unsigned int> childVec(2*m_maxNodesPerIteration,0);
		m_gfxMgr->copyFromGPU(m_devId,m_bufferIds["RFB_childIds"],&childVec[0],childVec.size()*sizeof(unsigned int));

		// Append data to buffers
		for(unsigned int i=0; i<m_maxNodesPerIteration; ++i){
			m_classProbVec.push_back(classProbVec[2*i]);
			m_classProbVec.push_back(classProbVec[2*i+1]);
			m_attributeVec.push_back(attributeVec[i]);
			m_splitPointsVec.push_back(splitPointsVec[i]);
			m_childVec.push_back(childVec[2*i]);
			m_childVec.push_back(childVec[2*i+1]);
		}
	}

	void GPURandomForest::evaluateTrees(){
		m_testSetVotes.assign(m_testSetClassVec.size()*2,0);
		for(unsigned int i=0; i<m_testSetVotes.size(); ++i){
			for(unsigned int d=0; d<m_gfxMgr->getNumDevices(); ++d){
				m_testSetVotes[i] += m_forests[d]->m_testSetVotes[i];
			}
		}

		m_clCorrect[0] = m_clCorrect[1] = 0;
		m_clWrong[0] = m_clWrong[1] = 0;
		m_auc = 0;
		unsigned int class1 = 0,class2 = 0;

		std::map<double,std::vector<bool>,std::greater<double>> cl1Ranking,cl2Ranking;
		std::map<double,std::vector<bool>,std::greater<double>> totalRanking;

		double probability;
		bool truePos;

		int classPrediction;
		for(unsigned int i=0; i<m_testSetClassVec.size(); ++i){
			if(m_testSetVotes[i*2] > m_testSetVotes[i*2+1]){
				classPrediction = 0;
				probability = double(m_testSetVotes[i*2])/double(m_testSetVotes[i*2]+m_testSetVotes[i*2+1]);
			}
			else{
				classPrediction = 1;
				probability = double(m_testSetVotes[i*2+1])/double(m_testSetVotes[i*2]+m_testSetVotes[i*2+1]);
			}
			
			if(m_testSetClassVec[i] == 0)
				++class1;
			else
				++class2;

			if(classPrediction == m_testSetClassVec[i]){
				++m_clCorrect[classPrediction];
				truePos = true;
			}
			else{
				++m_clWrong[classPrediction];
				truePos = false;
			}

			totalRanking[probability].push_back(truePos);
			cl1Ranking[double(m_testSetVotes[i*2])/double(m_testSetVotes[i*2]+m_testSetVotes[i*2+1])].push_back((0 == m_testSetClassVec[i]));
			cl2Ranking[double(m_testSetVotes[i*2+1])/double(m_testSetVotes[i*2]+m_testSetVotes[i*2+1])].push_back((1 == m_testSetClassVec[i]));

			m_data->m_gui->setProgressBar(IDC_PROGRESSBAR_PROGRESS2,m_testSetClassVec.size(),i);
		}
		m_accuracy = float(float(m_clCorrect[0]+m_clCorrect[1])/float(m_clCorrect[0]+m_clCorrect[1]+m_clWrong[0]+m_clWrong[1]))*100.0f;

		// AUC calculation
		m_auc = m_evaluation->calculateAUC(cl1Ranking);
		m_auc += m_evaluation->calculateAUC(cl2Ranking);
		m_auc /= 2.0;

		// Enrichment calculation
		std::pair<double,double> res;
		res = m_evaluation->calculateEnrichment(cl1Ranking,class1);
		m_enrichCl1 = res.first;
		m_maxEnrichCl1 = res.second;
		res = m_evaluation->calculateEnrichment(cl2Ranking,class2);
		m_enrichCl2 = res.first;
		m_maxEnrichCl2 = res.second;
	}

	void GPURandomForest::writeOutputStream(){
		PROCESS_MEMORY_COUNTERS pmc;
		GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
		size_t memUsageCPU = pmc.WorkingSetSize;
		size_t memUsageGPU = 0;
		for(unsigned int i=0; i<m_gfxMgr->getNumDevices(); ++i){
			memUsageGPU += m_gfxMgr->getAllocatedMemory(i);
		}

		m_buildTime = 0;
		m_baggingTime = 0;
		m_classificationTime = 0;
		for(unsigned int i=0; i<m_gfxMgr->getNumDevices(); ++i){
			m_buildTime = max(m_forests[i]->m_buildTime,m_buildTime);
			m_baggingTime = max(m_forests[i]->m_baggingTime,m_baggingTime);
			m_classificationTime = max(m_forests[i]->m_classificationTime,m_classificationTime);
		}

		m_resultWriter->addKeyValue("accuracy",m_accuracy);
		m_resultWriter->addKeyValue("cl1Correct",m_clCorrect[0]);
		m_resultWriter->addKeyValue("cl1Wrong",m_clWrong[0]);
		m_resultWriter->addKeyValue("cl2Correct",m_clCorrect[1]);
		m_resultWriter->addKeyValue("cl2Wrong",m_clWrong[1]);
		m_resultWriter->addKeyValue("trainingTime",m_buildTime+m_baggingTime);
		m_resultWriter->addKeyValue("testingTime",m_classificationTime);
		m_resultWriter->addKeyValue("totalTime",m_classificationTime+m_buildTime+m_baggingTime);
		m_resultWriter->addKeyValue("testingInstances",m_evaluation->getNumTestingInstances());
		m_resultWriter->addKeyValue("trainingInstances",m_evaluation->getNumTrainingInstances());
		m_resultWriter->addKeyValue("auc",m_auc);
		m_resultWriter->addKeyValue("memUsageCPU",memUsageCPU/(1024*1024));
		m_resultWriter->addKeyValue("memUsageGPU",memUsageGPU/(1024*1024));
		m_resultWriter->addKeyValue("cl1EnrichmentFactor",m_enrichCl1);
		m_resultWriter->addKeyValue("cl2EnrichmentFactor",m_enrichCl2);

		m_internalNodes = 0;
		m_leafNodes = 0;
		m_depth = 0;
		m_totalTime = 0;
		for(unsigned int i=0; i<m_forests.size(); ++i){
			m_internalNodes += m_forests[i]->m_internalNodes;
			m_leafNodes += m_forests[i]->m_leafNodes;
			m_depth = max(m_depth,m_forests[i]->m_depth);
			m_totalTime = max(m_totalTime,m_forests[i]->m_totalTime);
		}

		m_outputStream <<
			"	Total time:			" << m_totalTime << "s\r\n" <<
			"	Time spent on bagging:		" << m_baggingTime << "s\r\n" <<
			"	Time spent on building trees:		" << m_buildTime << "s\r\n" <<
			"	Time spent on classification:		" << m_classificationTime << "s\r\n\r\n";

		if(m_kernelSpecificTimings){
			std::map<std::wstring,double>::iterator kernelItr = m_kernelTimes.begin();
			while(kernelItr != m_kernelTimes.end()){
				m_outputStream <<
					"	Time spent on kernel " << kernelItr->first << ":	" << kernelItr->second << "s\r\n";
				kernelItr++;
			}
		}

		m_outputStream << 
			"\r\n	Nodes built: " << m_internalNodes+m_leafNodes << " (" << m_internalNodes << " internal nodes and "<< m_leafNodes <<" leaf nodes)\r\n" <<
			"	Deepest tree depth reached: " << m_depth;

		m_outputStream	<< "\r\n	";
		
		std::vector<std::vector<unsigned int>> confusMatrix(m_document->getNumClassValues(),std::vector<unsigned int>(m_document->getNumClassValues(),0));
		for(unsigned int i=0; i<m_document->getNumClassValues(); ++i){
			for(unsigned int j=0; j<m_document->getNumClassValues(); ++j){
				if(i == j)
					confusMatrix[i][j] = m_clCorrect[i];
				else
					confusMatrix[i][j] = m_clWrong[i];
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

		m_outputStream << 
			"\r\n	Accuracy: " << m_accuracy << "%" << "\r\n" <<
			"	AUC: " << m_auc << "\r\n" <<
			"	Enrichment factor Cl1: " << m_enrichCl1 << " (" << m_maxEnrichCl1 << ")\r\n" <<
			"	Enrichment factor Cl2: " << m_enrichCl2 << " (" << m_maxEnrichCl2 << ")\r\n";
	}

	void GPURandomForest::updateResourceBatchGPU(){
		// Initialize constant buffer
		m_constants.cb_attributeCount = m_document->getNumAttributes();
		m_constants.cb_instanceCount = m_evaluation->getNumTrainingInstances();
		m_constants.cb_numFeatures = m_numFeatures;
		m_constants.cb_maxDepth = m_MaxDepth;
		m_constants.cb_numTrees = m_maxTreesPerIteration;
		m_constants.cb_currentDepth = 0;
		m_constants.cb_nodeBufferEnd = m_maxNodesPerIteration-1;
		m_constants.cb_maxInstInNodes = m_maxInstInNodes;
		m_constantsClassify.cb_nodeBufferEnd = m_constants.cb_nodeBufferEnd;

		std::vector<unsigned int> innerNodeIdsInit;
		innerNodeIdsInit.reserve(m_maxTreesPerIteration);
		for(unsigned int i=0; i<m_maxTreesPerIteration; ++i){
			innerNodeIdsInit.push_back(i);
		}

		std::vector<unsigned int> bagWeightInit(m_maxTreesPerIteration*m_evaluation->getNumTrainingInstances(),0);
		std::vector<Value::v_precision> splitValInit(m_maxNodesPerIteration,-FLT_MAX);
		std::vector<int> attributeVecInit(m_maxNodesPerIteration,-3);
		std::vector<int> checkInit(4,0);
		checkInit[0] = m_maxTreesPerIteration;
		checkInit[1] = 0;
		checkInit[2] = 0;
		checkInit[3] = 0;
		m_gfxMgr->copyToGPU(m_devId,m_bufferIds["RFB_checkVariables"],&checkInit[0],checkInit.size()*sizeof(unsigned int));
		m_gfxMgr->copyToGPU(m_devId,m_bufferIds["RFB_bagWeights"],&bagWeightInit[0],bagWeightInit.size()*sizeof(unsigned int));
		m_gfxMgr->copyToGPU(m_devId,m_bufferIds["RFB_attributeBuffer"],&attributeVecInit[0],attributeVecInit.size()*sizeof(int));
		m_gfxMgr->copyToGPU(m_devId,m_bufferIds["RFB_splitVal"],&splitValInit[0],splitValInit.size()*sizeof(Value::v_precision));
		m_gfxMgr->copyToGPU(m_devId,m_bufferIds["RFB_innerNodeIds"],&innerNodeIdsInit[0],innerNodeIdsInit.size()*sizeof(unsigned int));
		
		runBaggingProcess();
	}

	void GPURandomForest::initResourceBatch(bool updateOnly){
		// Clear the existing buffers if any
		if(m_bufferIds.find("RFB_nodeIndicesLimits") != m_bufferIds.end()){
			if(updateOnly){
				updateResourceBatchGPU();
				return;
			}

			m_gfxMgr->deleteBuffer(m_devId,m_bufferIds["RFB_nodeIndicesLimits"]);
			m_gfxMgr->deleteBuffer(m_devId,m_bufferIds["RFB_nodeIndices"]);
			m_gfxMgr->deleteBuffer(m_devId,m_bufferIds["RFB_splitPoints"]);
			m_gfxMgr->deleteBuffer(m_devId,m_bufferIds["RFB_attributeBuffer"]);
			m_gfxMgr->deleteBuffer(m_devId,m_bufferIds["RFB_classProbs"]);
			m_gfxMgr->deleteBuffer(m_devId,m_bufferIds["RFB_treeIds"]);
			m_gfxMgr->deleteBuffer(m_devId,m_bufferIds["RFB_childIds"]);
			m_gfxMgr->deleteBuffer(m_devId,m_bufferIds["RFB_bagWeights"]);
			m_gfxMgr->deleteBuffer(m_devId,m_bufferIds["RFB_checkVariables"]);
			m_gfxMgr->deleteBuffer(m_devId,m_bufferIds["RFB_distBuffer"]);
			m_gfxMgr->deleteBuffer(m_devId,m_bufferIds["RFB_nodeIndicesMirror"]);
			m_gfxMgr->deleteBuffer(m_devId,m_bufferIds["RFB_splitVal"]);
			m_gfxMgr->deleteBuffer(m_devId,m_bufferIds["RFB_votes"]);
			m_gfxMgr->deleteBuffer(m_devId,m_bufferIds["RFB_innerNodeIds"]);
		}

		std::vector<unsigned int> voteInit(m_evaluation->getNumTestingInstances()*2,0);

		GraphicsManager::bufferFlags flags = GraphicsManager::bufferFlags();
		m_bufferIds["RFB_nodeIndicesLimits"] = m_gfxMgr->createBuffer(m_devId,flags,m_maxNodesPerIteration*2,NULL,2*m_maxNodesPerIteration*sizeof(unsigned int),GraphicsManager::GRT_WBUFFER,GraphicsManager::GRF_UINT);
		m_resourceTypeIds["RFB_nodeIndicesLimits"] = GraphicsManager::GRT_WBUFFER;
		m_bufferIds["RFB_nodeIndices"] = m_gfxMgr->createBuffer(m_devId,flags,m_maxTreesPerIteration*m_evaluation->getNumTrainingInstances(),NULL,m_maxTreesPerIteration*m_evaluation->getNumTrainingInstances()*sizeof(unsigned int),GraphicsManager::GRT_WBUFFER,GraphicsManager::GRF_UINT);
		m_resourceTypeIds["RFB_nodeIndices"] = GraphicsManager::GRT_WBUFFER;
		m_bufferIds["RFB_nodeIndicesMirror"] = m_gfxMgr->createBuffer(m_devId,flags,m_maxTreesPerIteration*m_evaluation->getNumTrainingInstances(),NULL,m_maxTreesPerIteration*m_evaluation->getNumTrainingInstances()*sizeof(unsigned int),GraphicsManager::GRT_WBUFFER,GraphicsManager::GRF_UINT);
		m_resourceTypeIds["RFB_nodeIndicesMirror"] = GraphicsManager::GRT_WBUFFER;
		m_bufferIds["RFB_splitPoints"] = m_gfxMgr->createBuffer(m_devId,flags,m_maxNodesPerIteration,NULL,m_maxNodesPerIteration*sizeof(Value::v_precision),GraphicsManager::GRT_WBUFFER,GraphicsManager::GRF_REAL);
		m_resourceTypeIds["RFB_splitPoints"] = GraphicsManager::GRT_WBUFFER;
		m_bufferIds["RFB_splitVal"] = m_gfxMgr->createBuffer(m_devId,flags,m_maxNodesPerIteration,NULL,m_maxNodesPerIteration*sizeof(Value::v_precision),GraphicsManager::GRT_WBUFFER,GraphicsManager::GRF_REAL);
		m_resourceTypeIds["RFB_splitVal"] = GraphicsManager::GRT_WBUFFER;
		m_bufferIds["RFB_attributeBuffer"] = m_gfxMgr->createBuffer(m_devId,flags,m_maxNodesPerIteration,NULL,m_maxNodesPerIteration*sizeof(int),GraphicsManager::GRT_WBUFFER,GraphicsManager::GRF_INT);
		m_resourceTypeIds["RFB_attributeBuffer"] = GraphicsManager::GRT_WBUFFER;
		m_bufferIds["RFB_classProbs"] = m_gfxMgr->createBuffer(m_devId,flags,2*m_maxNodesPerIteration,NULL,2*m_maxNodesPerIteration*sizeof(Value::v_precision),GraphicsManager::GRT_WBUFFER,GraphicsManager::GRF_REAL);
		m_resourceTypeIds["RFB_classProbs"] = GraphicsManager::GRT_WBUFFER;
		m_bufferIds["RFB_treeIds"] = m_gfxMgr->createBuffer(m_devId,flags,m_maxNodesPerIteration,NULL,m_maxNodesPerIteration*sizeof(unsigned int),GraphicsManager::GRT_WBUFFER,GraphicsManager::GRF_UINT);
		m_resourceTypeIds["RFB_treeIds"] = GraphicsManager::GRT_WBUFFER;
		m_bufferIds["RFB_childIds"] = m_gfxMgr->createBuffer(m_devId,flags,m_maxNodesPerIteration,NULL,m_maxNodesPerIteration*sizeof(unsigned int),GraphicsManager::GRT_WBUFFER,GraphicsManager::GRF_UINT);
		m_resourceTypeIds["RFB_childIds"] = GraphicsManager::GRT_WBUFFER;
		m_bufferIds["RFB_bagWeights"] = m_gfxMgr->createBuffer(m_devId,flags,m_maxTreesPerIteration*m_evaluation->getNumTrainingInstances(),NULL,m_maxTreesPerIteration*m_evaluation->getNumTrainingInstances()*sizeof(unsigned int),GraphicsManager::GRT_RESOURCE,GraphicsManager::GRF_UINT);
		m_resourceTypeIds["RFB_bagWeights"] = GraphicsManager::GRT_RESOURCE;
		m_bufferIds["RFB_checkVariables"] = m_gfxMgr->createBuffer(m_devId,flags,4,NULL,4*sizeof(int),GraphicsManager::GRT_WBUFFER,GraphicsManager::GRF_INT,GraphicsFunctionPtr());
		m_resourceTypeIds["RFB_checkVariables"] = GraphicsManager::GRT_WBUFFER;
		m_bufferIds["RFB_distBuffer"] = m_gfxMgr->createBuffer(m_devId,flags,AttributeMaxNominal*2*m_maxNodesPerIteration,NULL,AttributeMaxNominal*2*m_maxNodesPerIteration*sizeof(unsigned int),GraphicsManager::GRT_WBUFFER,GraphicsManager::GRF_UINT);
		m_resourceTypeIds["RFB_distBuffer"] = GraphicsManager::GRT_WBUFFER;
		m_bufferIds["RFB_votes"] = m_gfxMgr->createBuffer(m_devId,flags,voteInit.size(),&voteInit[0],voteInit.size()*sizeof(unsigned int),GraphicsManager::GRT_WBUFFER,GraphicsManager::GRF_UINT);
		m_resourceTypeIds["RFB_votes"] = GraphicsManager::GRT_WBUFFER;
		m_bufferIds["RFB_innerNodeIds"] = m_gfxMgr->createBuffer(m_devId,flags,2*m_maxNodesPerIteration,NULL,2*m_maxNodesPerIteration*sizeof(unsigned int),GraphicsManager::GRT_WBUFFER,GraphicsManager::GRF_UINT);
		m_resourceTypeIds["RFB_innerNodeIds"] = GraphicsManager::GRT_WBUFFER;
		setBufferSettings();

		updateResourceBatchGPU();
	}
	
	void GPURandomForest::initResources(){
		unsigned int memAfter, memBefore;
		memBefore = m_gfxMgr->getAvailableMemory(m_devId);

		std::vector<unsigned int> typeVec(m_document->getNumAttributes());
		for(unsigned int i=0; i<typeVec.size(); ++i){
			typeVec[i] = m_document->getAttribute(i)->getNumValues();
		}

		// Generate random seeds for GPU RNGs
		boost::random::uniform_int_distribution<> indRand(200,INT_MAX);
		boost::random::mt19937 rng;
		rng.seed(m_seed);
		std::vector<unsigned int> seedVector(m_maxNodesPerIteration*4);
		for(unsigned int i=0; i<seedVector.size(); ++i){
			seedVector[i] = indRand(rng);
		}

		// Generate class values vector
		std::vector<int> classValueVec(m_evaluation->getNumTrainingInstances());
		for(unsigned int i=0; i<m_evaluation->getNumTrainingInstances(); ++i){
			classValueVec[i] = m_evaluation->getTrainingInstance(i)->classValue();
		}

		// Generate suitable data representation of training data
		std::vector<Value::v_precision> dataVec;
		std::vector<unsigned int> dataIndVec;
		dataIndVec.reserve(m_document->getNumAttributes()*m_evaluation->getNumTrainingInstances());
		dataVec.reserve(m_document->getNumAttributes()*m_evaluation->getNumTrainingInstances());
		for(int a = 0; a < m_document->getNumAttributes(); ++a){
			for(int i = 0; i < m_evaluation->getNumTrainingInstances(); ++i){
				dataVec.push_back(m_evaluation->getTrainingInstance(i)->getValue(a,true));
				dataIndVec.push_back(i);
			}
			if(m_version != 3){
				quickSort(dataVec,dataIndVec,a*m_evaluation->getNumTrainingInstances(),a*m_evaluation->getNumTrainingInstances()+m_evaluation->getNumTrainingInstances()-1);
				std::vector<unsigned int> indVec(m_evaluation->getNumTrainingInstances(),0);
				for(unsigned int i=0; i<m_evaluation->getNumTrainingInstances(); ++i){
					indVec[dataIndVec[i+m_evaluation->getNumTrainingInstances()*a]] = i;
				}
				memcpy(&dataIndVec[m_evaluation->getNumTrainingInstances()*a],&indVec[0],sizeof(unsigned int)*m_evaluation->getNumTrainingInstances());
			}
		}

		// Get testing instances
		std::vector<Value::v_precision> testSetVec;
		testSetVec.reserve(m_document->getNumAttributes()*m_evaluation->getNumTestingInstances());
		for(unsigned int i=0; i<m_evaluation->getNumTestingInstances(); ++i){
			int classValue = m_evaluation->getTestingInstance(i)->classValue();
			m_testSetClassVec.push_back(classValue);
			for(unsigned int j=0; j<m_document->getNumAttributes(); ++j){
				testSetVec.push_back(m_evaluation->getTestingInstance(i)->getValue(j));
			}
		}

		m_constantsClassify.cb_attributeCount = m_document->getNumAttributes();
		m_constantsClassify.cb_instanceCount = m_evaluation->getNumTestingInstances();
		m_testSetVotes.assign(m_constantsClassify.cb_instanceCount*2,0);

		// Create gpu programs
		m_gpuFunctionIds["RFP_Bagging"] = m_gfxMgr->createGPUProgram(m_devId,"RF_Bagging",GraphicsFunctionPtr(new CUDA_RandomForestBagging));
		m_gpuFunctionIds["RFP_Build"] = m_gfxMgr->createGPUProgram(m_devId,"RF_Build",GraphicsFunctionPtr(new CUDA_RandomForestBuild));
		m_gpuFunctionIds["RFP_Sort"] = m_gfxMgr->createGPUProgram(m_devId,"RF_Sort",GraphicsFunctionPtr(new CUDA_RandomForestSort));
		m_gpuFunctionIds["RFP_FindSplit"] = m_gfxMgr->createGPUProgram(m_devId,"RF_FindSplit",GraphicsFunctionPtr(new CUDA_RandomForestFindSplit));
		m_gpuFunctionIds["RFP_EvaluateSplit"] = m_gfxMgr->createGPUProgram(m_devId,"RF_EvaluateSplit",GraphicsFunctionPtr(new CUDA_RandomForestEvaluateSplit));
		m_gpuFunctionIds["RFP_Split"] = m_gfxMgr->createGPUProgram(m_devId,"RF_Split",GraphicsFunctionPtr(new CUDA_RandomForestSplitData));
		m_gpuFunctionIds["RFP_Classify"] = m_gfxMgr->createGPUProgram(m_devId,"RF_Classify",GraphicsFunctionPtr(new CUDA_RandomForestClassify));
		m_gpuFunctionIds["RFP_KeplerBuild"] = m_gfxMgr->createGPUProgram(m_devId,"RF_KeplerBuild",GraphicsFunctionPtr(new CUDA_RandomForestKeplerBuild));
		m_gpuFunctionIds["RFP_ExFindSplit"] = m_gfxMgr->createGPUProgram(m_devId,"RF_ExFindSplit",GraphicsFunctionPtr(new CUDA_RandomForestExFindSplit));
		m_gpuFunctionIds["RFP_ExMakeSplit"] = m_gfxMgr->createGPUProgram(m_devId,"RF_ExMakeSplit",GraphicsFunctionPtr(new CUDA_RandomForestExMakeSplit));
		m_gpuFunctionIds["RFP_ExCreateNodes"] = m_gfxMgr->createGPUProgram(m_devId,"RF_ExCreateNodes",GraphicsFunctionPtr(new CUDA_RandomForestExCreateNodes));

		m_gpuFunctionIds["RFP_OOBE"] = m_gfxMgr->createGPUProgram(m_devId,"RF_OOBE");
		
		// Create gpu buffers
		GraphicsManager::bufferFlags flags = GraphicsManager::bufferFlags();
		m_bufferIds["RFB_constants"] = m_gfxMgr->createBuffer(m_devId,flags,1,NULL,sizeof(m_constants),GraphicsManager::GRT_CBUFFER,GraphicsManager::GRF_REAL,GraphicsFunctionPtr(new CUDA_RandomForestUpdateConstants));
		m_resourceTypeIds["RFB_constants"] = GraphicsManager::GRT_CBUFFER;

		m_bufferIds["RFB_statesRNG"] = m_gfxMgr->createBuffer(m_devId,flags,seedVector.size(),&seedVector[0],seedVector.size()*sizeof(unsigned int),GraphicsManager::GRT_WBUFFER,GraphicsManager::GRF_UINT);
		m_resourceTypeIds["RFB_statesRNG"] = GraphicsManager::GRT_WBUFFER;

		m_bufferIds["RFB_dataSet"] = m_gfxMgr->createBuffer(m_devId,flags,dataVec.size(),&dataVec[0],dataVec.size()*sizeof(Value::v_precision),GraphicsManager::GRT_RESOURCE,GraphicsManager::GRF_REAL,GraphicsFunctionPtr());
		m_resourceTypeIds["RFB_dataSet"] = GraphicsManager::GRT_RESOURCE;
		if(m_version != 3){
			m_bufferIds["RFB_dataSetInds"] = m_gfxMgr->createBuffer(m_devId,flags,dataIndVec.size(),&dataIndVec[0],dataIndVec.size()*sizeof(unsigned int),GraphicsManager::GRT_RESOURCE,GraphicsManager::GRF_UINT,GraphicsFunctionPtr());
			m_resourceTypeIds["RFB_dataSetInds"] = GraphicsManager::GRT_RESOURCE;
		}
		m_bufferIds["RFB_classValues"] =m_gfxMgr->createBuffer(m_devId,flags,classValueVec.size(),&classValueVec[0],classValueVec.size()*sizeof(int),GraphicsManager::GRT_RESOURCE,GraphicsManager::GRF_INT,GraphicsFunctionPtr());
		m_resourceTypeIds["RFB_classValues"] = GraphicsManager::GRT_RESOURCE;
		m_bufferIds["RFB_testSet"] = m_gfxMgr->createBuffer(m_devId,flags,testSetVec.size(),&testSetVec[0],testSetVec.size()*sizeof(Value::v_precision),GraphicsManager::GRT_RESOURCE,GraphicsManager::GRF_REAL,GraphicsFunctionPtr());
		m_resourceTypeIds["RFB_testSet"] = GraphicsManager::GRT_RESOURCE;

		m_bufferIds["RFB_attributeTypeBuffer"] = m_gfxMgr->createBuffer(m_devId,flags,typeVec.size(),&typeVec[0],typeVec.size()*sizeof(unsigned int),GraphicsManager::GRT_RESOURCE,GraphicsManager::GRF_UINT,GraphicsFunctionPtr());
		m_resourceTypeIds["RFB_attributeTypeBuffer"] = GraphicsManager::GRT_RESOURCE;
		
		memAfter = m_gfxMgr->getAvailableMemory(m_devId);

		std::wstringstream ss;
		ss << "\r\nMemory allocated: " << float(memBefore-memAfter)/float(1024*1024) << "mb\r\n";
		m_data->m_gui->postDebugMessage(ss.str());
	}
	
	void GPURandomForest::cleanResources(){
		std::map<std::string,int>::iterator itr = m_gpuFunctionIds.begin();
		while(itr != m_gpuFunctionIds.end()){
			m_gfxMgr->deleteGPUProgram(m_devId,itr->second);
			itr++;
		}
		itr = m_bufferIds.begin();
		while(itr != m_bufferIds.end()){
			m_gfxMgr->deleteBuffer(m_devId,itr->second);
			itr++;
		}
		m_gpuFunctionIds.clear();
		m_bufferIds.clear();
		m_resourceTypeIds.clear();
		m_setBufferIdsBuild.clear();
		m_setBufferIdsBagging.clear();
		m_setBufferIdsSort.clear();
		m_setBufferIdsDist.clear();
		m_setBufferIdsEval.clear();
		m_setBufferIdsSplit.clear();
		m_setBufferIdsClassify.clear();
		m_setResourceTypesBuild.clear();
		m_setResourceTypesBagging.clear();
		m_setResourceTypesSort.clear();
		m_setResourceTypesDist.clear();
		m_setResourceTypesEval.clear();
		m_setResourceTypesSplit.clear();
		m_setResourceTypesClassify.clear();
		m_splitPointsVec.clear();
		m_classProbVec.clear();
		m_attributeVec.clear();
		m_childVec.clear();
		m_baseIndVec.clear();
		m_testSetClassVec.clear();
		m_testSetVotes.clear();
		m_forests.clear();
	}

	void GPURandomForest::quickSort(std::vector<Value::v_precision> &vals, std::vector<unsigned int> &index, int left, int right){
		if(left < right){
			int middle = partition(vals, index, left, right);
			quickSort(vals, index, left, middle);
			quickSort(vals, index, middle + 1, right);
		}
	}

	int GPURandomForest::partition(std::vector<Value::v_precision> &vals, std::vector<unsigned int> &index, int l, int r){
		Value::v_precision pivot = vals[(l + r) / 2];
		int tmp1;
		Value::v_precision tmp2;

		while(l < r) {
			while((vals[l] < pivot) && (l < r)){
				l++;
			}
			while((vals[r] > pivot) && (l < r)){
				r--;
			}
			if (l < r) {
				tmp2 = vals[l];
				vals[l] = vals[r];
				vals[r] = tmp2;

				tmp1 = index[l];
				index[l] = index[r];
				index[r] = tmp1;
				
				l++;
				r--;
			}
		}
		if((l == r) && (vals[r] > pivot)){
			r--;
		}

		return r;
	}

	void GPURandomForest::quickSort(std::vector<Value::v_precision> &vals, std::vector<int> &index, int left, int right, std::vector<unsigned int>& sortBuff){
		if(left < right){
			int middle = partition(vals, index, left, right, sortBuff);
			quickSort(vals, index, left, middle, sortBuff);
			quickSort(vals, index, middle + 1, right, sortBuff);
		}
	}

	int GPURandomForest::partition(std::vector<Value::v_precision> &vals, std::vector<int> &index, int l, int r, std::vector<unsigned int>& sortBuff){
		Value::v_precision pivot = vals[index[sortBuff[(l + r) / 2]]];
		int tmp1;
		Value::v_precision tmp2;

		while(l < r) {
			while((vals[index[sortBuff[l]]] < pivot) && (l < r)){
				l++;
			}
			while((vals[index[sortBuff[r]]] > pivot) && (l < r)){
				r--;
			}
			if (l < r) {
				tmp2 = sortBuff[l];
				sortBuff[l] = sortBuff[r];
				sortBuff[r] = tmp2;
				
				l++;
				r--;
			}
		}
		if((l == r) && (vals[index[sortBuff[r]]] > pivot)){
			r--;
		}

		return r;
	}

	void GPURandomForest::setBufferSettings(){
		// Build buffer settings
		m_setBufferIdsBuild.clear();
		m_setBufferIdsBuild.push_back(m_bufferIds["RFB_dataSetInds"]);
		m_setBufferIdsBuild.push_back(m_bufferIds["RFB_bagWeights"]);
		m_setBufferIdsBuild.push_back(m_bufferIds["RFB_statesRNG"]);
		m_setBufferIdsBuild.push_back(m_bufferIds["RFB_nodeIndicesLimits"]);
		m_setBufferIdsBuild.push_back(m_bufferIds["RFB_nodeIndices"]);
		m_setBufferIdsBuild.push_back(m_bufferIds["RFB_treeIds"]);
		m_setBufferIdsBuild.push_back(m_bufferIds["RFB_childIds"]);
		m_setBufferIdsBuild.push_back(m_bufferIds["RFB_classValues"]);
		m_setBufferIdsBuild.push_back(m_bufferIds["RFB_attributeBuffer"]);
		m_setBufferIdsBuild.push_back(m_bufferIds["RFB_checkVariables"]);
		m_setBufferIdsBuild.push_back(m_bufferIds["RFB_splitPoints"]);
		m_setBufferIdsBuild.push_back(m_bufferIds["RFB_dataSet"]);
		m_setBufferIdsBuild.push_back(m_bufferIds["RFB_classProbs"]);
		
		m_setResourceTypesBuild.clear();
		m_setResourceTypesBuild.push_back(m_resourceTypeIds["RFB_dataSetInds"]);
		m_setResourceTypesBuild.push_back(m_resourceTypeIds["RFB_bagWeights"]);
		m_setResourceTypesBuild.push_back(m_resourceTypeIds["RFB_statesRNG"]);
		m_setResourceTypesBuild.push_back(m_resourceTypeIds["RFB_nodeIndicesLimits"]);
		m_setResourceTypesBuild.push_back(m_resourceTypeIds["RFB_nodeIndices"]);
		m_setResourceTypesBuild.push_back(m_resourceTypeIds["RFB_treeIds"]);
		m_setResourceTypesBuild.push_back(m_resourceTypeIds["RFB_childIds"]);
		m_setResourceTypesBuild.push_back(m_resourceTypeIds["RFB_classValues"]);
		m_setResourceTypesBuild.push_back(m_resourceTypeIds["RFB_attributeBuffer"]);
		m_setResourceTypesBuild.push_back(m_resourceTypeIds["RFB_checkVariables"]);
		m_setResourceTypesBuild.push_back(m_resourceTypeIds["RFB_splitPoints"]);
		m_setResourceTypesBuild.push_back(m_resourceTypeIds["RFB_dataSet"]);
		m_setResourceTypesBuild.push_back(m_resourceTypeIds["RFB_classProbs"]);

		// Build buffer settings
		m_setBufferIdsKeplerBuild.clear();
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_dataSetInds"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_bagWeights"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_statesRNG"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_nodeIndicesLimits"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_nodeIndices"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_nodeIndicesMirror"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_treeIds"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_childIds"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_distBuffer"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_classValues"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_attributeBuffer"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_checkVariables"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_splitVal"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_dataSet"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_classProbs"]);
		m_setBufferIdsKeplerBuild.push_back(m_bufferIds["RFB_splitPoints"]);
		
		m_setResourceTypesKeplerBuild.clear();
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_dataSetInds"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_bagWeights"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_statesRNG"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_nodeIndicesLimits"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_nodeIndices"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_nodeIndicesMirror"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_treeIds"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_childIds"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_distBuffer"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_classValues"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_attributeBuffer"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_checkVariables"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_splitVal"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_dataSet"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_classProbs"]);
		m_setResourceTypesKeplerBuild.push_back(m_resourceTypeIds["RFB_splitPoints"]);

		// Bagging buffer setting
		m_setBufferIdsBagging.clear();
		m_setBufferIdsBagging.push_back(m_bufferIds["RFB_statesRNG"]);
		m_setBufferIdsBagging.push_back(m_bufferIds["RFB_treeIds"]);
		m_setBufferIdsBagging.push_back(m_bufferIds["RFB_bagWeights"]);
		m_setBufferIdsBagging.push_back(m_bufferIds["RFB_nodeIndicesLimits"]);
		m_setBufferIdsBagging.push_back(m_bufferIds["RFB_nodeIndices"]);
		m_setBufferIdsBagging.push_back(m_bufferIds["RFB_nodeIndicesMirror"]);
		m_setBufferIdsBagging.push_back(m_bufferIds["RFB_classValues"]);
		m_setBufferIdsBagging.push_back(m_bufferIds["RFB_classProbs"]);

		m_setResourceTypesBagging.clear();
		m_setResourceTypesBagging.push_back(m_resourceTypeIds["RFB_statesRNG"]);
		m_setResourceTypesBagging.push_back(m_resourceTypeIds["RFB_treeIds"]);
		m_setResourceTypesBagging.push_back(m_resourceTypeIds["RFB_bagWeights"]);
		m_setResourceTypesBagging.push_back(m_resourceTypeIds["RFB_nodeIndicesLimits"]);
		m_setResourceTypesBagging.push_back(m_resourceTypeIds["RFB_nodeIndices"]);
		m_setResourceTypesBagging.push_back(m_resourceTypeIds["RFB_nodeIndicesMirror"]);
		m_setResourceTypesBagging.push_back(m_resourceTypeIds["RFB_classValues"]);
		m_setResourceTypesBagging.push_back(m_resourceTypeIds["RFB_classProbs"]);

		// Sort buffer settings
		m_setBufferIdsSort.clear();
		m_setBufferIdsSort.push_back(m_bufferIds["RFB_nodeIndicesLimits"]);
		m_setBufferIdsSort.push_back(m_bufferIds["RFB_nodeIndices"]);
		m_setBufferIdsSort.push_back(m_bufferIds["RFB_nodeIndicesMirror"]);
		m_setBufferIdsSort.push_back(m_bufferIds["RFB_dataSetInds"]);
		m_setBufferIdsSort.push_back(m_bufferIds["RFB_statesRNG"]);
		m_setBufferIdsSort.push_back(m_bufferIds["RFB_attributeTypeBuffer"]);
		m_setBufferIdsSort.push_back(m_bufferIds["RFB_innerNodeIds"]);
		
		m_setResourceTypesSort.clear();
		m_setResourceTypesSort.push_back(m_resourceTypeIds["RFB_nodeIndicesLimits"]);
		m_setResourceTypesSort.push_back(m_resourceTypeIds["RFB_nodeIndices"]);
		m_setResourceTypesSort.push_back(m_resourceTypeIds["RFB_nodeIndicesMirror"]);
		m_setResourceTypesSort.push_back(m_resourceTypeIds["RFB_dataSetInds"]);
		m_setResourceTypesSort.push_back(m_resourceTypeIds["RFB_statesRNG"]);
		m_setResourceTypesSort.push_back(m_resourceTypeIds["RFB_attributeTypeBuffer"]);
		m_setResourceTypesSort.push_back(m_resourceTypeIds["RFB_innerNodeIds"]);

		// Distribution kernel settings
		m_setBufferIdsDist.clear();
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_nodeIndicesLimits"]);
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_nodeIndices"]);
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_treeIds"]);
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_bagWeights"]);
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_distBuffer"]);
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_statesRNG"]);
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_dataSetInds"]);
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_attributeTypeBuffer"]);
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_innerNodeIds"]);
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_classValues"]);
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_attributeBuffer"]);
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_splitPoints"]);
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_dataSet"]);
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_splitVal"]);
		m_setBufferIdsDist.push_back(m_bufferIds["RFB_classProbs"]);
		
		m_setResourceTypesDist.clear();
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_nodeIndicesLimits"]);
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_nodeIndices"]);
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_treeIds"]);
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_bagWeights"]);
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_distBuffer"]);
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_statesRNG"]);
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_dataSetInds"]);
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_attributeTypeBuffer"]);
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_innerNodeIds"]);
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_classValues"]);
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_attributeBuffer"]);
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_splitPoints"]);
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_dataSet"]);
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_splitVal"]);
		m_setResourceTypesDist.push_back(m_resourceTypeIds["RFB_classProbs"]);

		// Evaluate split buffer settings
		m_setBufferIdsEval.clear();
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_dataSetInds"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_bagWeights"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_statesRNG"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_nodeIndicesLimits"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_nodeIndices"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_treeIds"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_childIds"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_distBuffer"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_nodeIndicesMirror"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_attributeTypeBuffer"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_innerNodeIds"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_classValues"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_attributeBuffer"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_checkVariables"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_splitPoints"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_dataSet"]);
		m_setBufferIdsEval.push_back(m_bufferIds["RFB_classProbs"]);
		
		m_setResourceTypesEval.clear();
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_dataSetInds"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_bagWeights"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_statesRNG"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_nodeIndicesLimits"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_nodeIndices"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_treeIds"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_childIds"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_distBuffer"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_nodeIndicesMirror"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_attributeTypeBuffer"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_innerNodeIds"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_classValues"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_attributeBuffer"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_checkVariables"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_splitPoints"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_dataSet"]);
		m_setResourceTypesEval.push_back(m_resourceTypeIds["RFB_classProbs"]);

		// Split data buffer settings
		m_setBufferIdsSplit.clear();
		m_setBufferIdsSplit.push_back(m_bufferIds["RFB_dataSetInds"]);
		m_setBufferIdsSplit.push_back(m_bufferIds["RFB_nodeIndicesLimits"]);
		m_setBufferIdsSplit.push_back(m_bufferIds["RFB_nodeIndices"]);
		m_setBufferIdsSplit.push_back(m_bufferIds["RFB_nodeIndicesMirror"]);
		m_setBufferIdsSplit.push_back(m_bufferIds["RFB_attributeTypeBuffer"]);
		m_setBufferIdsSplit.push_back(m_bufferIds["RFB_innerNodeIds"]);
		m_setBufferIdsSplit.push_back(m_bufferIds["RFB_distBuffer"]);
		m_setBufferIdsSplit.push_back(m_bufferIds["RFB_attributeBuffer"]);
		m_setBufferIdsSplit.push_back(m_bufferIds["RFB_dataSet"]);
		m_setBufferIdsSplit.push_back(m_bufferIds["RFB_splitPoints"]);

		m_setResourceTypesSplit.clear();
		m_setResourceTypesSplit.push_back(m_resourceTypeIds["RFB_dataSetInds"]);
		m_setResourceTypesSplit.push_back(m_resourceTypeIds["RFB_nodeIndicesLimits"]);
		m_setResourceTypesSplit.push_back(m_resourceTypeIds["RFB_nodeIndices"]);
		m_setResourceTypesSplit.push_back(m_resourceTypeIds["RFB_nodeIndicesMirror"]);
		m_setResourceTypesSplit.push_back(m_resourceTypeIds["RFB_attributeTypeBuffer"]);
		m_setResourceTypesSplit.push_back(m_resourceTypeIds["RFB_innerNodeIds"]);
		m_setResourceTypesSplit.push_back(m_resourceTypeIds["RFB_distBuffer"]);
		m_setResourceTypesSplit.push_back(m_resourceTypeIds["RFB_attributeBuffer"]);
		m_setResourceTypesSplit.push_back(m_resourceTypeIds["RFB_dataSet"]);
		m_setResourceTypesSplit.push_back(m_resourceTypeIds["RFB_splitPoints"]);

		// Classify data buffer settings
		m_setBufferIdsClassify.clear();
		m_setBufferIdsClassify.push_back(m_bufferIds["RFB_attributeTypeBuffer"]);
		m_setBufferIdsClassify.push_back(m_bufferIds["RFB_childIds"]);
		m_setBufferIdsClassify.push_back(m_bufferIds["RFB_votes"]);
		m_setBufferIdsClassify.push_back(m_bufferIds["RFB_attributeBuffer"]);
		m_setBufferIdsClassify.push_back(m_bufferIds["RFB_testSet"]);
		m_setBufferIdsClassify.push_back(m_bufferIds["RFB_classProbs"]);
		m_setBufferIdsClassify.push_back(m_bufferIds["RFB_splitPoints"]);

		m_setResourceTypesClassify.clear();
		m_setResourceTypesClassify.push_back(m_resourceTypeIds["RFB_attributeTypeBuffer"]);
		m_setResourceTypesClassify.push_back(m_resourceTypeIds["RFB_childIds"]);
		m_setResourceTypesClassify.push_back(m_resourceTypeIds["RFB_votes"]);
		m_setResourceTypesClassify.push_back(m_resourceTypeIds["RFB_attributeBuffer"]);
		m_setResourceTypesClassify.push_back(m_resourceTypeIds["RFB_testSet"]);
		m_setResourceTypesClassify.push_back(m_resourceTypeIds["RFB_classProbs"]);
		m_setResourceTypesClassify.push_back(m_resourceTypeIds["RFB_splitPoints"]);
		
		// Ex find split buffers
		m_setBufferIdsExFindSplit.clear();
		m_setBufferIdsExFindSplit.push_back(m_bufferIds["RFB_treeIds"]);
		m_setBufferIdsExFindSplit.push_back(m_bufferIds["RFB_statesRNG"]);
		m_setBufferIdsExFindSplit.push_back(m_bufferIds["RFB_nodeIndices"]);
		m_setBufferIdsExFindSplit.push_back(m_bufferIds["RFB_distBuffer"]);
		m_setBufferIdsExFindSplit.push_back(m_bufferIds["RFB_classValues"]);
		m_setBufferIdsExFindSplit.push_back(m_bufferIds["RFB_bagWeights"]);
		m_setBufferIdsExFindSplit.push_back(m_bufferIds["RFB_nodeIndicesLimits"]);
		m_setBufferIdsExFindSplit.push_back(m_bufferIds["RFB_attributeTypeBuffer"]);
		m_setBufferIdsExFindSplit.push_back(m_bufferIds["RFB_innerNodeIds"]);
		m_setBufferIdsExFindSplit.push_back(m_bufferIds["RFB_childIds"]);
		m_setBufferIdsExFindSplit.push_back(m_bufferIds["RFB_attributeBuffer"]);
		m_setBufferIdsExFindSplit.push_back(m_bufferIds["RFB_splitVal"]);
		m_setBufferIdsExFindSplit.push_back(m_bufferIds["RFB_splitPoints"]);
		m_setBufferIdsExFindSplit.push_back(m_bufferIds["RFB_dataSet"]);

		m_setResourceTypesExFindSplit.clear();
		m_setResourceTypesExFindSplit.push_back(m_resourceTypeIds["RFB_treeIds"]);
		m_setResourceTypesExFindSplit.push_back(m_resourceTypeIds["RFB_statesRNG"]);
		m_setResourceTypesExFindSplit.push_back(m_resourceTypeIds["RFB_nodeIndices"]);
		m_setResourceTypesExFindSplit.push_back(m_resourceTypeIds["RFB_distBuffer"]);
		m_setResourceTypesExFindSplit.push_back(m_resourceTypeIds["RFB_classValues"]);
		m_setResourceTypesExFindSplit.push_back(m_resourceTypeIds["RFB_bagWeights"]);
		m_setResourceTypesExFindSplit.push_back(m_resourceTypeIds["RFB_nodeIndicesLimits"]);
		m_setResourceTypesExFindSplit.push_back(m_resourceTypeIds["RFB_attributeTypeBuffer"]);
		m_setResourceTypesExFindSplit.push_back(m_resourceTypeIds["RFB_innerNodeIds"]);
		m_setResourceTypesExFindSplit.push_back(m_resourceTypeIds["RFB_childIds"]);
		m_setResourceTypesExFindSplit.push_back(m_resourceTypeIds["RFB_attributeBuffer"]);
		m_setResourceTypesExFindSplit.push_back(m_resourceTypeIds["RFB_splitVal"]);
		m_setResourceTypesExFindSplit.push_back(m_resourceTypeIds["RFB_splitPoints"]);
		m_setResourceTypesExFindSplit.push_back(m_resourceTypeIds["RFB_dataSet"]);

		// Ex make split buffers
		m_setBufferIdsExMakeSplit.clear();
		m_setBufferIdsExMakeSplit.push_back(m_bufferIds["RFB_attributeBuffer"]);
		m_setBufferIdsExMakeSplit.push_back(m_bufferIds["RFB_nodeIndices"]);
		m_setBufferIdsExMakeSplit.push_back(m_bufferIds["RFB_nodeIndicesMirror"]);
		m_setBufferIdsExMakeSplit.push_back(m_bufferIds["RFB_nodeIndicesLimits"]);
		m_setBufferIdsExMakeSplit.push_back(m_bufferIds["RFB_attributeTypeBuffer"]);
		m_setBufferIdsExMakeSplit.push_back(m_bufferIds["RFB_distBuffer"]);
		m_setBufferIdsExMakeSplit.push_back(m_bufferIds["RFB_innerNodeIds"]);
		m_setBufferIdsExMakeSplit.push_back(m_bufferIds["RFB_childIds"]);
		m_setBufferIdsExMakeSplit.push_back(m_bufferIds["RFB_classValues"]);
		m_setBufferIdsExMakeSplit.push_back(m_bufferIds["RFB_checkVariables"]);
		m_setBufferIdsExMakeSplit.push_back(m_bufferIds["RFB_splitPoints"]);
		m_setBufferIdsExMakeSplit.push_back(m_bufferIds["RFB_dataSet"]);

		m_setResourceTypesExMakeSplit.clear();
		m_setResourceTypesExMakeSplit.push_back(m_resourceTypeIds["RFB_attributeBuffer"]);
		m_setResourceTypesExMakeSplit.push_back(m_resourceTypeIds["RFB_nodeIndices"]);
		m_setResourceTypesExMakeSplit.push_back(m_resourceTypeIds["RFB_nodeIndicesMirror"]);
		m_setResourceTypesExMakeSplit.push_back(m_resourceTypeIds["RFB_nodeIndicesLimits"]);
		m_setResourceTypesExMakeSplit.push_back(m_resourceTypeIds["RFB_attributeTypeBuffer"]);
		m_setResourceTypesExMakeSplit.push_back(m_resourceTypeIds["RFB_distBuffer"]);
		m_setResourceTypesExMakeSplit.push_back(m_resourceTypeIds["RFB_innerNodeIds"]);
		m_setResourceTypesExMakeSplit.push_back(m_resourceTypeIds["RFB_childIds"]);
		m_setResourceTypesExMakeSplit.push_back(m_resourceTypeIds["RFB_classValues"]);
		m_setResourceTypesExMakeSplit.push_back(m_resourceTypeIds["RFB_checkVariables"]);
		m_setResourceTypesExMakeSplit.push_back(m_resourceTypeIds["RFB_splitPoints"]);
		m_setResourceTypesExMakeSplit.push_back(m_resourceTypeIds["RFB_dataSet"]);

		// Ex create nodes buffers
		m_setBufferIdsExCreateNodes.clear();
		m_setBufferIdsExCreateNodes.push_back(m_bufferIds["RFB_attributeBuffer"]);
		m_setBufferIdsExCreateNodes.push_back(m_bufferIds["RFB_nodeIndices"]);
		m_setBufferIdsExCreateNodes.push_back(m_bufferIds["RFB_nodeIndicesMirror"]);
		m_setBufferIdsExCreateNodes.push_back(m_bufferIds["RFB_distBuffer"]);
		m_setBufferIdsExCreateNodes.push_back(m_bufferIds["RFB_childIds"]);
		m_setBufferIdsExCreateNodes.push_back(m_bufferIds["RFB_nodeIndicesLimits"]);
		m_setBufferIdsExCreateNodes.push_back(m_bufferIds["RFB_checkVariables"]);
		m_setBufferIdsExCreateNodes.push_back(m_bufferIds["RFB_attributeTypeBuffer"]);
		m_setBufferIdsExCreateNodes.push_back(m_bufferIds["RFB_innerNodeIds"]);
		m_setBufferIdsExCreateNodes.push_back(m_bufferIds["RFB_treeIds"]);
		m_setBufferIdsExCreateNodes.push_back(m_bufferIds["RFB_classProbs"]);
		m_setBufferIdsExCreateNodes.push_back(m_bufferIds["RFB_splitPoints"]);
		
		m_setResourceTypesExCreateNodes.clear();
		m_setResourceTypesExCreateNodes.push_back(m_resourceTypeIds["RFB_attributeBuffer"]);
		m_setResourceTypesExCreateNodes.push_back(m_resourceTypeIds["RFB_nodeIndices"]);
		m_setResourceTypesExCreateNodes.push_back(m_resourceTypeIds["RFB_nodeIndicesMirror"]);
		m_setResourceTypesExCreateNodes.push_back(m_resourceTypeIds["RFB_distBuffer"]);
		m_setResourceTypesExCreateNodes.push_back(m_resourceTypeIds["RFB_childIds"]);
		m_setResourceTypesExCreateNodes.push_back(m_resourceTypeIds["RFB_nodeIndicesLimits"]);
		m_setResourceTypesExCreateNodes.push_back(m_resourceTypeIds["RFB_checkVariables"]);
		m_setResourceTypesExCreateNodes.push_back(m_resourceTypeIds["RFB_attributeTypeBuffer"]);
		m_setResourceTypesExCreateNodes.push_back(m_resourceTypeIds["RFB_innerNodeIds"]);
		m_setResourceTypesExCreateNodes.push_back(m_resourceTypeIds["RFB_treeIds"]);
		m_setResourceTypesExCreateNodes.push_back(m_resourceTypeIds["RFB_classProbs"]);
		m_setResourceTypesExCreateNodes.push_back(m_resourceTypeIds["RFB_splitPoints"]);
	}
}