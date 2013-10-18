#include "stdafx.h"
#include "GPUERT.h"

namespace DataMiner{
	GPUERT::GPUERT(GraphicsManagerPtr gfxMgr):GPURandomForest(gfxMgr){
		
	}
	
	GPUERT::~GPUERT(){
		cleanResources();
		m_gfxMgr->destroyDeviceContext(m_devId);
	}

	void GPUERT::runBuildProcess(int devId, int trees, BarrierPtr bar){
		m_numTrees = trees;
		m_devId = devId;
		m_version = 3;
		m_bar = bar;

		m_thread = ThreadPtr(new boost::thread(boost::bind(&GPUERT::deviceHandler,this)));
	}

	void GPUERT::deviceHandler(){
		m_devId = m_gfxMgr->createDeviceContext(m_devId);
		calculateMaxNodes(m_devId);
		initResources();

		m_constantsBagging.cb_baggingActivated = false;
		TimerPtr timerKernel = TimerPtr(new boost::timer);

		TimerPtr timer = TimerPtr(new boost::timer);
		TimerPtr timerTotal = TimerPtr(new boost::timer);

		m_internalNodes = 0, m_leafNodes = 0;
		int treesLeft = m_numTrees;
		int treesToLaunch = m_maxTreesPerIteration;
		int lastLaunch = 0;
		int checkSum = m_maxTreesPerIteration;
		int newNodes = m_maxTreesPerIteration;
		std::vector<int> checkVars(4,0);

		m_buildTime = 0;
		m_baggingTime = 0;
		m_classificationTime = 0;
		m_depth = 0;

		timerTotal->restart();
		int numIter = ceil(float(m_numTrees)/float(m_maxTreesPerIteration));
		for(unsigned int j=0; j<numIter; ++j){
			checkSum = treesToLaunch;
			newNodes = treesToLaunch;
			m_constants.cb_nodeBufferStart = 0;
			m_constants.cb_nodeIdFlip = 0;

			initResourceBatch(lastLaunch == treesToLaunch);
			lastLaunch = treesToLaunch;

			treesLeft -= treesToLaunch;
			timer->restart();
			for(unsigned int i=0; i<m_MaxDepth; ++i){
				if(i > m_depth)
					m_depth = i;

				assert(newNodes < m_maxNodesPerIteration);
				int nodeLimit = 10000;
				int innerNodes = newNodes;
				int numInnerIter = ceil(float(innerNodes)/float(nodeLimit));
				int launchCount = 0;

				m_constants.cb_availableNodes = newNodes;
				m_constants.cb_numFeatures = 0;
				timerKernel->restart();
				for(unsigned int k=0; k<m_numFeatures; ++k){
					innerNodes = newNodes;
					numInnerIter = ceil(float(innerNodes)/float(nodeLimit));
					launchCount = 0;

					// Best split kernel
					m_gfxMgr->setGPUBuffer(m_devId,m_setBufferIdsExFindSplit,m_setResourceTypesExFindSplit);
					m_gfxMgr->setGPUProgram(m_devId,m_gpuFunctionIds["RFP_ExFindSplit"]);

					innerNodes = newNodes;
					launchCount = 0;
					m_constants.cb_currentDepth = 0;
					for(unsigned int l=0; l<numInnerIter; ++l){
						launchCount = innerNodes > nodeLimit ? nodeLimit : innerNodes;
						
						m_gfxMgr->copyToGPU(m_devId,m_bufferIds["RFB_constants"],&ConstantUpdate(&m_constants,KID_ExtremeFindSplit),sizeof(m_constants));
						m_gfxMgr->launchComputation(m_devId,launchCount*thread_group_size,1,1);

						m_constants.cb_currentDepth += launchCount;
						innerNodes -= launchCount;
						if(innerNodes <= 0)
							break;
					}
					++m_constants.cb_numFeatures;
				}
				if(m_kernelSpecificTimings){
					m_gfxMgr->syncDevice(m_devId);
					m_kernelTimes[L"RFP_ExFindSplit"] += timerKernel->elapsed();
				}

				innerNodes = newNodes;
				launchCount = 0;
				m_constants.cb_currentDepth = 0;
				m_gfxMgr->setGPUBuffer(m_devId,m_setBufferIdsExMakeSplit,m_setResourceTypesExMakeSplit);
				m_gfxMgr->setGPUProgram(m_devId,m_gpuFunctionIds["RFP_ExMakeSplit"]);

				// Split data
				timerKernel->restart();
				for(unsigned int k=0; k<numInnerIter; ++k){
					launchCount = innerNodes > nodeLimit ? nodeLimit : innerNodes;
						
					m_gfxMgr->copyToGPU(m_devId,m_bufferIds["RFB_constants"],&ConstantUpdate(&m_constants,KID_ExtremeMakeSplit),sizeof(m_constants));
					m_gfxMgr->launchComputation(m_devId,launchCount*thread_group_size,1,1);

					m_constants.cb_currentDepth += launchCount;
					innerNodes -= launchCount;
					if(innerNodes <= 0)
						break;
				}
				if(m_kernelSpecificTimings){
					m_gfxMgr->syncDevice(m_devId);
					m_kernelTimes[L"RFP_ExMakeSplit"] += timerKernel->elapsed();
				}

				// Swap buffer to avoid unecessary copying
				int tmpBuffId = m_bufferIds["RFB_nodeIndices"];
				m_bufferIds["RFB_nodeIndices"] = m_bufferIds["RFB_nodeIndicesMirror"];
				m_bufferIds["RFB_nodeIndicesMirror"] = tmpBuffId;
				setBufferSettings();

				m_constants.cb_currentDepth = i;

				// Evaluate splits
				timerKernel->restart();
				m_gfxMgr->setGPUBuffer(m_devId,m_setBufferIdsExCreateNodes,m_setResourceTypesExCreateNodes);
				m_gfxMgr->copyToGPU(m_devId,m_bufferIds["RFB_constants"],&ConstantUpdate(&m_constants,KID_ExtremeCreateNodes),sizeof(m_constants));
				m_gfxMgr->setGPUProgram(m_devId,m_gpuFunctionIds["RFP_ExCreateNodes"]);
				m_gfxMgr->launchComputation(m_devId,newNodes,1,1);

				// Get continuation variables
				m_gfxMgr->copyFromGPU(m_devId,m_bufferIds["RFB_checkVariables"],&checkVars[0],checkVars.size()*sizeof(int));
				if(m_kernelSpecificTimings){
					m_gfxMgr->syncDevice(m_devId);
					m_kernelTimes[L"RFP_ExCreateNd"] += timerKernel->elapsed();
				}
			
				m_constants.cb_nodeBufferStart = checkSum;
				checkSum = checkVars[2];
				m_leafNodes += checkVars[3];
				newNodes = checkSum;
				m_internalNodes += checkSum;

				checkVars[1] = 0;
				checkVars[2] = 0;
				checkVars[3] = 0;
				m_gfxMgr->copyToGPU(m_devId,m_bufferIds["RFB_checkVariables"],&checkVars[0],checkVars.size()*sizeof(int));
				m_constants.cb_nodeIdFlip = (m_constants.cb_nodeIdFlip == 0) ? 1 : 0;

				if(newNodes <= 0)
					break;
			}
			m_buildTime += timer->elapsed();

			// Vote on test instances
			timer->restart();
			runClassificationProcess(treesToLaunch);
			m_classificationTime += timer->elapsed();

			if(m_saveModel)
				getResultsFromGPU();

			m_data->m_gui->setProgressBar(IDC_PROGRESSBAR_PROGRESS,100,(float(m_numTrees-treesLeft)/float(m_numTrees))*100);
			if(treesLeft != 0 && treesLeft < m_maxTreesPerIteration){
				treesToLaunch = treesLeft;
				m_maxTreesPerIteration = treesToLaunch;
			}
		}

		getVotesFromGPU();
		m_totalTime = timerTotal->elapsed();
		m_bar->wait();
	}
}