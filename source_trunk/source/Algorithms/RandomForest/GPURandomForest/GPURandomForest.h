#pragma once
#include "IAlgorithm.h"

namespace DataMiner{
	class GPURandomForest : public IAlgorithm{
	public:
		GPURandomForest(GraphicsManagerPtr gfxMgr);
		~GPURandomForest();
	
		struct SharedBuffer{
			unsigned int	cb_numTrees;
			unsigned int	cb_numFeatures;
			unsigned int	cb_maxDepth;
			unsigned int	cb_currentDepth;
			unsigned int	cb_availableNodes;
			unsigned int	cb_nodeBufferStart;
			unsigned int	cb_nodeBufferEnd;
			unsigned int	cb_maxInstInNodes;

			unsigned int	cb_instanceCount;
			unsigned int	cb_attributeCount;
			unsigned int	cb_nodeIdFlip;
		};

		struct ConstantBufferBagging{
			unsigned int	cb_treeOffset;
			unsigned int	cb_instanceCount;
			unsigned int	cb_nodeBufferEnd;
			bool			cb_baggingActivated;
		};

		struct ConstantBufferClassify{
			unsigned int	cb_numTrees;
			unsigned int	cb_treeOffset;
			unsigned int	cb_nodeBufferEnd;
			unsigned int	cb_majorityClass;

			unsigned int	cb_instanceCount;
			unsigned int	cb_attributeCount;
		};

		enum KernelID {KID_Bagging = 0, KID_Build, KID_KeplerBuild, KID_Sort, KID_FindSplit, KID_Split, KID_EvaluateSplit, KID_Classify, KID_ExtremeFindSplit, KID_ExtremeCreateNodes, KID_ExtremeMakeSplit};
		struct ConstantUpdate{
			ConstantUpdate(void* content, int id):m_content(content),m_id(id){}
			void* m_content;
			int m_id;
		};

	protected:
		void run();

		void calculateMaxNodes(int devId);
		void writeOutputStream();

		void runBaggingProcess();
		void runClassificationProcess(unsigned int trees);

		void getResultsFromGPU();
		void getVotesFromGPU();
		void evaluateTrees();

		void initResources();
		void initResourceBatch(bool updateOnly);
		void updateResourceBatchGPU();
		void cleanResources();
		void setBufferSettings();

		void quickSort(std::vector<Value::v_precision> &vals, std::vector<unsigned int> &index, int left, int right);
		int partition(std::vector<Value::v_precision> &vals, std::vector<unsigned int> &index, int l, int r);

		void quickSort(std::vector<Value::v_precision> &vals, std::vector<int> &index, int left, int right, std::vector<unsigned int>& sortBuff);
		int partition(std::vector<Value::v_precision> &vals, std::vector<int> &index, int l, int r, std::vector<unsigned int>& sortBuff);

		GraphicsManagerPtr m_gfxMgr;
		std::map<std::string,int> 
			m_gpuFunctionIds,
			m_bufferIds;
		
		std::map<std::string,GraphicsManager::ResourceType> m_resourceTypeIds;

		std::vector<int>	
			m_setBufferIdsBuild,
			m_setBufferIdsBagging,
			m_setBufferIdsSort,
			m_setBufferIdsDist,
			m_setBufferIdsEval,
			m_setBufferIdsSplit,
			m_setBufferIdsClassify,
			m_setBufferIdsKeplerBuild,
			m_setBufferIdsExFindSplit,
			m_setBufferIdsExMakeSplit,
			m_setBufferIdsExCreateNodes;
		
		std::vector<GraphicsManager::ResourceType>	
			m_setResourceTypesBuild,
			m_setResourceTypesBagging,
			m_setResourceTypesSort,
			m_setResourceTypesDist,
			m_setResourceTypesEval,
			m_setResourceTypesSplit,
			m_setResourceTypesClassify,
			m_setResourceTypesKeplerBuild,
			m_setResourceTypesExFindSplit,
			m_setResourceTypesExMakeSplit,
			m_setResourceTypesExCreateNodes;

		unsigned int	
			m_MaxDepth,
			m_KValue,
			m_numTrees,
			m_numFeatures,
			m_maxNodesPerIteration,
			m_maxTreesPerIteration,
			m_indVecSize,
			m_seed,
			m_depth,
			m_clCorrect[2],
			m_clWrong[2],
			m_internalNodes,
			m_leafNodes;

		int m_maxInstInNodes,
			m_version;

		SharedBuffer m_constants;
		ConstantBufferBagging m_constantsBagging;
		ConstantBufferClassify m_constantsClassify;

		std::vector<Value::v_precision> 
			m_splitPointsVec,
			m_classProbVec;
		
		std::vector<int> 
			m_attributeVec;
		
		std::vector<unsigned int> 
			m_childVec,
			m_baseIndVec,
			m_testSetClassVec,
			m_testSetVotes;

		std::map<std::wstring,double> m_kernelTimes;

		double 
			m_baggingTime,
			m_buildTime,
			m_classificationTime,
			m_totalTime,
			m_accuracy,
			m_enrichCl1,
			m_enrichCl2,
			m_maxEnrichCl1,
			m_maxEnrichCl2,
			m_auc;

		bool
			m_saveModel,
			m_kernelSpecificTimings;

		int m_devId;
		std::vector<GPURandomForestPtr> m_forests;
	};
}