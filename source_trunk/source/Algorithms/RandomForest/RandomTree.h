#pragma once
#include "DataCache.h"

namespace DataMiner{
	class RandomTree{
	public:
		RandomTree(unsigned int seed);

		virtual void build() = 0;
		unsigned int getNumNodes() { return m_nodesInTree; }
	protected:
		std::vector<boost::shared_ptr<RandomTree>> m_Successors;

		std::vector<double> m_Prop, m_ClassProbs;
		int m_Attribute;
		double m_SplitPoint;
		unsigned int m_nodesInTree;

		boost::shared_ptr<RandomForest> m_MotherForest;
		int m_MinNum;
		bool m_Debug;

		DataCachePtr data;

		int getMaxDepth();
		int getMinNum();
		int getKValue();
		static void normalize(std::vector<double> &doubles);
		static void normalize(std::vector<double> &doubles, double sum);

		static float fast_log2(float val);
		static double lnFunc(double num);
		static int maxIndex(std::vector<double> &vec);
		static double sum(std::vector<double> &vec);
		std::vector<double> countsToFreqs(std::vector<std::vector<double>> &dist);

		double distribution(std::vector<std::vector<double>> &props, std::vector<std::vector<std::vector<double>>> &dists, int attIndex, std::vector<int> &sortedIndices);
		void splitData(std::vector<std::vector<std::vector<int>>> &subsetIndices, int m_Attribute, float m_SplitPoint, std::vector<std::vector<int>> &sortedIndices);

		virtual std::vector<double> distributionForInstance(InstancePtr instance) = 0;
		virtual double classifyInstance(InstancePtr instance) = 0;

		// Split functions
		double entropyConditionedOnRows(std::vector<std::vector<double>> &matrix);
		double entropyOverColumns(std::vector<std::vector<double>> &matrix);

		unsigned int m_seed;

		friend class Bagger;
		friend class RandomForest;
		friend class VoteCollector;
		friend class RecursiveRandomTree;
		friend class SerialRandomTree;
	};
}