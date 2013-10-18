#pragma once
#include "DataDocument.h"

namespace DataMiner{
	class DataCache{
	public:
		DataCache(DataDocumentPtr origData, IEvaluationPtr eval, unsigned int seed);
		DataCache(DataCache* origData);

		bool isValueMissing(int attIndex, int instIndex);
		bool isAttrNominal(int attIndex);
		void createInBagSortedIndices();
		boost::shared_ptr<DataCache> resample(int bagSize);

		std::vector<int> sort(std::vector<float> &values);
		void quickSort(std::vector<float> &vals, std::vector<int> &index, int left, int right);
		int partition(std::vector<float> &vals, std::vector<int> &index, int l, int r);

		boost::shared_ptr<std::vector<std::vector<float>>> vals;

		/**
		 * Attribute description - holds a 0 for numeric attributes, and the number
		 * of available categories for nominal attributes.
		 */
		boost::shared_ptr<std::vector<int>> attNumVals;

		/** Number of attributes, including the class attribute. */
		int numAttributes;

		/** Number of classes. */
		int numClasses;

		/** Number of instances. */
		int numInstances;
  
		/** The class an instance belongs to. */
		boost::shared_ptr<std::vector<int>> instClassValues;

		/** Ordering of instances, indexed by attribute, then by instance. */ 
		boost::shared_ptr<std::vector<std::vector<int>>> sortedIndices;
  
		/** Weights of instances. */
		boost::shared_ptr<std::vector<double>> instWeights;
  
		/** Is instance in 'bag' created by bootstrap sampling. */
		std::vector<bool> inBag;
		/** How many instances are in 'bag' created by bootstrap sampling. */
		int numInBag;

		/** Used in training of FastRandomTrees. */
		std::vector<int> whatGoesWhere;

		boost::random::mt19937 m_rng;

		std::vector<std::vector<int>> classValuesSeparated;
	};
}