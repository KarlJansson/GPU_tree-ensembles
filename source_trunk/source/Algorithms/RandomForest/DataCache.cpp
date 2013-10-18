#include "stdafx.h"
#include "DataCache.h"
#include "IEvaluation.h"

namespace DataMiner{
	DataCache::DataCache(DataDocumentPtr origData, IEvaluationPtr eval, unsigned int seed){
		numAttributes = origData->getNumAttributes();
		numClasses = origData->getNumClassValues();
		numInstances = eval->getNumTrainingInstances();
		m_rng.seed(seed);

		attNumVals = boost::shared_ptr<std::vector<int>>(new std::vector<int>(numAttributes));
		for(size_t i = 0; i < attNumVals->size(); i++) {
			if(origData->isNumericAttribute(i)) {
				(*attNumVals)[i] = 0;
			} 
			else if(origData->isNominalAttribute(i)){
				(*attNumVals)[i] = origData->getAttribute(i)->getNumValues();
			}
		}

		/* Array is indexed by attribute first, to speed access in RF splitting. */
		vals = boost::shared_ptr<std::vector<std::vector<float>>>(new std::vector<std::vector<float>>(numAttributes,std::vector<float>(numInstances,0)));
		for (int a = 0; a < numAttributes; a++) {
			for (int i = 0; i < numInstances; i++) {
				(*vals)[a][i] = (float) eval->getTrainingInstance(i)->getValue(a);  // deep copy
			}
		}

		classValuesSeparated = std::vector<std::vector<int>>(numClasses,std::vector<int>());
		instWeights = boost::shared_ptr<std::vector<double>>(new std::vector<double>(numInstances,0));
		instClassValues = boost::shared_ptr<std::vector<int>>(new std::vector<int>(numInstances,0));
		for (int i = 0; i < numInstances; i++) {
			(*instWeights)[i] += eval->getTrainingInstance(i)->weight();
			(*instClassValues)[i] = eval->getTrainingInstance(i)->classValue();
			classValuesSeparated[(*instClassValues)[i]].push_back(i);
		}

		/* compute the sortedInstances for the whole dataset */
    
		sortedIndices = boost::shared_ptr<std::vector<std::vector<int>>>(new std::vector<std::vector<int>>(/*numAttributes*/1,std::vector<int>(numInstances,0)));
		for(int i = 0; i < numInstances; i++) {
			(*sortedIndices)[0][i] = i;
		}
		return;

		for(int a = 0; a < numAttributes; a++) { // ================= attr by attr
			//if(a == classIndex)
			//	continue;

			if((*attNumVals)[a] > 0) { // ------------------------------------- nominal
				// Handling nominal attributes. Putting indices of
				// instances with missing values at the end.
        
				(*sortedIndices)[a] = std::vector<int>(numInstances,0);
				int count = 0;

				for(int i = 0; i < numInstances; i++) {
					if(!this->isValueMissing(a, i) ) {
						(*sortedIndices)[a][count] = i;
						count++;
					}
				}

				for (int i = 0; i < numInstances; i++) {
					if ( this->isValueMissing(a, i) ) {
						(*sortedIndices)[a][count] = i;
						count++;
					}
				}
			} 
			else{ // ----------------------------------------------------- numeric
				// Sorted indices are computed for numeric attributes
				// missing values are coded as Float.MAX_VALUE and go to the end
				(*sortedIndices)[a] = sort((*vals)[a]); 
			} // ---------------------------------------------------------- attr kind
		} // ========================================================= attr by attr
		// System.out.println(" Done.");
	}

	std::vector<int> DataCache::sort(std::vector<float> &values){
		std::vector<float> vals = values;
		std::vector<int> index(values.size(),0);
		for(size_t i = 0; i < index.size(); i++)
			index[i] = i;
		quickSort(vals, index, 0, vals.size() - 1);
		return index;
	}

	void DataCache::quickSort(std::vector<float> &vals, std::vector<int> &index, int left, int right){
		if(left < right){
			int middle = partition(vals, index, left, right);
			quickSort(vals, index, left, middle);
			quickSort(vals, index, middle + 1, right);
		}
	}

	int DataCache::partition(std::vector<float> &vals, std::vector<int> &index, int l, int r){
		double pivot = vals[index[(l + r) / 2]];
		int help;

		while(l < r) {
			while((vals[index[l]] < pivot) && (l < r)){
				l++;
			}
			while((vals[index[r]] > pivot) && (l < r)){
				r--;
			}
			if (l < r) {
				help = index[l];
				index[l] = index[r];
				index[r] = help;
				l++;
				r--;
			}
		}
		if((l == r) && (vals[index[r]] > pivot)){
			r--;
		}

		return r;
	}

	DataCache::DataCache(DataCache* origData){
		numAttributes = origData->numAttributes; // copied
		numClasses = origData->numClasses;       // copied
		numInstances = origData->numInstances;   // copied
		m_rng = origData->m_rng;

		attNumVals = origData->attNumVals;       // shallow copied
		instClassValues =
				origData->instClassValues;       // shallow copied
		vals = origData->vals;                   // shallow copied - very big array!
		sortedIndices = origData->sortedIndices; // shallow copied - also big

		instWeights = origData->instWeights;     // shallow copied

		inBag = std::vector<bool>(numInstances,false);      // gets its own inBag array
		numInBag = 0;
    
		whatGoesWhere = std::vector<int>();     // this will be created when tree building starts

	}

	void DataCache::createInBagSortedIndices(){
		boost::shared_ptr<std::vector<std::vector<int>>> newSortedIndices = boost::shared_ptr<std::vector<std::vector<int>>>(new std::vector<std::vector<int>>(/*numAttributes*/1,std::vector<int>(this->numInBag,0)));
    
		for(int a = 0; a < /*numAttributes*/1; a++){
			//if (a == classIndex)
			//	continue;      
      
			(*newSortedIndices)[/*a*/0] = std::vector<int>(this->numInBag,0);
      
			int inBagIdx = 0;
			for (size_t i = 0; i < (*sortedIndices)[/*a*/0].size(); i++) {
				int origIdx = (*sortedIndices)[/*a*/0][i];
				if ( !this->inBag[origIdx] )
					continue;
				(*newSortedIndices)[/*a*/0][inBagIdx] = (*sortedIndices)[/*a*/0][i];
				inBagIdx++;
			}
		}
    
		this->sortedIndices = newSortedIndices;
	}

	boost::shared_ptr<DataCache> DataCache::resample(int bagSize){
		DataCachePtr result =
            DataCachePtr(new DataCache(this)); // makes shallow copy of vals matrix

		boost::shared_ptr<std::vector<double>> newWeights = boost::shared_ptr<std::vector<double>>(new std::vector<double>(numInstances,0)); // all 0.0 by default
    
		int classMax = classValuesSeparated.size();
		int classDec = 0;
		boost::random::uniform_int_distribution<> indRand(0,numInstances-1);
		for(int r = 0; r < bagSize; r++){
			if(classDec == classMax)
				classDec = 0;

			indRand = boost::random::uniform_int_distribution<>(0,classValuesSeparated[classDec].size()-1);
			int curIdx = classValuesSeparated[classDec][indRand(m_rng)]; //random.nextInt( numInstances );
			(*newWeights)[curIdx] += (*instWeights)[curIdx];
			if(!result->inBag[curIdx]){
				result->numInBag++;
				result->inBag[curIdx] = true;
			}
			classDec++;
		}

		result->instWeights = newWeights;

		// we also need to fill sortedIndices by peeking into the inBag array, but
		// this can be postponed until the tree training begins
		// we will use the "createInBagSortedIndices()" for this

		return result;
	}

	bool DataCache::isValueMissing(int attIndex, int instIndex){
		return (*this->vals)[attIndex][instIndex] == FLT_MAX;
	}

	bool DataCache::isAttrNominal(int attIndex){
		return (*attNumVals)[attIndex] > 0;
	}
}