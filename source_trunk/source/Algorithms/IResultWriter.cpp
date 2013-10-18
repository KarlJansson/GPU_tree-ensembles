#include "stdafx.h"
#include "IResultWriter.h"

namespace DataMiner{
	IResultWriter::IResultWriter(){
		
	}

	std::string IResultWriter::toString(){
		return "";
	}

	void IResultWriter::writeToFile(std::string filename){
		
	}

	void IResultWriter::addKeyValue(std::string key, double val){
		m_variableMap[key].push_back(val);
	}
	
	double IResultWriter::getDouble(std::string key){
		std::vector<double> &vec = m_variableMap[key];
		double result = 0;

		for(unsigned int i=0; i<vec.size(); ++i){
			result += vec[i];
		}

		return result;
	}
	
	int IResultWriter::getInt(std::string key){
		double res = getDouble(key);
		return res;
	}

	std::vector<double> IResultWriter::getDoubleVector(std::string key){
		std::vector<double> result;
		std::vector<double> &vec = m_variableMap[key];
		for(unsigned int i=0; i<vec.size(); ++i){
			result.push_back(vec[i]);
		}
		return result;
	}
}