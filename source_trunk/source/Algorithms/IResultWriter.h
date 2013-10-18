#pragma once

namespace DataMiner{
	class IResultWriter{
	public:
		IResultWriter();

		void writeToFile(std::string filename);
		std::string toString();

		void addKeyValue(std::string key, double val);
		
		double getDouble(std::string key);
		int getInt(std::string key);

		std::vector<double> getDoubleVector(std::string key);
	protected:
		std::map<std::string,std::vector<double>> m_variableMap;
	};
}