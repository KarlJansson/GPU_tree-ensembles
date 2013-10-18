#pragma once
#include "Value.h"

#define AttributeMaxNominal 20

namespace DataMiner{
	class Attribute{
	public:
		enum InputFormat { IF_NUMERIC, IF_NOMINAL };

		Attribute(std::string name):m_name(name),m_format(IF_NOMINAL),m_numValues(0),m_max(-ValueMax),m_min(ValueMax),m_missingSubstitute(-ValueMax),m_numMissing(0){}

		Value::v_precision getValue(unsigned int ind, bool replaceMissing = true, bool normalized = false);
		bool missing(unsigned int ind);
		unsigned int getNumMissing();
		int getNumValues(){ return m_numValues; }
		std::string& getName() { return m_name; }
		InputFormat getInputFormat() { return m_format; }
		Value::v_precision getMinValue() { return m_min; }
		Value::v_precision getMaxValue() { return m_max; }

		void calculateMissingSubstitute();
	private:
		std::vector<Value*> m_valueVector;
		std::string m_name;
		unsigned int m_numValues;
		unsigned int m_numMissing;

		Value::v_precision m_missingSubstitute;

		Value::v_precision m_max,m_min;
		InputFormat m_format;
		std::map<Value::v_precision,unsigned int> m_valueMap;

		void insertValue(Value* value, int pos = -1);

		friend class DataDocument;
	};
}