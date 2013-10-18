#pragma once
#include "Value.h"
#include "Instance.h"
#include "Attribute.h"

namespace DataMiner{
#define MissingVal -FLT_MAX

	class DataDocument : public boost::enable_shared_from_this<DataDocument>{
	public:
		DataDocument();

		InstancePtr getInstance(unsigned int index);
		AttributePtr getAttribute(unsigned int index);

		unsigned int getNumInstances();
		unsigned int getNumAttributes();
		unsigned int getNumMissing() { return m_missing.size(); }

		bool isNominalAttribute(int att);
		bool isNumericAttribute(int att);

		void addMissingValueMarker(std::string val);
		void addClassAttributeMarker(std::string val);

		bool isClassAttribute(std::string val);
		bool isMissingValueMarker(std::string val);

		int nominalValue(std::string value, int attribute);

		Attribute::InputFormat getClassFormat() { return m_classFormat; }
		unsigned int getNumClassValues() { return m_classValues.size(); }

	private:
		void addAttribute(AttributePtr attribute);
		void addAttributeValue(Value value, int attribute, bool missing = false);
		void addAttributeValue(std::string value, int attribute, bool missing = false);
		void addClassValue(Value value);
		void addClassValue(std::string value);
		void buildInstances();

		std::map<std::string,unsigned int> m_classValues;
		unsigned int m_classValCount;

		std::vector<Value> m_attributeData;
		std::vector<Value> m_classData;
		std::set<unsigned int> m_missing;
		std::vector<InstancePtr> m_instances;
		std::vector<AttributePtr> m_attributes;

		Attribute::InputFormat m_classFormat;

		std::map<std::string,int> m_classNominalMap;
		unsigned int m_nominalMapCount;

		std::map<int,unsigned int> m_nominalCount;
		std::map<std::string,std::map<int,int>> m_nominalMap;

		std::set<std::string> m_missingValueMarkers;
		std::set<std::string> m_classAttributeMarkers;

		friend class ParserARFF;
		friend class ParserRaw;
		friend class ParserRDS;
		friend class IParser;
	};
}