#include "stdafx.h"
#include "IParser.h"
#include "ResourceManager.h"
#include "DataDocument.h"

namespace DataMiner{
	void IParser::getToken(std::string &token, std::set<char> &stopChars){
		while(m_readPosition < m_size){
			if(stopChars.find(m_buffer[++m_readPosition]) != stopChars.end())
				break;
			token.push_back(m_buffer[m_readPosition]);
		}
	}
	
	void IParser::beginParsing(boost::filesystem::path path){
		std::ifstream open(path.generic_string(),std::ios_base::binary);
		m_size = boost::filesystem::file_size(path);
		m_readPosition = 0;
		m_buffer = new char[m_size];
		open.read(m_buffer,m_size);
		open.close();

		m_document = DataDocumentPtr(new DataDocument);
	}

	void IParser::endParsing(){
		m_document->m_attributeData.shrink_to_fit();
		m_document->m_classData.shrink_to_fit();

		m_document->buildInstances();
		delete[] m_buffer;
	}
}