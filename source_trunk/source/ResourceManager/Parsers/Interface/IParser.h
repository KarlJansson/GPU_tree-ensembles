#pragma once
#include "DataDocument.h"

namespace DataMiner{
	class IParser{
	public:
		virtual DataDocumentPtr parse(boost::filesystem::path path) = 0;
	protected:
		void getToken(std::string &token, std::set<char> &stopChars);
		void beginParsing(boost::filesystem::path path);
		void endParsing();

		unsigned int	m_size,
						m_readPosition;
		char* m_buffer;
		DataDocumentPtr m_document;
	};
}

typedef boost::shared_ptr<DataMiner::IParser> IParserPtr;