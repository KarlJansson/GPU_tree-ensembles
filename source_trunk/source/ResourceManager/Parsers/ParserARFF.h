#pragma once
#include "IParser.h"

namespace DataMiner{
	class ParserARFF : public IParser{
	public:
		DataDocumentPtr parse(boost::filesystem::path path);
	private:
	};
}