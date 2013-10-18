#pragma once
#include "IParser.h"

namespace DataMiner{
	class ParserRaw : public IParser{
	public:
		DataDocumentPtr parse(boost::filesystem::path path);
	private:
	};
}