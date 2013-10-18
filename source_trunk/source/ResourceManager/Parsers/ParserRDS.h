#pragma once
#include "IParser.h"

namespace DataMiner{
	class ParserRDS : public IParser{
	public:
		DataDocumentPtr parse(boost::filesystem::path path);
	private:
	};
}