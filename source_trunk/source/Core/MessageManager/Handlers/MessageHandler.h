#pragma once
#include "IDataPack.h"

namespace DataMiner{
	class MessageHandler{
	public:
		virtual void handle(IDataPackPtr dataPack) = 0;
		virtual void stop() = 0;
	private:
	};
}

typedef boost::shared_ptr<DataMiner::MessageHandler> MessageHandlerPtr;