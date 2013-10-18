#pragma once

namespace DataMiner{
	class Runnable{
	public:
		virtual void run(void *params) = 0;
	};
}

typedef boost::shared_ptr<DataMiner::Runnable> RunnablePtr;