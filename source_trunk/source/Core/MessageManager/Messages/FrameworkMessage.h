#pragma once
#include "IDataPack.h"

namespace DataMiner{
	class FrameworkMessage{
	public:
		FrameworkMessage(std::string message, IDataPackPtr dataPack, int barrier):m_message(message),m_dataPack(dataPack){
			m_barrier = BarrierPtr();
			if(barrier > 0)
				m_barrier = BarrierPtr(new boost::barrier(barrier+1));
		}

		std::string& getMessage() { return m_message; }
		IDataPackPtr getDataPack() { return m_dataPack; }
		void waitOnMessage() { if(m_barrier) m_barrier->wait(); }
	private:
		IDataPackPtr m_dataPack;
		std::string m_message;
		BarrierPtr m_barrier;
	};
}

typedef boost::shared_ptr<DataMiner::FrameworkMessage> FrameworkMessagePtr;