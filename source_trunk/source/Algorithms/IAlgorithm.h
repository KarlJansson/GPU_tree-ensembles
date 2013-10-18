#pragma once
#include "AlgorithmDataPack.h"
#include "IEvaluation.h"
#include "IResultWriter.h"

namespace DataMiner{
	class IAlgorithm : public boost::enable_shared_from_this<IAlgorithm>{
	public:
		typedef Value::v_precision alg_precision;

		IAlgorithm():m_stop(false){m_resultWriter = IResultWriterPtr(new IResultWriter);}
		virtual ~IAlgorithm(){}
		void runAlgorithm(AlgorithmDataPackPtr data, IEvaluationPtr eval);

		std::wstringstream& getoutputStream()	{return m_outputStream;}
		IResultWriterPtr getResultWriter()		{ return m_resultWriter; }

		void stop() { m_stop = true; }
		void start() { m_stop = false; }
	protected:
		virtual void run() = 0;

		std::wstringstream m_outputStream;
		IResultWriterPtr m_resultWriter;

		DataDocumentPtr m_document;
		AlgorithmDataPackPtr m_data;

		IEvaluationPtr m_evaluation;

		bool m_stop;
	};
}