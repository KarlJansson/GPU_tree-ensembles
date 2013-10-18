#pragma once

namespace DataMiner{
	typedef boost::shared_ptr<boost::function<void(void)>> TM_runFunctionPtr;
	typedef boost::shared_ptr<boost::function<void(int)>> TM_callbackFunctionPtr;

	class ThreadManager{
	public:
		static void initialize();
		static void shutdown();
		static int launchWorkPackage(boost::shared_ptr<boost::function<void(void)>> func, boost::shared_ptr<boost::function<void(int)>> callback);
		static int queueWorkPackage(boost::shared_ptr<boost::function<void(void)>> func, boost::shared_ptr<boost::function<void(int)>> callback);
		static void executeWorkQueue();
	private:
		ThreadManager();

		struct workPackage{
			unsigned int workId;
			boost::shared_ptr<boost::function<void(void)>> func;
			boost::shared_ptr<boost::function<void(int)>> callback;
		};

		static void threadFunction(int threadId, workPackage pack);

		static int m_maxThreads;
		static unsigned int m_workId;

		static std::queue<int> m_freeThreads;
		static std::map<int,boost::shared_ptr<boost::thread>> m_threadPool;
		static std::queue<workPackage> m_workQueue;

		static boost::mutex m_manageMutex;
	};
}