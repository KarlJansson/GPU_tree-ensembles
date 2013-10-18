#include "stdafx.h"
#include "ThreadManager.h"

namespace DataMiner{
	std::map<int,boost::shared_ptr<boost::thread>> ThreadManager::m_threadPool;
	std::queue<ThreadManager::workPackage> ThreadManager::m_workQueue;
	std::queue<int> ThreadManager::m_freeThreads;

	int ThreadManager::m_maxThreads;
	unsigned int ThreadManager::m_workId;

	boost::mutex ThreadManager::m_manageMutex;

	void ThreadManager::initialize(){
		m_workId = 0;

		SYSTEM_INFO sysinfo;
		GetSystemInfo(&sysinfo);
		m_maxThreads = sysinfo.dwNumberOfProcessors;

		std::string nrThreads = ConfigManager::getSetting("pThreads");
		if(nrThreads.compare("All") != 0){
			std::stringstream ss(nrThreads);
			int nr;
			ss >> nr;
			if(!ss.fail() && nr > 0)
				m_maxThreads = min(nr,m_maxThreads);
		}

		for(unsigned int i=0; i<m_maxThreads; i++){
			m_freeThreads.push(i);
		}
	}

	void ThreadManager::shutdown(){
		// TODO: shutdown active threads!
	}

	int ThreadManager::launchWorkPackage(boost::shared_ptr<boost::function<void(void)>> func, boost::shared_ptr<boost::function<void(int)>> callback = boost::shared_ptr<boost::function<void(int)>>()){
		if(!func)
			return -1;
		
		workPackage package;
		package.callback = callback;
		package.func = func;

		m_manageMutex.lock();
		{
			package.workId = m_workId++;

			if(!m_freeThreads.empty()){
				int threadId = m_freeThreads.front();
				m_freeThreads.pop();

				if(m_threadPool[threadId])
					m_threadPool[threadId]->join();
				m_threadPool[threadId] = boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(&ThreadManager::threadFunction,threadId,package)));
			}
			else{
				m_workQueue.push(package);
			}
		}
		m_manageMutex.unlock();

		return package.workId;
	}

	int ThreadManager::queueWorkPackage(boost::shared_ptr<boost::function<void(void)>> func, boost::shared_ptr<boost::function<void(int)>> callback){
		if(!func)
			return -1;
		
		workPackage package;
		package.callback = callback;
		package.func = func;

		m_manageMutex.lock();
			package.workId = m_workId++;
			m_workQueue.push(package);
		m_manageMutex.unlock();

		return package.workId;
	}

	void ThreadManager::executeWorkQueue(){
		m_manageMutex.lock();
		{
			while(!m_freeThreads.empty() && !m_workQueue.empty()){
				int threadId = m_freeThreads.front();
				workPackage package = m_workQueue.front();
				m_freeThreads.pop();
				m_workQueue.pop();

				if(m_threadPool[threadId])
					m_threadPool[threadId]->join();
				m_threadPool[threadId] = boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(&ThreadManager::threadFunction,threadId,package)));
			}
		}
		m_manageMutex.unlock();
	}

	void ThreadManager::threadFunction(int id, workPackage package){
		// Run work package
		(*package.func)();
		if((*package.callback))
			(*package.callback)(package.workId);

		// Check work queue
		bool recur = false;
		workPackage newPackage;

		do{
			m_manageMutex.lock();
			{
				if(m_workQueue.empty()){
					m_freeThreads.push(id);
					recur = false;
				}
				else{
					recur = true;
					newPackage = m_workQueue.front();
					m_workQueue.pop();
				}
			}
			m_manageMutex.unlock();

			if(recur){
				// Run work package
				(*newPackage.func)();
				if((*newPackage.callback))
					(*newPackage.callback)(newPackage.workId);
			}
		}while(recur);
	}
}