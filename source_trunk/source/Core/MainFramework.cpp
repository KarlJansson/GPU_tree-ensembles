#include "stdafx.h"
#include "MainFramework.h"
#include "StartAlgorithmMessageHandler.h"
#include "StopAlgorithmMessage.h"
#include "GUIManager.h"
#include "MinerGUI.h"

#include "DirectXManager.h"
#include "CUDAManager.h"
#include "OpenCLManager.h"

namespace DataMiner{
	MainFramework::MainFramework(GraphicsManagerPtr gmgr, GUIManagerPtr guimgr):m_endParsing(false){
		m_rManager = ResourceManagerPtr(new ResourceManager);
		m_gManager = gmgr;
		m_guiManager = guimgr;
		m_messageStackMutex = MutexPtr(new boost::mutex);
		m_parseCondition = ConditionPtr(new boost::condition_variable);
		m_minerGUI = MinerGUIPtr(new MinerGUI(m_guiManager));

		//m_GPUManagers["DirectX"] = GraphicsManagerPtr(new DirectXManager);
		m_GPUManagers["CUDA"] = GraphicsManagerPtr(new CUDAManager);
		//m_GPUManagers["OpenCL"] = GraphicsManagerPtr(new OpenCLManager);
	}

	MainFramework::~MainFramework(){
		m_messageHandlers.clear();
		m_endParsing = true;
		m_parseCondition->notify_all();
		m_messageParser->join();

		std::map<std::string,boost::shared_ptr<boost::thread>>::iterator itr = m_runningThreads.begin();
		while(itr != m_runningThreads.end()){
			itr->second->interrupt();
			itr->second->join();
			itr++;
		}
		m_runningThreads.clear();
	}

	void MainFramework::run(){
		if(!m_messageParser)
			m_messageParser = ThreadPtr(new boost::thread(&MainFramework::parseMessages,this));
	}

	void MainFramework::postMessage(FrameworkMessagePtr message){
		m_messageStackMutex->lock();
		m_messageStack.push_front(message);
		m_messageStackMutex->unlock();

		m_parseCondition->notify_one();
	}

	void MainFramework::initHandlers(){
		// Add messageHandlers
		m_messageHandlers["StartAlgorithm"] = MessageHandlerPtr(new StartAlgorithmMessageHandler(m_gManager));
	}

	void MainFramework::parseMessages(){
		//m_GPUManagers["DirectX"]->initialize();
		m_GPUManagers["CUDA"]->initialize();
		//m_GPUManagers["OpenCL"]->initialize();
		initHandlers();
		boost::mutex mut;
		boost::unique_lock<boost::mutex> lock(mut);
		while(!m_endParsing){
			std::list<FrameworkMessagePtr> stack;
			m_messageStackMutex->lock();
			stack = m_messageStack;
			m_messageStack.clear();
			m_messageStackMutex->unlock();

			std::map<std::string,MessageHandlerPtr>::iterator handlerItr;
			while(!stack.empty()){
				if((handlerItr = m_messageHandlers.find(stack.front()->getMessage())) != m_messageHandlers.end()){
					std::string gpuapi = m_guiManager->getEditText(IDC_ALGORITHM_GPUAPI);
					if(gpuapi.compare("DirectX") == 0){
						m_gManager = m_GPUManagers["DirectX"];
					}
					else if(gpuapi.compare("CUDA") == 0){
						m_gManager = m_GPUManagers["CUDA"];
					}
					else if(gpuapi.compare("OpenCL") == 0){
						m_gManager = m_GPUManagers["OpenCL"];
					}

					IDataPackPtr dataPack = stack.front()->getDataPack();
					dataPack->m_gfxMgr = m_gManager;
					dataPack->m_recMgr = m_rManager;
					dataPack->m_gui = m_guiManager;
					dataPack->m_minerGUI = m_minerGUI;

					m_runningThreads[stack.front()->getMessage()] = boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(&MainFramework::runThread,this,stack.front(),dataPack)));
					if(SetThreadPriority(m_runningThreads[stack.front()->getMessage()]->native_handle(),THREAD_PRIORITY_BELOW_NORMAL) == 0){
						assert(0);
					}
				}
				else if(stack.front()->getMessage().compare("StopAlgorithm") == 0){
					m_guiManager->getWindow(IDC_BUTTON_STOP)->disable();
					m_messageHandlers["StartAlgorithm"]->stop();

					boost::shared_ptr<boost::thread> thread = m_runningThreads["StartAlgorithm"];
					if(thread){
						thread->join();
					}

					m_minerGUI->enableAllButStop();
					m_guiManager->setText(IDC_STATIC_INFOTEXT,L"");
					m_guiManager->setText(IDC_STATIC_DEBUG,L"");

					stack.front()->waitOnMessage();
				}
				else{
					TRACE_DEBUG("Message without registered handler recieved.");
					assert(0);
				}

				stack.pop_front();
			}

			m_parseCondition->wait(lock);
		}
	}

	void MainFramework::addHandler(std::string messageId, MessageHandlerPtr handler){
		m_messageHandlers[messageId] = handler;
	}

	void MainFramework::runThread(FrameworkMessagePtr message, IDataPackPtr dataPack){
		std::map<std::string,MessageHandlerPtr>::iterator handlerItr;
		handlerItr = m_messageHandlers.find(message->getMessage());
		handlerItr->second->handle(dataPack);
		message->waitOnMessage();
		m_runningThreads.erase(message->getMessage());
	}
}