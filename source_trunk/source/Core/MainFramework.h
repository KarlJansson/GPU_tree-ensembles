#pragma once
#include "GraphicsManager.h"
#include "ResourceManager.h"
#include "FrameworkMessage.h"
#include "MessageHandler.h"
#include "IAlgorithm.h"
#include "GUIManager.h"

namespace DataMiner{
	class MainFramework{
	public:
		MainFramework(GraphicsManagerPtr gmgr, GUIManagerPtr guimgr);
		~MainFramework();

		void run();
		void postMessage(FrameworkMessagePtr message);

		GUIManagerPtr getGuiPtr() {return m_guiManager;}
		MinerGUIPtr getMinerGUI() {return m_minerGUI; }
		ResourceManagerPtr getResourcePtr() {return m_rManager;}
		GraphicsManagerPtr getGraphicsPtr() {return m_gManager;}
	private:
		void initHandlers();
		void parseMessages();
		void addHandler(std::string messageId, MessageHandlerPtr handler);
		void runThread(FrameworkMessagePtr message, IDataPackPtr dataPack);

		bool m_endParsing;
		std::list<FrameworkMessagePtr> m_messageStack;

		std::map<std::string,GraphicsManagerPtr> m_GPUManagers;
		
		GraphicsManagerPtr m_gManager;
		ResourceManagerPtr m_rManager;
		GUIManagerPtr m_guiManager;
		MinerGUIPtr m_minerGUI;

		std::map<std::string,MessageHandlerPtr> m_messageHandlers;

		MutexPtr m_messageStackMutex;
		ThreadPtr m_messageParser;
		ConditionPtr m_parseCondition;

		std::map<std::string,boost::shared_ptr<boost::thread>> m_runningThreads;
	};
}

typedef boost::shared_ptr<DataMiner::MainFramework> MainFrameworkPtr;