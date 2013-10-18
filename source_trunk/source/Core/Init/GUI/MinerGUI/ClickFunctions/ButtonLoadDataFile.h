#pragma once
#include "Runnable.h"
#include "MainFramework.h"
#include "StartAlgorithmMessage.h"

namespace DataMiner{
	class ButtonLoadDataFile : public Runnable{
	public:
		ButtonLoadDataFile():m_loadThread(boost::shared_ptr<boost::thread>()){}

		void run(void* params){
			mainFramework = (MainFramework*)params;
			mainFramework->getMinerGUI()->disableAllButStop();

			if(m_loadThread)
				m_loadThread->join();

			m_loadThread = boost::shared_ptr<boost::thread>(new boost::thread(boost::function<void(void)>(boost::bind(&ButtonLoadDataFile::loadFile,this))));
		}
	private:
		void loadFile(){
			GUIManagerPtr guiPtr = mainFramework->getGuiPtr();
			std::stringstream ss;
			ss << guiPtr->getEditText(IDC_EDIT_FILEPATH);
			DataDocumentPtr doc = mainFramework->getResourcePtr()->getDocumentResource(ss.str());
	
			std::wstringstream wss;
			if(doc){
				wss << ss.str().c_str() << "\r\nAttributes: " << doc->getNumAttributes() << "\r\nInstances: " << doc->getNumInstances() << "\r\n"
					<< "Missing values: " << doc->getNumMissing();
			}
			else{
				wss << "Specified data document not found.\r\n";
			}
	
			guiPtr->postDebugMessage(wss.str());

			mainFramework->getMinerGUI()->enableAllButStop();
		}

		MainFramework *mainFramework;
		boost::shared_ptr<boost::thread> m_loadThread;
	};
}