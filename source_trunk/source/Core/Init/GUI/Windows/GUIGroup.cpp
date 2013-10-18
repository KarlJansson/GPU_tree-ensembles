#include "stdafx.h"
#include "GUIGroup.h"
#include "GUIManager.h"

namespace DataMiner{
	GUIGroup::~GUIGroup(){
	
	}

	void GUIGroup::addWindow(int window){
		m_windows.push_back(window);
		GUIWindowPtr windowPtr = m_guiManager->getWindow(window);
		if(!windowPtr)
			return;
		std::pair<int,int> pos = windowPtr->getPosition();

		windowPtr->setPosition(m_posX+pos.first,m_posY+pos.second);
	}

	void GUIGroup::setPosition(int x, int y){
		m_posX = x;
		m_posY = y;

		GUIWindowPtr windowPtr;
		std::pair<int,int> pos;
		for(unsigned int i=0; i<m_windows.size(); i++){
			windowPtr = m_guiManager->getWindow(m_windows[i]);
			pos = windowPtr->getPosition();
			windowPtr->setPosition(m_posX+pos.first,m_posY+pos.second);
		}
	}
	
	void GUIGroup::hide(){
		for(unsigned int i=0; i<m_windows.size(); i++){
			m_guiManager->getWindow(m_windows[i])->hide();
		}
	}

	void GUIGroup::show(){
		for(unsigned int i=0; i<m_windows.size(); i++){
			m_guiManager->getWindow(m_windows[i])->show();
		}
	}

	void GUIGroup::disable(){
		for(unsigned int i=0; i<m_windows.size(); i++){
			m_guiManager->getWindow(m_windows[i])->disable();
		}
	}

	void GUIGroup::enable(){
		for(unsigned int i=0; i<m_windows.size(); i++){
			m_guiManager->getWindow(m_windows[i])->enable();
		}
	}
}