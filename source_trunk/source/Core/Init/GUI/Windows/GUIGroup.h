#pragma once
#include "GUIWindow.h"

namespace DataMiner{
	class GUIGroup{
	public:
		GUIGroup(GUIManagerPtr gui, int x, int y):
		  m_guiManager(gui),m_posX(x),m_posY(y){}
		~GUIGroup();

		void addWindow(int id);

		void setPosition(int x, int y);

		void hide();
		void show();
		void disable();
		void enable();
	private:
		std::vector<int> m_windows;
		GUIManagerPtr m_guiManager;

		int m_posX,
			m_posY;
	};
}