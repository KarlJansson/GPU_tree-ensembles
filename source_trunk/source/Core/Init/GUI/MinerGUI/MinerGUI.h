#pragma once
#include "WindowIds.h"

namespace DataMiner{
	class MinerGUI{
	public:
		MinerGUI(GUIManagerPtr guiManager);
		~MinerGUI();

		void enableAllButStop();
		void disableAllButStop();
	private:
		GUIManagerPtr m_guiManager;

		// Groups
		GUIGroupPtr m_mainGroup,
					m_svmGroup,
					m_randForestGroup,
					m_algorithmGroup,
					m_evalMethodGroup,
					m_schemeGroup;
	};
}