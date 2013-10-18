#pragma once
#include "GUIWindow.h"

namespace DataMiner{
	class GUIManager{
	public:
		GUIManager();

		virtual void setText(int id, std::wstring message) = 0;
		virtual std::string getEditText(int id) = 0;
		virtual void postDebugMessage(std::wstring message) = 0;

		virtual void setProgressBar(int id, unsigned int max, unsigned int progress) = 0;
		virtual unsigned int addWindow(int id, std::wstring type, DWORD style1, DWORD style2, int x, int y, int width, int height, std::wstring text) = 0;

		unsigned int addGroup(int id, GUIGroupPtr group) { m_groups[id] = group; return id; }
		void removeWindow(unsigned int id) { m_windows.erase(id); }
		GUIGroupPtr getGroup(int id) { return m_groups[id]; }
		GUIWindowPtr getWindow(int id) { return m_windows[id]; }

		template<typename T>
		T getSetting(int id){
			T result;
			std::string setting = getEditText(id);
			try{
				result = boost::lexical_cast<T,std::string>(setting);
			}
			catch(...){
				assert(0);
				TRACE_DEBUG("Error: in GuiManager getSettings(int id)\n");
			}
			return result;
		}

		virtual void processWindowsMessage(int wmid, void* params) = 0;
	protected:
		std::map<unsigned int, GUIWindowPtr> m_windows;
		std::map<unsigned int, GUIGroupPtr> m_groups;

		std::wstring m_debugString;
	};
}