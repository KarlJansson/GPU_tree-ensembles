#include "stdafx.h"
#include "GUIManagerCmd.h"

namespace DataMiner{
	GUIManagerCmd::GUIManagerCmd(){
		
	}

	void GUIManagerCmd::setText(int id, std::wstring message){
		if(id == IDC_STATIC_INFOTEXT){
			std::wstring str = m_settings[IDC_EDIT_FILEPATH];

			int pos;
			if((pos = str.find_last_of(L'\\')) == std::wstring::npos){
				if((pos = str.find_last_of(L'/')) == std::wstring::npos){
					pos = 0;
				}
			}

			str = str.substr(pos+1,str.find_last_of('.')-(pos+1));
			ConfigManager::writeToFile(std::string(message.begin(),message.end()),std::string(str.begin(),str.end())+"_output.txt",std::ios_base::trunc);
		}
		m_settings[id] = message;
	}

	std::string GUIManagerCmd::getEditText(int id){
		return std::string(m_settings[id].begin(),m_settings[id].end());
	}

	void GUIManagerCmd::postDebugMessage(std::wstring message){
		//std::string msg(message.begin(),message.end());
		//printf(std::string(msg+"\n").c_str());
	}

	unsigned int GUIManagerCmd::addWindow(int id, std::wstring type, DWORD style1, DWORD style2, int x, int y, int width, int height, std::wstring text){
		m_windows[id] = GUIWindowPtr(new GUIWindow(id,type,0,style1,style2,x,y,width,height,0,text));
		return id;
	}
}