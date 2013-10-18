#pragma once
#include "GraphicsManager.h"
#include "ResourceManager.h"
#include "GUIManager.h"
#include "MinerGUI.h"

namespace DataMiner{
	class IDataPack{
	public:
		GraphicsManagerPtr m_gfxMgr;
		ResourceManagerPtr m_recMgr;
		GUIManagerPtr m_gui;
		MinerGUIPtr m_minerGUI;

		boost::shared_ptr<boost::function<void (boost::shared_ptr<IAlgorithm> a)>> m_callBack;
	};
}

typedef boost::shared_ptr<DataMiner::IDataPack> IDataPackPtr;