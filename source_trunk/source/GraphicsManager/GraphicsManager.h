#pragma once

namespace DataMiner{
	#define thread_group_size 64

	class GraphicsManager : public boost::enable_shared_from_this<GraphicsManager>{
	public:
		GraphicsManager():m_availableMemory(0),m_totalMemory(0),m_allocatedMemory(0),m_contextCounter(0){
			std::string numGPUs = ConfigManager::getSetting("pThreads");
			if(numGPUs.compare("All") != 0){
				std::stringstream ss(numGPUs);
				int nr;
				ss >> nr;
				if(!ss.fail() && nr > 0)
					m_numDevices = nr;
			}
			else
				m_numDevices = 0;
		}

		enum ResourceType {GRT_CBUFFER = 0, GRT_RESOURCE, GRT_WBUFFER};
		enum ResourceFormat {GRF_REAL = 0, GRF_INT, GRF_BOOL, GRF_STRING, GRF_UINT, GRF_ULONG};
		enum BufferAccess {GBA_WRITABLE = 0, GBA_STATIC, GBA_CONSTANT};

		struct bufferFlags{
			DWORD dxUsageFlags;
			DWORD dxAccessFlags;
			DWORD openCLFlags;
		};

		virtual void initialize() = 0;

		virtual int createDeviceContext(int devId) = 0;
		virtual void destroyDeviceContext(int id) = 0;

		virtual void launchComputation(int devId, int x, int y, int z) = 0;
		virtual void launchComputation(int x, int y, int z, bool split = true) = 0;

		virtual void syncDevice(int devId) = 0;
		virtual void syncDevice() = 0;

		virtual void setGPUProgram(int devId, int pid) = 0;
		virtual void setGPUProgram(int pid) = 0;

		virtual void setGPUBuffer(int devId, std::vector<int> resId, std::vector<ResourceType> rType) = 0;
		virtual void setGPUBuffer(std::vector<int> resId, std::vector<ResourceType> rType) = 0;
		virtual int getNumDevices() = 0;

		virtual void copyFromGPU(int devId, int resId, void* buff, unsigned int byteSize) = 0;
		virtual void copyFromGPU(int resId, void* buff, unsigned int byteSize, unsigned int elements, bool split = true) = 0;
		virtual void copyToGPU(int devId, int resId, void* buff, unsigned int byteSize) = 0;
		virtual void copyToGPU(int resId, void* buff, unsigned int byteSize, unsigned int elements, bool split = true) = 0;

		virtual int createBuffer(int devId, bufferFlags flags, UINT numElements, void* initData, unsigned int byteSize, ResourceType bType, ResourceFormat bForm = GRF_REAL, GraphicsFunctionPtr func = GraphicsFunctionPtr()) = 0;
		virtual int createBuffer(bufferFlags flags, UINT numElements, void* initData, unsigned int byteSize, ResourceType bType, ResourceFormat bForm = GRF_REAL, GraphicsFunctionPtr func = GraphicsFunctionPtr(), bool split = true) = 0;
		virtual int createGPUProgram(int devId, std::string filename, GraphicsFunctionPtr func = GraphicsFunctionPtr()) = 0;
		virtual int createGPUProgram(std::string filename, GraphicsFunctionPtr func = GraphicsFunctionPtr()) = 0;
		virtual void deleteBuffer(int devId, int resId) = 0;
		virtual void deleteBuffer(int resId) = 0;
		virtual void deleteGPUProgram(int devId, int pid) = 0;
		virtual void deleteGPUProgram(int pid) = 0;

		virtual int markGPUTime(int devId) = 0;
		virtual int markGPUTime() = 0;
		virtual float getGPUTime(int devId, int timer) = 0;
		virtual float getGPUTime(int timer) = 0;

		size_t getAvailableMemory(int devId) { return m_availableMemory[devId]; }
		size_t getTotalMemory(int devId) { return m_totalMemory[devId]; }
		size_t getAllocatedMemory(int devId) { return m_allocatedMemory[devId]; }
	protected:
		std::vector<size_t> m_availableMemory;
		std::vector<size_t> m_totalMemory;
		std::vector<size_t> m_allocatedMemory;
		unsigned int m_contextCounter;
		unsigned int m_numDevices;
	};
}

typedef boost::shared_ptr<DataMiner::GraphicsManager> GraphicsManagerPtr;