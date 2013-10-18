#pragma once
#include "GraphicsManager.h"
#include "Value.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace DataMiner{
	class CUDAManager : public GraphicsManager{
	public:
		CUDAManager();
		~CUDAManager();

		void initialize();

		int createDeviceContext(int devId);
		void destroyDeviceContext(int id);

		void launchComputation(int devId, int x, int y, int z);
		void launchComputation(int x, int y, int z, bool split = true);
		void syncDevice(int devId);
		void syncDevice();

		void setGPUProgram(int devId, int pid);
		void setGPUProgram(int pid);
		void setGPUBuffer(int devId, std::vector<int> resId, std::vector<ResourceType> rType);
		void setGPUBuffer(std::vector<int> resId, std::vector<ResourceType> rType);
		int getNumDevices();

		void copyFromGPU(int devId, int resId, void* buff, unsigned int byteSize);
		void copyFromGPU(int resId, void* buff, unsigned int byteSize, unsigned int elements, bool split = true);
		void copyToGPU(int devId, int resId, void* buff, unsigned int byteSize);
		void copyToGPU(int resId, void* buff, unsigned int byteSize, unsigned int elements, bool split = true);

		int createBuffer(int devId, bufferFlags flags, UINT numElements, void* initData, unsigned int byteSize, ResourceType bType, ResourceFormat bForm = GRF_REAL, GraphicsFunctionPtr func = GraphicsFunctionPtr());
		int createBuffer(bufferFlags flags, UINT numElements, void* initData, unsigned int byteSize, ResourceType bType, ResourceFormat bForm = GRF_REAL, GraphicsFunctionPtr func = GraphicsFunctionPtr(), bool split = true);
		int createGPUProgram(int devId, std::string filename, GraphicsFunctionPtr func = GraphicsFunctionPtr());
		int createGPUProgram(std::string filename, GraphicsFunctionPtr func = GraphicsFunctionPtr());
		void deleteBuffer(int devId, int resId);
		void deleteBuffer(int resId);
		void deleteGPUProgram(int devId, int pid);
		void deleteGPUProgram(int pid);

		int markGPUTime(int devId);
		int markGPUTime();
		float getGPUTime(int devId, int timer);
		float getGPUTime(int timer);
	
		void* getResource(int devId, int resId);
	private:
		void updateAvailableMemory(int devId);

		struct cudaDeviceContext{
			cudaDeviceContext():m_resourceIdCounter(0){}

			std::map<int,std::pair<Value::v_precision*,size_t>> m_cudaReal;
			std::map<int,std::pair<unsigned int*,size_t>> m_cudaUInt;
			std::map<int,std::pair<int*,size_t>> m_cudaInt;
			std::map<int,std::pair<bool*,size_t>> m_cudaBool;
			std::map<int,std::pair<unsigned int*,size_t>> m_cudaULong;

			std::map<int,ResourceFormat> m_formatMap;
			std::vector<int> m_setResourceIds;

			std::map<int,cudaEvent_t> m_cudaEvents;

			CUcontext m_context;

			GraphicsFunctionPtr m_currentFunction;
			std::map<int,GraphicsFunctionPtr> m_cudaFunctions;

			std::set<int> m_excludeBuffers;
			int m_resourceIdCounter;
		};
		
		std::map<int,cudaDeviceContext> m_contexts;
		bool m_autoDivideWork;
		MutexPtr m_contextMutex;
	};
}