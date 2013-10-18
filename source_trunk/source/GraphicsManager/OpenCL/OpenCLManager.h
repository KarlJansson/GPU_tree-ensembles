#pragma once
#include "GraphicsManager.h"
#include <CL/opencl.h>

namespace DataMiner{
	class OpenCLManager : public GraphicsManager{
	public:
		OpenCLManager();
		~OpenCLManager();

		void initialize();

		int createDeviceContext(int devId){return 0;}
		void destroyDeviceContext(int id){}

		void launchComputation(int devId, int x, int y, int z);
		void launchComputation(int x, int y, int z, bool split = true){launchComputation(0,x,y,z);}
		void syncDevice(int devId){}
		void syncDevice(){syncDevice(0);}

		void setGPUProgram(int devId, int pid);
		void setGPUProgram(int pid){setGPUProgram(0,pid);}
		void setGPUBuffer(int devId, std::vector<int> resId, std::vector<ResourceType> rType);
		void setGPUBuffer(std::vector<int> resId, std::vector<ResourceType> rType) {setGPUBuffer(0,resId,rType);}
		
		int getNumDevices();

		void copyFromGPU(int devId, int resId, void* buff, unsigned int byteSize);
		void copyFromGPU(int resId, void* buff, unsigned int byteSize, unsigned int elements, bool split = true){copyFromGPU(0,resId,buff,byteSize);}
		void copyToGPU(int devId, int resId, void* buff, unsigned int byteSize);
		void copyToGPU(int resId, void* buff, unsigned int byteSize, unsigned int elements, bool split = true){copyToGPU(0,resId,buff,byteSize);}

		int createBuffer(int devId, bufferFlags flags, UINT numElements, void* initData, unsigned int byteSize, ResourceType bType, ResourceFormat bForm = GRF_REAL, GraphicsFunctionPtr func = GraphicsFunctionPtr());
		int createBuffer(bufferFlags flags, UINT numElements, void* initData, unsigned int byteSize, ResourceType bType, ResourceFormat bForm = GRF_REAL, GraphicsFunctionPtr func = GraphicsFunctionPtr(), bool split = true){return createBuffer(0,flags,numElements,initData,byteSize,bType,bForm,func);}
		int createGPUProgram(int devId, std::string filename, GraphicsFunctionPtr func = GraphicsFunctionPtr());
		int createGPUProgram(std::string filename, GraphicsFunctionPtr func = GraphicsFunctionPtr()){return createGPUProgram(0,filename,func);}
		void deleteBuffer(int devId, int resId);
		void deleteBuffer(int resId){deleteBuffer(0,resId);}
		void deleteGPUProgram(int devId, int pid);
		void deleteGPUProgram(int pid){deleteGPUProgram(0,pid);}

		int markGPUTime(int devId){return 0;}
		int markGPUTime(){return markGPUTime(0);}
		float getGPUTime(int devId, int timer){return 0;}
		float getGPUTime(int timer){return getGPUTime(0,timer);}
	private:
		void checkError(cl_int error);

		cl_context m_context;
		cl_command_queue m_cQueue;

		std::map<int,cl_program> m_programs;
		std::map<int,cl_kernel> m_kernels;
		std::map<int,cl_mem> m_buffers;

		int m_currentKernel;

		int m_resourceIdCount;
	};
}