#include "stdafx.h"
#include "CUDAManager.h"
#include "GraphicsFunction.h"

namespace DataMiner{
	CUDAManager::CUDAManager(){
		m_autoDivideWork = true;
		m_contextMutex = MutexPtr(new boost::mutex);
	}

	CUDAManager::~CUDAManager(){
		std::map<int,cudaDeviceContext>::iterator itr = m_contexts.begin();
		while(itr != m_contexts.end()){
			cuCtxDestroy(itr->second.m_context);
			itr++;
		}
	}

	void CUDAManager::initialize(){
		int nNoOfGPUs;

		cuInit(0);
		cuDeviceGetCount( &nNoOfGPUs );
		m_numDevices = (m_numDevices < nNoOfGPUs && m_numDevices > 0) ? m_numDevices : nNoOfGPUs;
	}

	int CUDAManager::createDeviceContext(int devId){
		m_contextMutex->lock();
			int id = m_contextCounter++;

			CUresult result;
			CUdevice device;
			CUcontext context;

			m_contexts[id] = cudaDeviceContext();
			cuDeviceGet( &device, devId );
			result = cuCtxCreate( &m_contexts[id].m_context, 0, device );
			assert(result == cudaSuccess);

			result = cuCtxAttach(&m_contexts[id].m_context,0);
			assert(result == cudaSuccess);

			m_availableMemory.push_back(0);
			m_allocatedMemory.push_back(0);
			m_totalMemory.push_back(0);
			updateAvailableMemory(id);
		m_contextMutex->unlock();
		return id;
	}

	void CUDAManager::destroyDeviceContext(int id){
		m_contextMutex->lock();
			cuCtxDetach(m_contexts[id].m_context);
			cuCtxDestroy(m_contexts[id].m_context);
			m_contexts.erase(id);
		m_contextMutex->unlock();
	}

	void CUDAManager::launchComputation(int devId, int x, int y, int z){
		cudaDeviceContext &ctx = m_contexts[devId];

		cuCtxPushCurrent(ctx.m_context);
		if(ctx.m_currentFunction){
			ctx.m_currentFunction->run(devId,ctx.m_setResourceIds,shared_from_this(),x,y,z);
			cudaError_t error = cudaGetLastError();
			assert(error == cudaSuccess);
		}
		cuCtxPopCurrent(&ctx.m_context);
	}

	void CUDAManager::launchComputation(int x, int y, int z, bool split){
		if(split){
			unsigned int totalBlocks = x/thread_group_size;
			if(totalBlocks == 0)
				totalBlocks = 1;
			unsigned int blocksPerGPU = totalBlocks/m_contexts.size();
			unsigned int restBlocks = totalBlocks % m_contexts.size();

			for(unsigned int i=0; i<m_contexts.size(); ++i){
				if(i == m_contexts.size()-1)
					blocksPerGPU += restBlocks;
				launchComputation(i,thread_group_size*blocksPerGPU,y,z);		
			}
		}
		else{
			for(unsigned int i=0; i<m_contexts.size(); ++i){
				launchComputation(i,x,y,z);
			}
		}
	}

	void CUDAManager::syncDevice(int devId){
		cudaDeviceContext &ctx = m_contexts[devId];

		cuCtxPushCurrent(ctx.m_context);
		cudaDeviceSynchronize();
		cuCtxPopCurrent(&ctx.m_context);
	}

	void CUDAManager::syncDevice(){
		for(unsigned int i=0; i<m_contexts.size(); ++i){
			syncDevice(i);
		}
	}

	void CUDAManager::setGPUProgram(int devId, int pid){
		cudaDeviceContext &ctx = m_contexts[devId];
		ctx.m_currentFunction = ctx.m_cudaFunctions[pid];
	}

	void CUDAManager::setGPUProgram(int pid){
		for(unsigned int i=0; i<m_contexts.size(); ++i){
			setGPUProgram(i,pid);
		}
	}
	
	void CUDAManager::setGPUBuffer(int devId, std::vector<int> resId, std::vector<ResourceType> rType){
		cudaDeviceContext &ctx = m_contexts[devId];
		ctx.m_setResourceIds.clear();
		for(unsigned int i=0; i<resId.size(); i++){
			if(ctx.m_excludeBuffers.find(resId[i]) == ctx.m_excludeBuffers.end())
				ctx.m_setResourceIds.push_back(resId[i]);
		}
	}

	void CUDAManager::setGPUBuffer(std::vector<int> resId, std::vector<ResourceType> rType){
		for(unsigned int i=0; i<m_contexts.size(); ++i){
			setGPUBuffer(i,resId,rType);
		}
	}

	int CUDAManager::getNumDevices(){
		return m_numDevices;
	}

	void CUDAManager::copyFromGPU(int devId, int resId, void* buff, unsigned int byteSize){
		cudaDeviceContext &ctx = m_contexts[devId];
		cuCtxPushCurrent(ctx.m_context);
		cudaError_t error = cudaMemcpy(buff,getResource(devId,resId),byteSize,cudaMemcpyDeviceToHost);
		cuCtxPopCurrent(&ctx.m_context);
	}

	void CUDAManager::copyFromGPU(int resId, void* buff, unsigned int byteSize, unsigned int elements, bool split){
		if(split){
			unsigned int bytePerElement = byteSize/elements;
			unsigned int elementsPerGPU = elements/m_contexts.size();
			unsigned int restElements = elements % m_contexts.size();
			char *outData = (char*)buff;

			for(unsigned int i=0; i<m_contexts.size(); ++i){
				if(i == m_contexts.size()-1)
					elementsPerGPU += restElements;
				copyFromGPU(i,resId,outData,bytePerElement*elementsPerGPU);
				outData += elementsPerGPU*bytePerElement;
			}
		}
		else{
			for(unsigned int i=0; i<m_contexts.size(); ++i){
				copyFromGPU(i,resId,buff,byteSize);
			}
		}
	}

	void CUDAManager::copyToGPU(int devId, int resId, void* buff, unsigned int byteSize){
		cudaDeviceContext &ctx = m_contexts[devId];
		cuCtxPushCurrent(ctx.m_context);
		if(ctx.m_cudaFunctions.find(resId) != ctx.m_cudaFunctions.end())
			ctx.m_cudaFunctions[resId]->run(devId,std::vector<int>(),shared_from_this(),1,1,1,buff);
		else
			cudaMemcpy(getResource(devId,resId),buff,byteSize,cudaMemcpyHostToDevice);
		cuCtxPopCurrent(&ctx.m_context);
	}

	void CUDAManager::copyToGPU(int resId, void* buff, unsigned int byteSize, unsigned int elements, bool split){
		if(split){
			unsigned int bytePerElement = byteSize/elements;
			unsigned int elementsPerGPU = elements/m_contexts.size();
			unsigned int restElements = elements % m_contexts.size();
			char *inData = (char*)buff;

			for(unsigned int i=0; i<m_contexts.size(); ++i){
				if(i == m_contexts.size()-1)
					elementsPerGPU += restElements;
				copyToGPU(i,resId,inData,bytePerElement*elementsPerGPU);
				inData += elementsPerGPU*bytePerElement;
			}
		}
		else{
			for(unsigned int i=0; i<m_contexts.size(); ++i){
				copyToGPU(i,resId,buff,byteSize);
			}
		}
	}

	int CUDAManager::createBuffer(int devId, bufferFlags flags, UINT numElements, void* initData, unsigned int byteSize, ResourceType bType, ResourceFormat bForm, GraphicsFunctionPtr func){
		cudaDeviceContext &ctx = m_contexts[devId];
		cuCtxPushCurrent(ctx.m_context);
		cudaError_t error = cudaSuccess;
		if(bType == GRT_CBUFFER){
			ctx.m_cudaFunctions[ctx.m_resourceIdCounter++] = func;
			ctx.m_excludeBuffers.insert(ctx.m_resourceIdCounter-1);
		}
		else{
			switch(bForm){
			case GRF_REAL:
				error = cudaMalloc<Value::v_precision>(&ctx.m_cudaReal[ctx.m_resourceIdCounter++].first,byteSize);
				ctx.m_cudaReal[ctx.m_resourceIdCounter-1].second = byteSize;
				ctx.m_formatMap[ctx.m_resourceIdCounter-1] = GRF_REAL;
				break;
			case GRF_INT:
				error = cudaMalloc<int>(&ctx.m_cudaInt[ctx.m_resourceIdCounter++].first,byteSize);
				ctx.m_cudaInt[ctx.m_resourceIdCounter-1].second = byteSize;
				ctx.m_formatMap[ctx.m_resourceIdCounter-1] = GRF_INT;
				break;
			case GRF_UINT:
				error = cudaMalloc<unsigned int>(&ctx.m_cudaUInt[ctx.m_resourceIdCounter++].first,byteSize);
				ctx.m_cudaUInt[ctx.m_resourceIdCounter-1].second = byteSize;
				ctx.m_formatMap[ctx.m_resourceIdCounter-1] = GRF_UINT;
				break;
			case GRF_BOOL:
				error = cudaMalloc<bool>(&ctx.m_cudaBool[ctx.m_resourceIdCounter++].first,byteSize);
				ctx.m_cudaBool[ctx.m_resourceIdCounter-1].second = byteSize;
				ctx.m_formatMap[ctx.m_resourceIdCounter-1] = GRF_BOOL;
				break;
			case GRF_ULONG:
				error = cudaMalloc<unsigned int>(&ctx.m_cudaULong[ctx.m_resourceIdCounter++].first,byteSize);
				ctx.m_cudaULong[ctx.m_resourceIdCounter-1].second = byteSize;
				ctx.m_formatMap[ctx.m_resourceIdCounter-1] = GRF_ULONG;
				break;
			}
		}
		cuCtxPopCurrent(&ctx.m_context);

		//m_allocatedMemory[devId] += byteSize;

		assert(error == cudaSuccess);

		if(initData)
			copyToGPU(devId,ctx.m_resourceIdCounter-1,initData,byteSize);

		updateAvailableMemory(devId);
		return ctx.m_resourceIdCounter-1;
	}

	int CUDAManager::createBuffer(bufferFlags flags, UINT numElements, void* initData, unsigned int byteSize, ResourceType bType, ResourceFormat bForm, GraphicsFunctionPtr func, bool split){
		unsigned int id;
		if(split){
			unsigned int bytePerElement = byteSize/numElements;
			unsigned int elementsPerGPU = numElements/m_contexts.size();
			unsigned int restElements = numElements % m_contexts.size();
			char *iData = (char*)initData;

			for(unsigned int i=0; i<m_contexts.size(); ++i){
				if(i == m_contexts.size()-1)
					elementsPerGPU += restElements;
				id = createBuffer(i,flags,elementsPerGPU,iData,bytePerElement*elementsPerGPU,bType,bForm,func);
				cudaError_t error = cudaGetLastError();
				assert(error == cudaSuccess);
				if(iData)
					iData += elementsPerGPU*bytePerElement;
			}
		}
		else{
			for(unsigned int i=0; i<m_contexts.size(); ++i){
				id = createBuffer(i,flags,numElements,initData,byteSize,bType,bForm,func);
			}
		}
		return id;
	}
	
	int CUDAManager::createGPUProgram(int devId, std::string filename, GraphicsFunctionPtr func){
		cudaDeviceContext &ctx = m_contexts[devId];
		ctx.m_cudaFunctions[ctx.m_resourceIdCounter++] = func;

		return ctx.m_resourceIdCounter-1;
	}

	int CUDAManager::createGPUProgram(std::string filename, GraphicsFunctionPtr func){
		int result = 0;
		for(unsigned int i=0; i<m_contexts.size(); ++i){
			result = createGPUProgram(i,filename,func);
		}
		return result;
	}

	void CUDAManager::deleteBuffer(int devId, int resId){
		cudaDeviceContext &ctx = m_contexts[devId];
		cuCtxPushCurrent(ctx.m_context);
		switch(ctx.m_formatMap[resId]){
		case GRF_REAL:
			cudaFree(ctx.m_cudaReal[resId].first);
			//m_allocatedMemory[devId] -= ctx.m_cudaReal[resId].second;
			ctx.m_cudaReal.erase(resId);
			break;
		case GRF_INT:
			cudaFree(ctx.m_cudaInt[resId].first);
			//m_allocatedMemory[devId] -= ctx.m_cudaInt[resId].second;
			ctx.m_cudaInt.erase(resId);
			break;
		case GRF_UINT:
			cudaFree(ctx.m_cudaUInt[resId].first);
			//m_allocatedMemory[devId] -= ctx.m_cudaUInt[resId].second;
			ctx.m_cudaUInt.erase(resId);
			break;
		case GRF_BOOL:
			cudaFree(ctx.m_cudaBool[resId].first);
			//m_allocatedMemory[devId] -= ctx.m_cudaBool[resId].second;
			ctx.m_cudaBool.erase(resId);
			break;
		case GRF_ULONG:
			cudaFree(ctx.m_cudaULong[resId].first);
			//m_allocatedMemory[devId] -= ctx.m_cudaULong[resId].second;
			ctx.m_cudaULong.erase(resId);
			break;
		}
		cuCtxPopCurrent(&ctx.m_context);
		//updateAvailableMemory(devId);
	}

	void CUDAManager::deleteBuffer(int resId){
		for(unsigned int i=0; i<m_contexts.size(); ++i){
			deleteBuffer(i,resId);
		}
	}

	void CUDAManager::deleteGPUProgram(int devId, int pid){
		m_contexts[devId].m_cudaFunctions.erase(pid);
	}

	void CUDAManager::deleteGPUProgram(int pid){
		for(unsigned int i=0; i<m_contexts.size(); ++i){
			deleteGPUProgram(i,pid);
		}
	}

	int CUDAManager::markGPUTime(int devId){
		cudaDeviceContext &ctx = m_contexts[devId];
		cuCtxPushCurrent(ctx.m_context);
		cudaEventCreate(&ctx.m_cudaEvents[ctx.m_resourceIdCounter]);
		cudaEventRecord(ctx.m_cudaEvents[ctx.m_resourceIdCounter]);
		cuCtxPopCurrent(&ctx.m_context);
		return ctx.m_resourceIdCounter++;
	}

	int CUDAManager::markGPUTime(){
		return 0;
	}

	float CUDAManager::getGPUTime(int devId, int timer){
		cudaDeviceContext &ctx = m_contexts[0];
		cuCtxPushCurrent(ctx.m_context);
		cudaEvent_t start = ctx.m_cudaEvents[timer],stop;
		cudaEventCreate(&stop);

		cudaDeviceSynchronize();
		cudaEventRecord(stop);

		float timeMs;
		cudaEventElapsedTime(&timeMs,start,stop);

		cudaEventDestroy(stop);
		cudaEventDestroy(start);
		ctx.m_cudaEvents.erase(timer);
		cuCtxPopCurrent(&ctx.m_context);
		return timeMs;
	}

	float CUDAManager::getGPUTime(int timer){
		return 0;
	}

	void* CUDAManager::getResource(int devId, int resId){
		cudaDeviceContext &ctx = m_contexts[devId];
		switch(ctx.m_formatMap[resId]){
		case GRF_REAL:
			return ctx.m_cudaReal[resId].first;
		case GRF_INT:
			return ctx.m_cudaInt[resId].first;
		case GRF_BOOL:
			return ctx.m_cudaBool[resId].first;
		case GRF_UINT:
			return ctx.m_cudaUInt[resId].first;
		case GRF_ULONG:
			return ctx.m_cudaULong[resId].first;
		}
		return NULL;
	}

	void CUDAManager::updateAvailableMemory(int devId){
		size_t free = 0;
		size_t total = 0;
		CUresult result = CUDA_ERROR_UNKNOWN;
		cudaDeviceContext &ctx = m_contexts[devId];
		cuCtxPushCurrent(ctx.m_context);
		while(result != CUDA_SUCCESS){
			result = cuMemGetInfo(&free, &total);
			m_availableMemory[devId] = free;
			m_totalMemory[devId] = total;
			m_allocatedMemory[devId] = total-free;
		}
		cuCtxPopCurrent(&ctx.m_context);
	}
}