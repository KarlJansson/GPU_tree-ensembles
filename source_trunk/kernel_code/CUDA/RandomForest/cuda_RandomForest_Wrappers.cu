#include "cuda_RandomForest_Constants.cu"

namespace Bagging{
	__global__ void kernel_entry(paramPack_Kernel params);
	__host__ void cuda_RandomForest_UpdateConstants(void* src);
}

namespace ExtremeCreateNodes{
	__global__ void kernel_entry(paramPack_Kernel params);
	__host__ void cuda_RandomForest_UpdateConstants(void* src);
}

namespace ExtremeFindSplit{
	__global__ void kernel_entry(paramPack_Kernel params);
	__host__ void cuda_RandomForest_UpdateConstants(void* src);
}

namespace ExtremeMakeSplit{
	__global__ void kernel_entry(paramPack_Kernel params);
	__host__ void cuda_RandomForest_UpdateConstants(void* src);
}

namespace Sort{
	__global__ void kernel_entry(paramPack_Kernel params);
	__host__ void cuda_RandomForest_UpdateConstants(void* src);
}

namespace FindSplit{
	__global__ void kernel_entry(paramPack_Kernel params);
	__host__ void cuda_RandomForest_UpdateConstants(void* src);
}

namespace EvaluateSplit{
	__global__ void kernel_entry(paramPack_Kernel params);
	__host__ void cuda_RandomForest_UpdateConstants(void* src);
}

namespace RandomForest_SplitData_Kernel{
	__global__ void kernel_entry(paramPack_Kernel params);
	__host__ void cuda_RandomForest_UpdateConstants(void* src);
}

namespace RandomForest_Kernel_Classify{
	__global__ void kernel_entry(paramPack_Kernel params);
	__host__ void cuda_RandomForest_UpdateConstants(void* src);
}

enum KernelID {KID_Bagging = 0, KID_Build, KID_KeplerBuild, KID_Sort, KID_FindSplit, KID_Split, KID_EvaluateSplit, KID_Classify, KID_ExtremeFindSplit, KID_ExtremeCreateNodes, KID_ExtremeMakeSplit};
extern "C" void cuda_RandomForest_Wrapper_UpdateConstants(void* src, int id){
	switch(id){
	case KID_Bagging:
		Bagging::cuda_RandomForest_UpdateConstants(src);
		break;
	case KID_Sort:
		Sort::cuda_RandomForest_UpdateConstants(src);
		break;
	case KID_FindSplit:
		FindSplit::cuda_RandomForest_UpdateConstants(src);
		break;
	case KID_Split:
		RandomForest_SplitData_Kernel::cuda_RandomForest_UpdateConstants(src);
		break;
	case KID_EvaluateSplit:
		EvaluateSplit::cuda_RandomForest_UpdateConstants(src);
		break;
	case KID_Classify:
		RandomForest_Kernel_Classify::cuda_RandomForest_UpdateConstants(src);
		break;
	case KID_ExtremeFindSplit:
		ExtremeFindSplit::cuda_RandomForest_UpdateConstants(src);
		break;
	case KID_ExtremeCreateNodes:
		ExtremeCreateNodes::cuda_RandomForest_UpdateConstants(src);
		break;
	case KID_ExtremeMakeSplit:
		ExtremeMakeSplit::cuda_RandomForest_UpdateConstants(src);
		break;
	default:
		break;
	}	
}

dim3 getGrid(unsigned int num_threads){
	int gridX = ceil((c_precision(num_threads)/c_precision(thread_group_size)));

	return dim3(gridX,1,1);
}

extern "C" void cuda_RandomForest_Wrapper_Bagging(unsigned int num_threads, void* params){
	dim3 grid = getGrid(num_threads);
    dim3 threads(thread_group_size, 1, 1);
    Bagging::kernel_entry<<< grid, threads >>>(*((Bagging::paramPack_Kernel*)params));
}

extern "C" void cuda_RandomForest_Wrapper_Build(unsigned int num_threads, void* params){
}

extern "C" void cuda_RandomForest_Wrapper_KeplerBuild(unsigned int num_threads, void* params){
}

extern "C" void cuda_RandomForest_Wrapper_Sort(unsigned int num_threads, void* params){
	dim3 grid = getGrid(num_threads);
    dim3 threads(thread_group_size, 1, 1);
    Sort::kernel_entry<<< grid, threads >>>(*((Sort::paramPack_Kernel*)params));
}

extern "C" void cuda_RandomForest_Wrapper_FindSplit(unsigned int num_threads, void* params){
	dim3 grid = getGrid(num_threads);
    dim3 threads(thread_group_size, 1, 1);
    FindSplit::kernel_entry<<< grid, threads >>>(*((FindSplit::paramPack_Kernel*)params));
}

extern "C" void cuda_RandomForest_Wrapper_EvaluateSplit(unsigned int num_threads, void* params){
	dim3 grid = getGrid(num_threads);
    dim3 threads(thread_group_size, 1, 1);
	EvaluateSplit::kernel_entry<<< grid, threads >>>(*((EvaluateSplit::paramPack_Kernel*)params));
}

extern "C" void cuda_RandomForest_Wrapper_SplitData(unsigned int num_threads, void* params){
	dim3 grid = getGrid(num_threads);
    dim3 threads(thread_group_size, 1, 1);
	RandomForest_SplitData_Kernel::kernel_entry<<< grid, threads >>>(*((RandomForest_SplitData_Kernel::paramPack_Kernel*)params));
}

extern "C" void cuda_RandomForest_Wrapper_Classify(unsigned int num_threads, void* params){
	dim3 grid = getGrid(num_threads);
    dim3 threads(thread_group_size, 1, 1);
	RandomForest_Kernel_Classify::kernel_entry<<< grid, threads >>>(*((RandomForest_Kernel_Classify::paramPack_Kernel*)params));
}

extern "C" void cuda_RandomForest_Wrapper_ExtremeFindSplit(unsigned int num_threads, void* params){
	dim3 grid = getGrid(num_threads);
    dim3 threads(thread_group_size, 1, 1);
    ExtremeFindSplit::kernel_entry<<< grid, threads >>>(*((ExtremeFindSplit::paramPack_Kernel*)params));
}

extern "C" void cuda_RandomForest_Wrapper_ExtremeCreateNodes(unsigned int num_threads, void* params){
	dim3 grid = getGrid(num_threads);
    dim3 threads(thread_group_size, 1, 1);
    ExtremeCreateNodes::kernel_entry<<< grid, threads >>>(*((ExtremeCreateNodes::paramPack_Kernel*)params));
}

extern "C" void cuda_RandomForest_Wrapper_ExtremeMakeSplit(unsigned int num_threads, void* params){
	dim3 grid = getGrid(num_threads);
    dim3 threads(thread_group_size, 1, 1);
    ExtremeMakeSplit::kernel_entry<<< grid, threads >>>(*((ExtremeMakeSplit::paramPack_Kernel*)params));
}