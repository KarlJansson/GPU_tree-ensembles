#pragma once
#include "cuda_Common_Include.cu"

static __device__ unsigned int xorShift128(stateRNG_xorShift128* state){
	unsigned int t;

	t = state->x ^ (state->x << 11);
	state->x = state->y; state->y = state->z; state->z = state->w;
	return state->w = state->w ^ (state->w >> 19) ^ (t ^ (t >> 8));
}