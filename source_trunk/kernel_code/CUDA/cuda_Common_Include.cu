#pragma once
#define FLT_MAX		3.402823466e+38F
#define UINT_MAX	0xffffffff
#define INT_MAX     2147483647

struct stateRNG_xorShift128{
	unsigned int x;
	unsigned int y;
	unsigned int z;
	unsigned int w;
};