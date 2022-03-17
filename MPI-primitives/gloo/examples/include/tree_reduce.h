#ifndef GLOO_TREE_REDUCE_H
#define GLOO_TREE_REDUCE_H

#pragma once
#include <stdio.h>
#include <assert.h>

class tree_reduce {
public:
    void runReduce(const int rank, int size, int inputEle, int* sendBuffer, int* recvBuffer, bool enable_debug);
};

#endif //GLOO_TREE_REDUCE_H
