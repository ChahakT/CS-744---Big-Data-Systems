//
// Created by Chahak Tharani on 12/15/21.
//

#ifndef GLOO_TREE_REDUCE_GLOO_H
#define GLOO_TREE_REDUCE_GLOO_H

#pragma once
#include <stdio.h>
#include <assert.h>

class tree_reduce_gloo {
public:
    void runReduce(const int rank, int size, int inputEle, int* sendBuffer, int* recvBuffer, bool enable_debug);
};

#endif //GLOO_TREE_REDUCE_GLOO_H
