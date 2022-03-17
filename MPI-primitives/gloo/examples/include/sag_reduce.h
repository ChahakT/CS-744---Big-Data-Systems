//
// Created by Chahak Tharani on 12/15/21.
//

#ifndef GLOO_SAG_REDUCE_H
#define GLOO_SAG_REDUCE_H

#pragma once
#include <stdio.h>
#include <assert.h>

class sag_reduce {
public:
    void runReduce(int rank, int size, int inputEle, int* sendBuffer, int* recvBuffer, bool enable_debug);
};

#endif //GLOO_SAG_REDUCE_H
