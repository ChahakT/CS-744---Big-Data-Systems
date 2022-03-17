//
// Created by Chahak Tharani on 12/15/21.
//

#ifndef GLOO_TREEBROADCAST_H
#define GLOO_TREEBROADCAST_H

#pragma once
#include <stdio.h>
#include <assert.h>

class treebroadcast {
public:
    void runBcast(const int rank, int size, int inputEle, int* sendBuffer, int* recvBuffer, bool enable_debug);
};

#endif //GLOO_TREEBROADCAST_H
