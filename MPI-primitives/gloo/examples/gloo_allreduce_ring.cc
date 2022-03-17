/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

#include "helper.h"
#include <iostream>
#include <memory>
#include <chrono>
#include <bits/stdc++.h>
#include <vector>
#include <numeric>
#include <mpi.h>
#include <gloo/barrier.h>
#include "gloo/allreduce_ring.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/file_store.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/transport/tcp/device.h"

std::shared_ptr<gloo::AllreduceRing<int>>
instantiateInput(int inputEle, int size, std::shared_ptr<gloo::Context> k_context) {
    std::vector<int> data(inputEle);
    for (int i = 0; i < data.size(); i++) {
        data[i] = i;
    }
    std::vector<int *> ptrs;
    ptrs.push_back(&data[0]);
    int *outPtr = reinterpret_cast<int *>(malloc(sizeof(int) * inputEle * size));
    int count = data.size();
    return std::make_shared<gloo::AllreduceRing<int>>(
            k_context, ptrs, count);
}

//void instantiateAlgo() {
//    // Instantiate the collective algorithm.
//    allreduce = std::make_shared<gloo::AllreduceRing<int>>(
//            k_context, ptrs, count);
//}

void runAlgo(std::shared_ptr<gloo::AllreduceRing<int>> allreduce){
    allreduce->run();
}
