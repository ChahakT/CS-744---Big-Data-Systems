#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <ctime>
#include <numeric>
#include <memory>
#include <string>
#include "mpiutil.cc"
#include <unistd.h>
#include <math.h>
#include "limits.h"
#include "../include/tree_reduce.h"

void runReduce(const int rank, int size, int inputEle, int* sendBuffer, int* recvBuffer, bool enable_debug) {
    const int tag = 5643;
    int mask = 1;
//    std::cout << "tree_red run: " << round << " " << sizeof(sendBuffer) << " " << inputEle << "\n";
    while (mask < size) {
        const int partner = rank ^ mask;

        if (rank & mask ) {
            mpiutil::MPI_Send(sendBuffer, sizeof(sendBuffer), partner, tag, MPI_COMM_WORLD);
            return;
        } else {
            mpiutil::MPI_Recv(recvBuffer, sizeof(recvBuffer), partner, tag, MPI_COMM_WORLD);
            for (int c = 0; c < inputEle; c++) {
                sendBuffer[c] += recvBuffer[c];
            }
        }
        mask <<= 1;
    }

    if (enable_debug){
        printf("rank = %d returning %p: [", rank, recvBuffer);
        for (int i = 0; i < inputEle; ++i) printf("%d, ", sendBuffer[i]);
        printf("]\n");
    }
}