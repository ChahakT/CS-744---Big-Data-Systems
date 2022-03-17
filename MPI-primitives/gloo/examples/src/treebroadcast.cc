#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <numeric>
#include <memory>
#include <string>
#include "mpiutil.cc"
#include <limits.h>
#include "../include/treebroadcast.h"

void runBcast(int rank, int size, int vsize, int* buffer, bool enable_debug) {
    int debug = 0;
    if (debug)
        std::cout << "Bcast " << rank << " " << size << "\n";
    int tag = 5643;

    int logn = 1  << ( __builtin_ctz(rank));
    if (rank == 0) logn = 1  << (__builtin_clz(size));
    logn >>= 1;

    if (rank != 0) {
        const int partner = rank ^ (1 << __builtin_ctz(rank));\
        if (debug)
            std::cout << "Waiting at " << rank << " for " << partner << "\n";
        mpiutil::MPI_Recv(buffer, sizeof(buffer), partner, tag);
        if (debug)
            std::cout << "\tReceived" << sizeof(buffer) << "\n";
    }

    while (logn > 0) {
        const int partner = rank | logn;
        if (partner > rank && partner < size) {
            if (debug)
                std::cout << "Sending from " << rank << " to " << partner << "\n";
            mpiutil::MPI_Send(buffer, sizeof(buffer), partner, tag);
            if (debug)
                std::cout << "\tSent" << "\n";
        }
        logn >>= 1;
    }
}