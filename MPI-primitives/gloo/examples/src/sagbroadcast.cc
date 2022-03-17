#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <numeric>
#include <memory>
#include <string>
#include <limits.h>
#include "mpiutil.cc"
#include "../include/sagbroadcast.h"

void runBcast(int rank, int size, int arrSize, int* sendbuf, int* recvbuf, bool enable_debug) {
    int debug = enable_debug;
    if (debug)
        std::cout << "Bcast " << rank << " " << size << "\n";
    int tag = 5643;
    int val;

    if (debug)
        std::cout << "Running scatter on " << rank << "\n";

    int w;
    int n = size;
    int count = arrSize / size;

    if (rank == 0) {
        if (__builtin_popcount(n) > 1) // count number of 1s set in the binary representation
            w = (1 << (32-__builtin_clz(n)));
        else w = n;
    } else
        w = (1 << __builtin_ctz(rank)); // count trailing number of 0s in binary representation
    if (rank != 0) {
        if (debug)
            std::cout << "\tWaiting to receive from " << (rank ^ w) << " elements count=" << (rank * arrSize / size) << "\n";
        mpiutil::MPI_Recv(recvbuf + (rank * arrSize / size), sizeof(int) * (count * w), rank ^ w, tag);

        if (debug) {
            std::cout << "\tReceived ";
            for (int i = 0; i < count * w; i++) {
                std::cout << recvbuf[i + (rank * arrSize / size)] << " ";
            }
            std::cout << " w=" << w << " count=" << count << "\n";
        }
    } else {
    }

    int k = 0;
    const int cn = count * n;
    static std::vector<std::unique_ptr<gloo::transport::UnboundBuffer>> pending_req;
    pending_req.clear();
    while (w > 0) {
        const int partner = rank | w;
        if (debug)
            std::cout << "w=" << w << "\n";
        if (partner > rank && partner < n) {
            const int wc = w * count;
            const int bytes = ((wc << 1) >= cn) ? (cn - wc): wc;
            if (debug) {
                for (int i = 0; i < bytes; i++) {
                    std::cout << "\tSending new sendbuf[" << ((partner * arrSize) / size + i) << "]"
                              << sendbuf[(partner * arrSize) / size + i] << " to " << partner << " w=" << w
                              << " count=" << count << " bytes=" << bytes << "\n";
                }
            }
            pending_req.push_back(std::move(mpiutil::MPI_ISend(sendbuf + partner * arrSize / size, bytes * sizeof(int), partner, tag)));
        }
        w >>= 1;
    }

    for (auto& i: pending_req) i->waitSend();

    if (debug) {
        std::cout << "Received Array: ";
        for (int i = 0; i < arrSize; i++) {
            std::cout << recvbuf[i] << " ";
        }
        std::cout << "\n";
        std::cout << "Running gather on " << rank << "\n";
    }

    // Ring All gather
    const int partner = (rank + 1) % n;
    const int partnerp = (rank - 1 + n) % n;
    int ri = rank, rp = rank - 1;
    if (rp < 0) rp = n - 1;
    for (int i = 0; i < n - 1; ++i) {

        if (rank == 0) {
            if (debug) {
                for (int j = 0; j < count; j++) {
                    std::cout << "\tSending buffer[" << (ri * count + j) << "] = " << sendbuf[(ri * count + j)]
                              << " from " << rank << " to " << partner << " and receiving from "
                              << partnerp << " in buffer[" << (rp * count + j) << "]\n";
                }
            }
            mpiutil::MPI_SendRecv(sendbuf + ri * count, sendbuf + rp * count,
                         sizeof(sendbuf[ri * count]) * count, sizeof(sendbuf[rp * count]) * count,
                         partner, partnerp, tag);
            if (debug) {
                for (int j = 0; j < count; j++) {
                    std::cout << "\tSent=" << sendbuf[ri * count + j] << " Received=" << sendbuf[rp * count + j]
                              << "\n";
                }
            }
        } else {
            if (debug) {
                for (int j = 0; j < count; j++) {
                    std::cout << "\tSending buffer[" << (ri * count + j) << "] = " << recvbuf[(ri * count + j)]
                              << " from " << rank << " to " << partner << " and receiving from "
                              << partnerp << " in buffer[" << (rp * count + j) << "]\n";
                }
            }
            mpiutil::MPI_SendRecv(recvbuf + ri * count, recvbuf + rp * count,
                         sizeof(recvbuf[ri * count]) * count, sizeof(recvbuf[rp * count]) * count,
                         partner, partnerp, tag);
            if (debug) {
                for (int j = 0; j < count; j++) {
                    std::cout << "\tSent=" << recvbuf[ri * count + j] << " Received=" << recvbuf[rp * count + j]
                              << "\n";
                }
            }
        }


        if (--ri == -1) ri = n-1;
        if (--rp == -1) rp = n-1;
    }
}