#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <memory>
#include <string>
#include <iostream>
#include "gloo/allreduce_ring.h"
#include <gloo/allreduce.h>
#include <mpi.h>
#include <algorithm>
#include <numeric>
#include "mpiutil.cc"
#include "../include/sag_reduce.h"

#include <unistd.h>
#include <limits.h>

int MPI_Recv1(
        void *buf,
        ssize_t bytes,
        int source,
        int tag,
        int receiveOffset,
        size_t receiveBytes,
        MPI_Comm comm) {
    auto ubuf = k_context->createUnboundBuffer(buf, bytes);
    ubuf->recv(source, tag, receiveOffset, receiveBytes);
    ubuf->waitRecv();
}

int MPI_Send_n_Recieve(
        const void *ubuf,
        ssize_t sbytes,
        int dest,
        int dtag,
        const void *vbuf,
        ssize_t rbytes,
        int src,
        int stag,
        int sendOffset,
        int receiveOffset,
        size_t sendReceiveBytes,
        MPI_Comm comm) {
    auto sbuf = k_context->createUnboundBuffer(const_cast<void*>(ubuf), sbytes);
    auto rbuf = k_context->createUnboundBuffer(const_cast<void*>(vbuf), rbytes);
    sbuf->send(dest, dtag, sendOffset, sendReceiveBytes);
    rbuf->recv(src, stag, receiveOffset, sendReceiveBytes);
    sbuf->waitSend();
    rbuf->waitRecv();
}

int MPI_Send1(
        const void *cbuf,
        ssize_t bytes,
        int dest,
        int tag,
        int sendOffset,
        size_t sendBytes,
        MPI_Comm comm) {
    auto ubuf = k_context->createUnboundBuffer(const_cast<void*>(cbuf), bytes);
    ubuf->send(dest, tag, sendOffset, sendBytes);
    ubuf->waitSend();
}

void runReduce(int rank, int size, int inputEle, int* sendBuffer, int* recvBuffer, bool enable_debug) {
    int tag = 5643;
    int partner;
    int mask = size >> 1;
    int begin = 0;
    int end = inputEle;
    int offset, sendOffset, receiveOffset;
    size_t sendReceiveBytes;

    while (mask){
        partner = mask ^ rank;

        if (rank & mask){
            std::fill(recvBuffer, recvBuffer + inputEle, 0);
            offset = begin + (end - begin) / 2;
            sendOffset = begin * sizeof(int);
            receiveOffset = offset * sizeof(int);
            sendReceiveBytes = (receiveOffset - sendOffset);

            MPI_Send_n_Recieve(
                    sendBuffer, sizeof(sendBuffer), partner, tag,
                    recvBuffer, sizeof(recvBuffer), partner, tag,
                    sendOffset, receiveOffset, sendReceiveBytes,
                    MPI_COMM_WORLD);
//            for (int c = 0; c < 8 ; c++) {
//                std::cout << recvBuffer[c] << " ";
//            }
            for (int c = 0; c < inputEle ; c++) {
                sendBuffer[c] = sendBuffer[c] + recvBuffer[c];
//                std::cout << sendBuffer[c] << " ";
            }
//            std::cout << std::endl;
            begin = offset;
        } else {
            std::fill(recvBuffer, recvBuffer + inputEle, 0);
            offset = begin + (end - begin) / 2;
            sendOffset = offset * sizeof(int);
            receiveOffset = begin * sizeof(int);
            sendReceiveBytes = (sendOffset - receiveOffset);
            MPI_Send_n_Recieve(
                    sendBuffer, sizeof(sendBuffer), partner, tag,
                    recvBuffer, sizeof(recvBuffer), partner, tag,
                    sendOffset, receiveOffset, sendReceiveBytes,
                    MPI_COMM_WORLD);
            for (int c = 0; c < inputEle; c++) {
                sendBuffer[c] = sendBuffer[c] + recvBuffer[c];
            }
            end = offset;
        }
        mask >>= 1;
    }

    // run Gather
    if (rank == 0){
        for (int all = 1; all < size; all++) {
            int receiveOffset = all * inputEle / size * sizeof(int);
            int receiveBytes = inputEle / size * sizeof(int);
            MPI_Recv1(recvBuffer, sizeof(recvBuffer), all, tag, receiveOffset, receiveBytes, MPI_COMM_WORLD);
            for (int c = all * inputEle/size; c < (all+1)*inputEle/size; c++) {
                sendBuffer[c] = recvBuffer[c];
            }
        }
    } else {
        int sendBytes = inputEle / size * sizeof(int);
        MPI_Send1(sendBuffer, sizeof(sendBuffer), 0, tag, rank * inputEle / size * sizeof(int), sendBytes, MPI_COMM_WORLD);
    }
//    if (rank == 0) {
//        for (int c = 0; c < 8; c++) {
//            std::cout << sendBuffer[c] << " ";
//        }
//        std::cout << std::endl;
//    }
}