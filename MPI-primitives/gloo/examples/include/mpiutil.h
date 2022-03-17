//
// Created by Chahak Tharani on 12/15/21.
//

#ifndef GLOO_MPIUTIL_H
#define GLOO_MPIUTIL_H

#pragma once
#include <stdio.h>
#include <assert.h>

class mpiutil {
public:
    static int MPI_Recv(void *buf,
                        ssize_t bytes,
                        int source,
                        int tag);
    static int MPI_Send(const void *cbuf,
            ssize_t bytes,
            int dest,
            int tag);
    auto MPI_ISend(
            const void *cbuf,
            ssize_t bytes,
            int dest,
            int tag);
    int MPI_SendRecv(
            const void *sendbuf,
            const void *recvbuf,
            ssize_t send_bytes,
            ssize_t recv_bytes,
            int dest,
            int src,
            int tag);
    int MPI_Barrier();
};

#endif //GLOO_MPIUTIL_H
