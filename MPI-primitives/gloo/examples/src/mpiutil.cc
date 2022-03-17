#include <gloo/barrier.h>
#include "../include/mpiutil.h"

std::shared_ptr<gloo::Context> k_context;

int MPI_Recv(
        void *buf,
        ssize_t bytes,
        int source,
        int tag) {
    auto ubuf = k_context->createUnboundBuffer(buf, bytes);
    ubuf->recv(source, tag);
    ubuf->waitRecv();
}

int MPI_Send(
        const void *cbuf,
        ssize_t bytes,
        int dest,
        int tag) {
    auto ubuf = k_context->createUnboundBuffer(const_cast<void*>(cbuf), bytes);
    ubuf->send(dest, tag);
    ubuf->waitSend();
}

auto MPI_ISend(
        const void *cbuf,
        ssize_t bytes,
        int dest,
        int tag) {
    auto ubuf = k_context->createUnboundBuffer(const_cast<void*>(cbuf), bytes);
    ubuf->send(dest, tag);
    return std::move(ubuf);
}

int MPI_SendRecv(
        const void *sendbuf,
        const void *recvbuf,
        ssize_t send_bytes,
        ssize_t recv_bytes,
        int dest,
        int src,
        int tag)
{
    auto usendbuf = k_context->createUnboundBuffer(const_cast<void*>(sendbuf), send_bytes);
    auto urecvbuf = k_context->createUnboundBuffer(const_cast<void*>(recvbuf), recv_bytes);
    usendbuf->send(dest, tag);
    urecvbuf->recv(src, tag);
    usendbuf->waitSend();
    urecvbuf->waitRecv();
}

int MPI_Barrier() {
    gloo::BarrierOptions opts(k_context);
    gloo::barrier(opts);
}