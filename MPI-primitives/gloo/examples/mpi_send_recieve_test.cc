/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <memory>
#include <string>
#include <mpi.h>

#include "gloo/allreduce_ring.h"
#include <gloo/allreduce.h>
#include <gloo/barrier.h>
#include "gloo/mpi/context.h"
#include "gloo/transport/tcp/device.h"

#define ASSERT(expr)                                            \
  do {                                                          \
    if (!(expr)) {                                              \
      throw std::runtime_error("Assertion failed: " #expr);     \
    }                                                           \
  } while (0);

// Global context
std::shared_ptr<gloo::Context> kContext;
int rank;
int size;

int MPI_Recv(
        void *buf,
        ssize_t bytes,
        int source,
        int tag,
        MPI_Comm comm)
{
    auto ubuf = kContext->createUnboundBuffer(buf, bytes);
    ubuf->recv(source, tag);
    ubuf->waitRecv();
}

int MPI_Send(
        const void *cbuf,
        ssize_t bytes,
        int dest,
        int tag,
        MPI_Comm comm)
{
    // Argument is logically const if we're only sending.
    auto ubuf = kContext->createUnboundBuffer(const_cast<void*>(cbuf), bytes);
    ubuf->send(dest, tag);
    ubuf->waitSend();
}

// Entrypoint of this example.
int run() {
    // Send on rank 0
    if (rank == 0) {
        const int dst = 1;
        const int tag = 1234;
        int pid = getpid();
        MPI_Send(&pid, sizeof(pid), dst, tag, MPI_COMM_WORLD);
        std::cout << "Sent to rank " << dst << ": " << pid << std::endl;
    }

    // Recv on rank 1
    if (rank == 1) {
        const int src = 0;
        const int tag = 1234;
        int pid = -1;
        MPI_Recv(&pid, sizeof(pid), src, tag, MPI_COMM_WORLD);
        std::cout << "Received from rank " << src << ": " << pid << std::endl;
    }

    // Barrier before exit
    MPI_Barrier(MPI_COMM_WORLD);
}

int run1() {
    // All connections are now established. We can now initialize some
    // test data, instantiate the collective algorithm, and run it.
    std::array<int, 4> data;
    std::cout << "Input: " << std::endl;
    for (int i = 0; i < data.size(); i++) {
        data[i] = i;
        std::cout << "data[" << i << "] = " << data[i] << std::endl;
    }

    // Allreduce operates on memory that is already managed elsewhere.
    // Every instance can take multiple pointers and perform reduction
    // across local buffers as well. If you have a single buffer only,
    // you must pass a std::vector with a single pointer.
    std::vector<int*> ptrs;
    ptrs.push_back(&data[0]);

    // The number of elements at the specified pointer.
    int count = data.size();

    // Instantiate the collective algorithm.
    auto allreduce = std::make_shared<gloo::AllreduceRing<int>>(
                    kContext, ptrs, count);

    // Run the algorithm.
    allreduce->run();

    // Print the result.
    std::cout << "Output: " << std::endl;
    for (int i = 0; i < data.size(); i++) {
        std::cout << "data[" << i << "] = " << data[i] << std::endl;
    }
    return 0;
}
// See example1.cc in this directory for a walkthrough of initialization.
void init() {
    // Initialize device
    gloo::transport::tcp::attr attr;
    attr.iface = "lo";
    attr.ai_family = AF_UNSPEC;
    auto dev = gloo::transport::tcp::CreateDevice(attr);
    // Initialize global context
    auto context = gloo::mpi::Context::createManaged();

    context->connectFullMesh(dev);
    kContext = std::move(context);
    rank = kContext->rank;
    size = kContext->size;
    std::cout<<"rank " << rank << " size " << size << std::endl;
    run();
}

int main(int argc, char** argv) {
    init();
    return 0;
}