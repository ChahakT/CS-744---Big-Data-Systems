/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

#include <iostream>
#include <memory>
#include <array>
#include <typeinfo>

#include "gloo/allreduce.h"
#include "gloo/allreduce_ring.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/file_store.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/transport/tcp/device.h"

std::shared_ptr<gloo::Context> k_context;
int rank;
int size;
char hostname[HOST_NAME_MAX];

// Usage:
//
// Open two terminals. Run the same program in both terminals, using
// a different RANK in each. For example:
//
// A: PREFIX=test1 SIZE=2 RANK=0 example_allreduce
// B: PREFIX=test1 SIZE=2 RANK=1 example_allreduce
//
// Expected output:
//
//   data[0] = 18
//   data[1] = 18
//   data[2] = 18
//   data[3] = 18
//

void mysum(void* c_, const void* a_, const void* b_, int n) {
  printf("n=%d\r\n", n);
  int* c = static_cast<int*>(c_);
  const int* a = static_cast<const int*>(a_);
  const int* b = static_cast<const int*>(b_);
  for (auto i = 0; i < n; i++) {
    printf("a[%d]=%d\r\n", i, a[i]);
    printf("b[%d]=%d\r\n", i, b[i]);
    c[i] = a[i] + b[i];
    printf("c[%d]=%d\r\n", i, c[i]);
  }
}

void init(int rank, int size, std::string prefix, std::string network) {
    gloo::transport::tcp::attr attr;
    attr.iface = network;
    attr.ai_family = AF_UNSPEC;

    auto dev = gloo::transport::tcp::CreateDevice(attr);
    auto fileStore = gloo::rendezvous::FileStore("/proj/UWMadison744-F21/groups/akc/rendezvous_checkpoint-CT");
    auto prefixStore = gloo::rendezvous::PrefixStore(prefix, fileStore);
    auto context = std::make_shared<gloo::rendezvous::Context>(rank, size);
    context->connectFullMesh(prefixStore, dev);
    k_context = std::move(context);
    rank = k_context->rank;
    size = k_context->size;
}
int main(void) {
    if (getenv("PREFIX") == nullptr ||
        getenv("SIZE") == nullptr ||
        getenv("RANK") == nullptr ||
        getenv("ITERS") == nullptr ||
        getenv("NETWORK") == nullptr ||
        getenv("INPUT_SIZE") == nullptr) {
        std::cerr << "Please set environment variables PREFIX, SIZE, and RANK."
                  << std::endl;
        return 1;
    }
    std::string prefix = getenv("PREFIX");
    int rank = atoi(getenv("RANK"));
    int size = atoi(getenv("SIZE"));
    int iterations = atoi(getenv("ITERS"));
    std::string network = getenv("NETWORK");
    int inputEle = atoi(getenv("INPUT_SIZE"));

    init(rank, size, prefix, network);

  size_t elements = inputEle;
  std::vector<int*> inputPointers;
  std::vector<int*> outputPointers;
  for (size_t i = 0; i < elements; i++) {
    int *value = reinterpret_cast<int*>(malloc(sizeof(int)));
    *value = i * (rank + 1);
    inputPointers.push_back(value);
    int *value1 = reinterpret_cast<int*>(malloc(sizeof(int)));
    *value1 = 0;
    outputPointers.push_back(value1);
  }

  // Configure Allreduce Options struct
  gloo::AllreduceOptions opts_(k_context);
  opts_.setInputs(inputPointers, 1);
  opts_.setOutputs(outputPointers, 1);
  opts_.setAlgorithm(gloo::AllreduceOptions::Algorithm::RING);
  void (*fn)(void*, const void*, const void*, int) = &mysum;
  opts_.setReduceFunction(fn);
  gloo::allreduce(opts_);

  // Print the result.
  std::cout << "Output: " << std::endl;
  for (int i = 0; i < outputPointers.size(); i++) {
    std::cout << "data[" << i << "] = " << *outputPointers[i] << std::endl;
  }

  return 0;
}
