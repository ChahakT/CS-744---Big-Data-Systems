#include <chrono>
#include <bits/stdc++.h>
#include "tree_reduce.cc"
#include "sag_reduce.cc"
#include "sagbroadcast.cc"
#include "treebroadcast.cc"
#include <unistd.h>
#include <exception>

#include "gloo/transport/tcp/device.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/file_store.h"
#include "gloo/rendezvous/prefix_store.h"

int rank;
int size;
char hostname[HOST_NAME_MAX];
bool enable_debug;

enum class RunType {
    GLOO_TREE_REDUCE,
    GLOO_SAG_REDUCE,
    GLOO_TREE_BROADCAST,
    GLOO_SAG_BROADCAST,
    MY_TREE_REDUCE,
    MY_SAG_REDUCE,
    MY_TREE_BROADCAST,
    MY_SAG_BROADCAST,
    NONE
};

bool once = false;
__attribute__((always_inline))
inline void do_run(int rank, int size, int inputEle, int* sendBuffer, int* recvBuffer, RunType runtype, bool enable_debug) {
    if (runtype == RunType::MY_TREE_REDUCE) {
        if (!once) { std::cerr << "running my tree reduce\n"; once = true; }
        tree_reduce::runReduce(rank, size, inputEle, sendBuffer, recvBuffer, enable_debug);
    } else if (runtype == RunType::MY_SAG_REDUCE) {
        if (!once) { std::cerr << "running bcast-tree\n"; once = true; }
        sag_reduce::runReduce(rank, size, inputEle, sendBuffer, recvBuffer, enable_debug);
    } else if (runtype == RunType::GLOO_TREE_REDUCE) {
        if (!once) { std::cerr << "running bcast-sag\n"; once = true; }
        tree_reduce_gloo::runReduce(rank, size, inputEle, sendBuffer, recvBuffer, enable_debug);
    } else if (runtype == RunType::GLOO_SAG_REDUCE) {
        if (!once) { std::cerr << "running bcast-sag\n"; once = true; }
        sag_reduce_gloo::runReduce(rank, size, inputEle, sendBuffer, recvBuffer, enable_debug);
    } else if (runtype == RunType::MY_TREE_BROADCAST) {
        if (!once) { std::cerr << "running my tree reduce\n"; once = true; }
        treebroadcast::runBcast(rank, size, inputEle, sendBuffer, enable_debug);
    } else if (runtype == RunType::GLOO_TREE_BROADCAST) {
        if (!once) { std::cerr << "running bcast-tree\n"; once = true; }
        treebroadcast_gloo::runBcast(rank, size, inputEle, sendBuffer, enable_debug);
    } else if (runtype == RunType::MY_SAG_BROADCAST) {
        if (!once) { std::cerr << "running bcast-sag\n"; once = true; }
        sagbroadcast::runBcast(rank, size, inputEle, sendBuffer, recvBuffer, enable_debug);
    } else if (runtype == RunType::GLOO_SAG_BROADCAST) {
        if (!once) { std::cerr << "running bcast-sag\n"; once = true; }
        sagbroadcast_gloo::runBcast(rank, size, inputEle, sendBuffer, recvBuffer, enable_debug);
    } else {
        throw std::logic_error("no such runtype. check command line");
    }
}

void init(int rank, int size, std::string prefix, std::string network, std::string cpDir) {
    gloo::transport::tcp::attr attr;
    attr.iface = network;
    attr.ai_family = AF_UNSPEC;

    auto dev = gloo::transport::tcp::CreateDevice(attr);
    auto fileStore = gloo::rendezvous::FileStore(cpDir);
    auto prefixStore = gloo::rendezvous::PrefixStore(prefix, fileStore);
    auto context = std::make_shared<gloo::rendezvous::Context>(rank, size);
    context->connectFullMesh(prefixStore, dev);
    k_context = std::move(context);
    rank = k_context->rank;
    size = k_context->size;
}

double maxReduce(const int rank , int size, double input) {
    double sendBuffer[1] = {input};
    double recvBuffer[1] = {0};
    const int tag = 564;
    std::vector<double> allTimes;

    if (rank != 0) {
        MPI_Send(sendBuffer, sizeof(sendBuffer), 0, tag, MPI_COMM_WORLD);
    } else {
        for (int all = 1; all < size; all++) {
            MPI_Recv(recvBuffer, sizeof(recvBuffer), all, tag, MPI_COMM_WORLD);
            allTimes.push_back(recvBuffer[0]);
        }
    }

    if (rank == 0) {
        auto it  = std::max_element(std::begin(allTimes), std::end(allTimes));
        return *it;
    } else{
        return 0;
    }
}

int main (int argc, char *argv[]) {
    if (getenv("PREFIX") == nullptr ||
        getenv("SIZE") == nullptr ||
        getenv("RANK") == nullptr ||
        getenv("ITERATIONS") == nullptr ||
        getenv("NETWORK") == nullptr ||
        getenv("INPUT_SIZE") == nullptr ||
        getenv("RUN_TYPE") == nullptr ||
        getenv("CP_DIR") == nullptr ||
        getenv("DEBUG") == nullptr) {
        std::cerr << "Please set environment variables\n";
        return 1;
    }
    std::string prefix = getenv("PREFIX");
    int rank = atoi(getenv("RANK"));
    int size = atoi(getenv("SIZE"));
    int iterations = atoi(getenv("ITERATIONS"));
    std::string network = getenv("NETWORK");
    int inputEle = atoi(getenv("INPUT_SIZE"));
    std::string runType = getenv("RUN_TYPE");
    std::string cpDir = getenv("CP_DIR");
    enable_debug = getenv("DEBUG");

    RunType runtype = RunType::NONE;
    init(rank, size, prefix, network, cpDir);

    if (runType == "my_red_tree") {
        runtype = RunType::MY_TREE_REDUCE;
        std::cerr << "tree reduce\n";
    } else if (runType == "my_red_sag") {
        runtype = RunType::MY_SAG_REDUCE;
        std::cerr << "bcast\n";
    } else if (runType == "gloo_red_tree") {
        runtype = RunType::GLOO_TREE_REDUCE;
        std::cerr << "bcast\n";
    } else if (runType == "gloo_red_sag") {
        runtype = RunType::GLOO_SAG_REDUCE;
        std::cerr << "bcast\n";
    } else if (runType == "my_bcast_tree") {
        runtype = RunType::MY_TREE_BROADCAST;
        std::cerr << "bcast\n";
    } else if (runType == "my_bcast_sag") {
        runtype = RunType::MY_SAG_BROADCAST;
        std::cerr << "bcast\n";
    } else if (runType == "gloo_bcast_tree") {
        runtype = RunType::GLOO_TREE_BROADCAST;
        std::cerr << "bcast\n";
    } else if (runType == "gloo_bcast_sag") {
        runtype = RunType::GLOO_SAG_BROADCAST;
        std::cerr << "bcast\n";
    }

    int* sendBuffer = new int[inputEle];
    int* recvBuffer = new int[inputEle];

    for (int i = 0; i < 50; i++) {
        do_run(rank, size, inputEle, sendBuffer, recvBuffer, runtype, enable_debug);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    std::vector<double> all_stat;
    for (int i = 0; i < iterations; i++) {
        MPI_Barrier(MPI_COMM_WORLD);

        const auto start = std::chrono::high_resolution_clock::now();
        do_run(rank, size, inputEle, sendBuffer, recvBuffer, runtype, enable_debug);
        const auto end = std::chrono::high_resolution_clock::now();

        const std::chrono::duration<double> ets = end - start;
        const double elapsed_ts = ets.count();
        MPI_Barrier(MPI_COMM_WORLD);

        double maxTime = maxReduce(rank, size, elapsed_ts);
        if (rank == 0) {
            all_stat.push_back(maxTime);
        }
    }

    if (rank == 0) {
        double sum = std::accumulate(all_stat.begin(), all_stat.end(), 0.0);
        std::sort(all_stat.begin(), all_stat.end());
        double median = all_stat[all_stat.size()/2];
        double avg = sum / iterations;
        std::cout << median << "\n";
//        std::cout << avg << std::endl;
    }

    delete [] sendBuffer;
    delete [] recvBuffer;
    return 0;
}