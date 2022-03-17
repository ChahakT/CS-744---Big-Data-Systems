//
// Created by Chahak Tharani on 12/17/21.
//

#ifndef GLOO_HELPER_H
#define GLOO_HELPER_H

#include "gloo/rendezvous/context.h"
#include "gloo/allreduce_ring.h"

std::shared_ptr<gloo::AllreduceRing<int>>
instantiateInput(int , int, std::shared_ptr<gloo::Context> );
//void instantiateAlgo();
void runAlgo(std::shared_ptr<gloo::AllreduceRing<int>>);
#endif //GLOO_HELPER_H
