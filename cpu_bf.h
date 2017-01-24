#ifndef CPU_BF_H
#define CPU_BF_H

#include "si.h"
#include "cpu_bv.h"

template <typename T> class CPU_BellmanFord : public CPUBV<T>
{
    std::vector<int> bf_inf; // Infinite valuation for bellman ford
    std::vector<int> on_neg_cycle; // Vertex is on cycle at end of bellman ford
    std::vector<int> succ; // successors for the b-f algorithm
    int p1_vertices;
    void best_response_alg(bool detect_neg_cycle, int* br_count); 
    bool init;
public:
    CPU_BellmanFord(Game& g);
    void best_response(int* br_count); 
};

#endif
