#ifndef GPU_LIST_RANKING
#define GPU_LIST_RANKING

#include "si.h"
#include "gpu_game.h"
#include "gpu_parity.h"
#include "game.h"

void process_reduced(int* split_succ, int* split_val, int* cum_val, int* split_inf, int n_pris, int s);

class GPUList : public GPUParity
{
    Game& g;
    GPUGame& g_dev; 

    int l;
    int s;

    int* h_split_succ;
    int* h_split_val;
    int* h_cum_val;
    int* h_split_inf;

    // Used by the GPU algorithm
    int* succ; // The successors for the linked list
    int* scratch; // Scratch space for the splitters
    int* split_succ; // The successors for the reduced list
    int* split_val; // The valuations for the segments in the reduced list
    int* split_inf; // The infinite markings the segments in the reduced list
    int* vert_list; // Stores the list that each vertex is contained in

public:

    GPUList(Game& g, GPUGame& g_dev);
    /*virtual void init_strat();*/
    virtual void compute_valuation();
    virtual void compute_first_val();
    /*virtual void mark_solved(int player);*/
    /*virtual int switch_strategy(int player);*/
    /*virtual void get_winning(std::vector<int>& out, int player);*/
};

#endif
