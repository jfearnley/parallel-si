#ifndef GPU_PARITY
#define GPU_PARITY

#include "si.h"
#include "gpu_game.h"
#include "game.h"

class GPUParity : public StrategyImprovement
{
protected:
    Game& g;
    GPUGame& g_dev; 

    int val_size;

    int* strategy;
    int* infinite;
    int* solved;

    int* current_val;

    // Scratch space used by the algorithm
    int* next1;
    int* next2;
    int* val1;
    int* val2;
    int* val_diffs;
    int* done;

public:
    GPUParity(Game& g, GPUGame& g_dev);
    virtual void init_strat();
    virtual void compute_valuation();
    virtual void compute_first_val() {compute_valuation();};
    virtual void mark_solved(int player);
    virtual int switch_strategy(int player);
    virtual void get_winning(std::vector<int>& out, int player);
};

#endif
