#ifndef CPU_VJ
#define CPU_VJ

#include "si.h"
#include "game.h"
#include "valuation.h"

#include <vector>

template <typename T> class CPUVJ : public StrategyImprovement
{
    Game& g;
    std::vector < VJVal<T> > vals;
    std::vector < int > strat;
    std::vector <int> done;
    std::vector <int> on_cycle;

    int find_p_on_cycle(int v);
    void find_p(int v);
    void find_sd(int v);
public:
    CPUVJ(Game& g);
    virtual void init_strat();
    virtual void compute_valuation();
    virtual void compute_first_val() {compute_valuation();};
    virtual void mark_solved(int player);
    virtual int switch_strategy(int player);
    virtual void get_winning(std::vector<int>& out, int player);
};

#endif
