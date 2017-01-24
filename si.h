#ifndef SI_H
#define SI_H

#include <vector>

class StrategyImprovement
{
public:
    virtual void init_strat() = 0;
    virtual void compute_valuation() = 0;
    virtual void compute_first_val() {compute_valuation();};
    virtual void mark_solved(int player) = 0;
    virtual int switch_strategy(int player) = 0;
    virtual void get_winning(std::vector<int>& out, int player) = 0;
    virtual void reset_opponent_strategy() {};
};

#endif
