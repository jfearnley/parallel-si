#ifndef BVSI_H
#define BVSI_H

#include <vector>

#include "game.h"
#include "si.h"

class CPUParity : public StrategyImprovement
{
protected:
    Game& g;
    std::vector<int> zero_val;
    std::vector< int > done;
    std::vector< int > infinite;
    std::vector< int > solved;
    std::vector< std::vector<int> > vals;
    std::vector< int> strat;


    bool even_cycle(int v);
    void compute_valuation(int v);
    int compare_valuation(std::vector<int>* v_val, std::vector<int>* u_val);
public:

    CPUParity(Game& g);

    int compare_valuation(int v, int u);
    void print_val(int v);
    bool is_infinite(int v) { return infinite[v] == 1; };

    virtual void init_strat();
    virtual void compute_valuation();
    virtual void mark_solved(int player);
    virtual int switch_strategy(int player);
    virtual void get_winning(std::vector<int>& out, int player);

    virtual void reset_opponent_strategy();
};

#endif
