#ifndef CPUBV_H
#define CPUBV_H

#include <vector>

#include "game.h"
#include "si.h"

template <typename T> class CPUBV : public StrategyImprovement
{
protected:
    Game& g;
    T zero_val;
    std::vector< int > done;
    std::vector< int > infinite;
    std::vector< int > solved;
    std::vector< T > vals;
    std::vector< int> strat;


    bool even_cycle(int v);
    void compute_valuation(int v);
public:

    CPUBV(Game& g);

    int compare_valuation(int v, int u);
    void print_val(int v);
    bool is_infinite(int v) { return infinite[v] == 1; };

    virtual void init_strat();
    virtual void compute_valuation();
    virtual void mark_solved(int player);
    virtual int switch_vertex(int vertex, int player);
    virtual int switch_strategy(int player);
    virtual void get_winning(std::vector<int>& out, int player);
};

#endif
