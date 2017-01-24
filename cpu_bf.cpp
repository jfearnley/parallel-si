#include "cpu_bf.h"
#include "valuation.h"
#include <iostream>
#include <stdlib.h>

using namespace std;

template <typename T> CPU_BellmanFord<T>::CPU_BellmanFord(Game& g)
    : CPUBV<T>(g)
{
    init = false;

    for(int v = 0; v < g.get_n_vertices(); v++)
    {
        bf_inf.push_back(1);
        on_neg_cycle.push_back(0);

        succ.push_back(0);

        if(g.get_player(v) == 1)
            p1_vertices++;
    }
}

template <typename T> void CPU_BellmanFord<T>::best_response(int* br_count)
{
    if(not init)
    {
        // Use strategy improvement to do the first iter
        while(true)
        {
            this->compute_valuation();
            this->mark_solved(1);
            if(this->switch_strategy(1) == 0)
                break;
        }

        init = true;
        return;

    }

    // Now compute the actual best response
    // Every vertex has infinite valuation at the start
    for(int v = 0; v < this->g.get_n_vertices(); v++)
        bf_inf[v] = 1;

    best_response_alg(false, br_count);
}

template <typename T> void CPU_BellmanFord<T>::best_response_alg(bool detect_neg_cycle, int* br_count)
{
    bool done = false; 
    int threshold = this->g.get_n_vertices();

    // Reinitialize valuation
    for(int v = 0; v < this->g.get_n_vertices(); v++)
    {
        this->vals[v] = T(this->g);
    }

    for(int i = 0; i < threshold; i++)
    {

        (*br_count)++;
        done = true;
        for(int v = 0; v < this->g.get_n_vertices(); v++)
        {
            if(this->solved[v])
                continue;

            // Player 0 vertices should just look at their strategy
            if(this->g.get_player(v) == 0)
            {
                int u = this->strat[v];
                if(u != -1 and bf_inf[u])
                    // Nothing changes if u is infinite
                    continue;

                if(bf_inf[v])
                {
                    // U's valuation is finite, so our valuation will also be
                    // finite
                    bf_inf[v] = 0;
                    done = false;
                }


                T u_val(this->g);
                if(u == -1)
                    u_val = this->zero_val;
                else
                    u_val = this->vals[u];
                u_val.add_vertex(v);

                if(this->vals[v] != u_val)
                {
                    this->vals[v] = u_val;
                    done = false;
                }
                continue;
            }

            // Player 1 vertices must take the minimum of their edges
            for(int u : this->g.get_edges(v))
            {
                if(u != -1 and bf_inf[u])
                {
                    // Other vertex has infinite valuation, ignore it
                    continue;
                }

                if(bf_inf[v])
                {
                    // Other vertex has finite valuation and we have infinite
                    // valuation. Copy their valuation
                    if (u != -1)
                        this->vals[v] = this->vals[u];
                    else
                        this->vals[v] = this->zero_val;

                    // And increment our priority
                    this->vals[v].add_vertex(v);

                    bf_inf[v] = 0;
                    succ[v] = u;
                    done = false;
                    continue;
                }

                // We both have finite valuations, do the comparison
                T val_cmp = this->vals[u];
                val_cmp.add_vertex(v);

                int comparison = this->vals[v].compare_valuation(val_cmp);

                if(comparison == 1)
                {

                    this->vals[v] = val_cmp;
                    succ[v] = u;
                    done = false;
                }
            }
        }

        if(done)
        {
            //cout << "done after " << i << " iterations" << endl;
            break;
        }
    }


    // Now update infinite[v] and strat[v]
    for(int v = 0; v < this->g.get_n_vertices(); v++)
    {
        if(bf_inf[v] and not this->solved[v])
        {
            //cout << "setting " << v << " to infinite" << endl;
            this->infinite[v] = 1;
        }
        if(this->g.get_player(v) == 1)
            this->strat[v] = this->succ[v];
    }
}

template class CPU_BellmanFord<ArrayVal>;
template class CPU_BellmanFord<MapVal>;
