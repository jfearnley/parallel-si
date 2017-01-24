#include "cpu_vj.h"
#include "valuation.h"

#include <iostream>

using namespace std;

/* XXX NOTE: This implementation is not currently working XXX */

template <typename T> CPUVJ<T>::CPUVJ(Game& g) : g(g)
{
    for(int i = 0; i < g.get_n_vertices(); i++)
    {
        vals.push_back( VJVal<T>(g) );

        done.push_back(0);

        on_cycle.push_back(0);
    }
}

template <typename T> void CPUVJ<T>::init_strat()
{
    for(int i = 0; i < g.get_n_vertices(); i++)
        strat.push_back(g.get_edges(i)[0]); // Pick an arbitrary strategy
}

// Requires that v is on a cycle
// Returns the largest priority on the cycle starting at v
template <typename T> int CPUVJ<T>::find_p_on_cycle(int v)
{
    int current = v;
    int maxpri = -1;

    while(true)
    {
        current = strat[current];

        on_cycle[current] = 1;

        int current_pri = g.get_priority(current);
        if(current_pri > maxpri)
            maxpri = current_pri;
     
        if(current == v)
            break;
    }

    return maxpri;
}

// Finds the largest priority on the cycle starting from v 
template <typename T> void CPUVJ<T>::find_p(int v)
{
    // Done meaning:
    // 0 - not yet touched
    // 1 - touched and recursive call made
    // 2 - finished and p value set

    int succ = strat[v];

    if(done[succ] == 2)
    {
        // Our successor is finalised, just copy the p value
        vals[v].copy_p(vals[succ]);
        done[v] = 2;
        return;
    }

    if(done[succ] == 1)
    {
        // We have just finished the cycle find the p value 
        int p = find_p_on_cycle(v);
        vals[v].set_p(p);
        done[v] = 2;
        return;
    }

    if(done[succ] == 0)
    {
        // Make the recursive call and copy the successors p value

        done[v] = 1;
        find_p(succ);
        vals[v].copy_p(vals[succ]);
        done[v] = 2;
        return;
    }
}

// Once the p value has been filled, this function finds s and d
template <typename T> void CPUVJ<T>::find_sd(int v)
{
    // Base case
    if(on_cycle[v] == 1 and g.get_priority(v) == vals[v].get_p())
    {
        done[v] = 3;
        vals[v].zero_sd();
        return;
    }

    // Make a recursive call and update the val using the successor
    int succ = strat[v];

    if(done[succ] != 3)
        find_sd(succ);

    vals[v].copy_and_add(vals[succ], v);

    done[v] = 3;
}

template <typename T> void CPUVJ<T>::compute_valuation()
{
    // Initialize helper variables
    for(int i = 0; i < g.get_n_vertices(); i++)
    {
        done[i] = 0;
        on_cycle[i] = 0;
    }

    // find_p changes done[v] from 0 to 2
    for(int i = 0; i < g.get_n_vertices(); i++)
        if(!done[i])
            find_p(i);

    // find_sd changes done[v] from 2 to 3
    for(int i = 0; i < g.get_n_vertices(); i++)
        if(done[i] == 2)
            find_sd(i);
}

template <typename T> void CPUVJ<T>::mark_solved(int player)
{
}

template <typename T> int CPUVJ<T>::switch_strategy(int player)
{
    int total_switched = 0;

    for(int v = 0; v < g.get_n_vertices(); v++)
    {
        if(g.get_player(v) != player)
            continue;

        int current = strat[v];
        for(int u : g.get_edges(v))
        {
            int comparison = vals[u].compare_to(vals[current]);

            if((player == 0 and comparison == 1) or (player == 1 and comparison == -1))
            {
                //if(player == 0)
                {
                    cout << "player " << player << " switched " << v << " to " << u << endl;
                    cout << vals[u].to_string() << " vs " << vals[current].to_string() << " = " << comparison << endl;
                }

                strat[v] = u;
                current = u;
                total_switched++;
            }
        }
    }


    return total_switched;
}

template <typename T> void CPUVJ<T>::get_winning(std::vector<int>& out, int player)
{
    for(int i = 0; i < g.get_n_vertices(); i++)
    {
        if(player == 0 and (vals[i].get_p() % 2) == 0)
            out.push_back(i);
        else if(player == 1 and (vals[i].get_p() % 2) == 1)
            out.push_back(i);
    }
}

template class CPUVJ<ArrayVal>;
template class CPUVJ<MapVal>;
