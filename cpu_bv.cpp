#include "cpu_bv.h"
#include "game.h"
#include "valuation.h"

#include <iostream>
#include <thread>

using namespace std;

template <typename T> CPUBV<T>::CPUBV(Game& g) : g(g), zero_val(g)
{
    for(int i = 0; i < g.get_n_vertices(); i++)
    {
        // Initialize valuation
        vals.push_back( T(g) );

        // Initialize the done array, used by compute_valuation
        done.push_back(0);

        infinite.push_back(0);

        solved.push_back(0);

    }
    // Initialize the zero valuation

}

// Determines whether the cycle starting at v is even or odd.
template <typename T> bool CPUBV<T>::even_cycle(int v)
{
    int current = v;
    int maxpri = 0;

    while(true)
    {
        current = strat[current];
        int current_pri = g.get_priority(current);
        if(current_pri > maxpri)
            maxpri = current_pri;

        if(current == v)
            break;
    }

    return maxpri % 2 == 0;
}

template <typename T> void CPUBV<T>::compute_valuation(int v)
{
    // The meaning of done:
    //      2 - we are done with the vertex
    //      1 - we have recursed on the vertex, if we see it again then we have     a cycle in the strategy
    //      0 - we have not touched the vertex

    if(infinite[v] == 1)
    {
        done[v] = 2;
        return;
    }

    if(strat[v] == -1) // Are we pointing to the sink?
    {
        // Initialize valuation
        vals[v].init_from_vertex(v);
        done[v] = 2;
        return;
    }

    int next = strat[v];

    // Check if we have recursed back to a vertex that we have visited
    if(done[next] == 1)
    {
        if(even_cycle(v))
            infinite[v] = 1;
        else
        {
            infinite[v] = -1;
            cout << "Game contains an odd cycle that uses only odd nodes" << endl;
            exit(1);
        }
        done[v] = 2;
        return;
    }

    // Recurse if necessary
    if(done[next] == 0)
    {
        done[v] = 1;
        compute_valuation(next);
    }

    // Check if a cycle was found
    if(infinite[next] != 0)
    {
        infinite[v] = infinite[next];
        done[v] = 2;
        return;
    }

    vals[v].copy_and_add(vals[next], v);

    done[v] = 2;

}

template <typename T> void CPUBV<T>::compute_valuation()
{
    for(int i = 0; i < g.get_n_vertices(); i++)
    {
        if(!solved[i])
        {
            done[i] = 0;
            infinite[i] = 0;
        }
        else
            done[i] = 2;
    }

    for(int i = 0; i < g.get_n_vertices(); i++)
        if(!done[i])
            compute_valuation(i);

}

template <typename T> void CPUBV<T>::print_val(int v)
{
    cout << "vertex " << v << " valuation: [";
    if(infinite[v] == 1)
        cout << "+inf";
    else if(infinite[v] == -1)
        cout << "-inf";
    else
    {
        if(v == -1)
            cout << zero_val.to_string();
        else
            cout << vals[v].to_string();
    }
    cout << "]" << endl;
}

template <typename T> int CPUBV<T>::compare_valuation(int v, int u)
{
    // First check if either valuation is infinte
    if(v != -1 and u != -1 and infinite[v] and infinite[u])
    {
        if(infinite[v] == 1 and infinite[u] == -1)
            return 1;
        if(infinite[v] == -1 and infinite[u] == 1)
            return -1;
        return 0;
    }
    if((v != -1 and infinite[v]) and (u == -1 or not infinite[u]))
    {
        if(infinite[v] == 1)
            return 1;
        else
            return -1;
    }
    if((u != -1 and infinite[u]) and (v == -1 or not infinite[v]))
    {
        if(infinite[u] == 1)
            return -1;
        else
            return 1;
    }

    // Both valuations are finite
    T* v_val;
    T* u_val; 

    if(v == -1)
        v_val = &zero_val;
    else
        v_val = &vals[v];

    if(u == -1)
        u_val = &zero_val;
    else
        u_val = &vals[u];


    return v_val->compare_valuation(*u_val);
}

template <typename T> int CPUBV<T>::switch_vertex(int v, int player)
{
    if(g.get_player(v) != player)
        return 0;

    int current = strat[v];
    int switched = 0;
    for(int u : g.get_edges(v))
    {
        int comparison = compare_valuation(u, current);

        if((player == 0 and comparison == 1) or (player == 1 and comparison == -1))
        {
            strat[v] = u;
            current = u;
            switched++;
        }
    }
    return switched;
}

// Switch strategy for player i using valuation v
template <typename T> int CPUBV<T>::switch_strategy(int player)
{
    int total_switched = 0;

    for(int v = 0; v < g.get_n_vertices(); v++)
    {
        total_switched += switch_vertex(v, player);
    }

    return total_switched;
}

template <typename T> void CPUBV<T>::mark_solved(int player)
{
    for(int i = 0; i < g.get_n_vertices(); i++)
    {
        if(player == 0 and infinite[i] == 1)
            solved[i] = 1;
        else if(player == 1 and infinite[i] == -1)
            solved[i] = 1;
    }
}

template <typename T> void CPUBV<T>::init_strat()
{
    // Initialize strategy
    for(int i = 0; i < g.get_n_vertices(); i++)
    {
        if(g.get_player(i) == 0)
            strat.push_back(-1); // The sink
        else
            strat.push_back(g.get_edges(i)[0]); // Arbitrary
    }
}

template <typename T> void CPUBV<T>::get_winning(std::vector<int>& out, int player)
{
    for(int i = 0; i < infinite.size(); i++)
    {
        if(infinite[i] == 1 and player == 0)
            out.push_back(i);
        else if(infinite[i] != 1 and player == 1)
            out.push_back(i);
    }
}

// Explicit template instantiations
template class CPUBV<ArrayVal>;
template class CPUBV<MapVal>;
