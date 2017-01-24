#include "cpu_parity.h"
#include "game.h"

#include <iostream>
#include <thread>

using namespace std;

CPUParity::CPUParity(Game& g) : g(g)
{
    for(int i = 0; i < g.get_n_vertices(); i++)
    {
        // Initialize valuation
        vals.push_back( vector<int>() );
        for(int j = 0; j <= g.get_max_pri(); j++)
        {
            vals[i].push_back(0);
        }

        // Initialize the done array, used by compute_valuation
        done.push_back(0);

        infinite.push_back(0);

        solved.push_back(0);

    }
    // Initialize the zero valuation
    for(int j = 0; j <= g.get_max_pri(); j++)
        zero_val.push_back(0);

}

// Determines whether the cycle starting at v is even or odd.
bool CPUParity::even_cycle(int v)
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

void CPUParity::compute_valuation(int v)
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
        for(int i = 0; i <= g.get_max_pri(); i++)
            vals[v][i] = 0;

        vals[v][g.get_priority(v)] = 1;
        done[v] = 2;
        return;
    }

    int next = strat[v];

    // Check for cycles
    if(done[next] == 1)
    {
        if(even_cycle(v))
            infinite[v] = 1;
        else
            infinite[v] = -1;
        done[v] = 2;
        return;
    }

    // Recurse
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

    // Copy the successor's valuation
    for(int i = 0; i <= g.get_max_pri(); i++)
        vals[v][i] = vals[next][i];

    vals[v][g.get_priority(v)] += 1;

    done[v] = 2;

}

void CPUParity::compute_valuation()
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

void CPUParity::print_val(int v)
{
    cout << "vertex " << v << " valuation: [";
    if(infinite[v] == 1)
        cout << "+inf";
    else if(infinite[v] == -1)
        cout << "-inf";
    else
    {
        if(v == -1)
            for(int i : zero_val)
                cout << i << " ";
        else
            for(int i : vals[v])
                cout << i << " ";
    }
    cout << "]" << endl;
}
// Returns 1 if v > u, 0 if v = u, and -1 if u > v
int CPUParity::compare_valuation(vector<int>* v_val, vector<int>* u_val)
{
    // Find the maxdiff
    int maxdiff = -1;
    for(int i = v_val->size() - 1; i >= 0; i--)
    {
        if((*v_val)[i] != (*u_val)[i])
        {
            maxdiff = i;
            break;
        }
    }

    if(maxdiff == -1) // Both valuations were identical
        return 0;

    int v_count = (*v_val)[maxdiff];
    int u_count = (*u_val)[maxdiff];

    if(maxdiff % 2 == 0) // Maxdiff is even
    {
        if(v_count > u_count)
            return 1;
        else if(v_count == u_count)
            return 0;
        else
            return -1;
    }

    if(maxdiff % 2 == 1) // Maxdiff is odd
    {
        if(v_count < u_count)
            return 1;
        else if(v_count == u_count)
            return 0;
        else
            return -1;
    }
}

int CPUParity::compare_valuation(int v, int u)
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
    vector<int>* v_val;
    vector<int>* u_val; 

    if(v == -1)
        v_val = &zero_val;
    else
        v_val = &vals[v];

    if(u == -1)
        u_val = &zero_val;
    else
        u_val = &vals[u];


    return compare_valuation(v_val, u_val);
}

// Switch strategy for player i using valuation v
int CPUParity::switch_strategy(int player)
{
    int total_switched = 0;

    for(int v = 0; v < g.get_n_vertices(); v++)
    {
        if(g.get_player(v) != player)
            continue;

        int current = strat[v];
        for(int u : g.get_edges(v))
        {
            int comparison = compare_valuation(u, current);

            if((player == 0 and comparison == 1) or (player == 1 and comparison == -1))
            {
                strat[v] = u;
                current = u;
                total_switched++;
            }
        }
    }

    return total_switched;
}

void CPUParity::mark_solved(int player)
{
    for(int i = 0; i < g.get_n_vertices(); i++)
    {
        if(player == 0 and infinite[i] == 1)
            solved[i] = 1;
        else if(player == 1 and infinite[i] == -1)
            solved[i] = 1;
    }
}

void CPUParity::init_strat()
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

void CPUParity::get_winning(std::vector<int>& out, int player)
{
    for(int i = 0; i < infinite.size(); i++)
    {
        if(infinite[i] == 1 and player == 0)
            out.push_back(i);
        else if(infinite[i] != 1 and player == 1)
            out.push_back(i);
    }
}


void CPUParity::reset_opponent_strategy()
{
    for(int i = 0; i < g.get_n_vertices(); i++)
    {
        if(g.get_player(i) == 1)
            strat[i] = g.get_edges(i)[0]; // Arbitrary
    }
}
