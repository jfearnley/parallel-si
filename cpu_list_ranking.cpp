#include "cpu_list_ranking.h"
#include "gpu_list_ranking.h"
#include "valuation.h"
#include <thread>
#include <vector>
#include <iostream>
#include <stdlib.h>

/*
 * CPU reimplementation of the list ranking algorithm in gpu_list_ranking.cu
 * See that file for more substantive comments
 */

using namespace std;

template <typename T> CPUList<T>::CPUList(Game& g, int threads) 
        : CPUBV<T>(g), threads(threads)
{
    n = g.get_n_vertices();
    p = g.get_max_pri() + 1;
    l = 2*n + 2;
    s = (l-1)/128 + 1;

    succ = (int*)malloc(l * sizeof(int));
    vert_list = (int*)malloc((n+1) * sizeof(int));
    scratch = (int*)malloc(s * sizeof(int));
    split_succ = (int*)malloc(s * sizeof(int));
    split_inf = (int*)malloc(s * sizeof(int));

    for(int i = 0; i < n; i++)
        val_scratch.push_back(T(g));
    for(int i = 0; i < s; i++)
    {
        split_val.push_back(T(g));
        split_val_sum.push_back(T(g));
    }
}

// Dispatch [threads] many threads to member function f. 
// Return when all worker threads have returned
template <typename T> void CPUList<T>::launch(void (CPUList<T>::*f)(int, int), int n)
{

    // If there are too few values, just run the function in one thread
    if(n < threads)
    {
        (this->*f)(0, n-1);
        return;
    }

    // Launch threads
    vector<thread*> thread_list;
    for(int i = 0; i < threads; i++)
    {
        // Compute the subrange for the thread
        int start = i * (n/threads);
        int end;
        if(i == threads - 1)
            end = n-1;
        else
            end = (i+1) * (n/threads) -1;

        thread_list.push_back(new thread(f, this, start, end));
    }

    // Kill threads
    for(int i = 0; i < threads; i++)
    {
        thread_list[i]->join();
        delete thread_list[i];
    }
}

// Initialise the list that we are about to build
template <typename T> void CPUList<T>::init_list(int start, int end)
{
    for(int i = start; i <= end; i++)
    {
        succ[2 * i + 1] = 2*i;

        if(i == n)
            succ[2*n] = -1;
        else
            vert_list[i] = -1;
    }
}

template <typename T> void CPUList<T>::build_list(int start, int end)
{
    for(int i = start; i <= end; i++)
    {
        int next = this->strat[i];

        int next_up;
        if(next == -1)
            next_up = 2 * n + 1;
        else
            next_up = 2 * next + 1;

        int my_up = 2 * i + 1;
        int my_down = 2 * i;

        int new_down;
        __atomic_exchange(&succ[next_up], &my_up, &new_down, __ATOMIC_RELAXED);
        succ[my_down] = new_down;
    }
}

template <typename T> void CPUList<T>::pick_splitters(int start, int end)
{
    for(int i = start; i <= end; i++)
    {
        int splitter;
        if(i == s-1)
            splitter = l - 1; 
        else
            splitter = (int)(((float)(l-1) / (float)s) * (float)i);

        scratch[i] = succ[splitter];
        succ[splitter] = -100 -i;
    }
}

template <typename T> void CPUList<T>::update_val(T& cum_val, int current, int split)
{
    int vertex = current / 2;

    if(vertex == n or current == -1) // Sink vertex has no priority
        return;

    // Direction is -1 for down and +1 for up
    int direction = ((current % 2) * 2 - 1);

    // Add in this vertex's priority
    int this_p = this->g.get_priority(vertex);
    cum_val.add_to_pri(direction, this_p);

    // If current is an up edge, update val_scratch
    if(current % 2 == 1)
    {
        for(int i = 0; i < p; i++)
        {
            val_scratch[vertex] = cum_val;
        }

        // Store the location of the list
        vert_list[vertex] = split;
    }

}

template <typename T> void CPUList<T>::traverse_sublists(int start, int end)
{
    for(int idx = start; idx <= end; idx++)
    {
        int current = scratch[idx];

        T cum_val(this->g); 

        while(current >= 0)
        {
            update_val(cum_val, current, idx);

            current = succ[current];
        }

        split_val[idx] = cum_val;

        split_succ[idx] = current;
    }
}

template <typename T> void CPUList<T>::process_reduced()
{
    // Initialize the infinite array
    for(int i = 0; i < s; i++)
        split_inf[i] = 1;

    // Initialize split_val_sum
    for(auto& i : split_val_sum)
        i.zero();

    // Walk the reduced list from the start
    int current = s-1; // Up edge of the sink

    T val(this->g);

    while(current != -1)
    {
        // Copy the current val into the out array
        split_val_sum[current] = val;

        // Add in the next segment
        val.add_val(split_val[current]);

        // This vertex is not infinite
        split_inf[current] = 0;

        // Move to the next vertex
        int next = split_succ[current];
        if(next == -1) // The sink
            break;
        current = -(next+100); // -100 offset was used in pick_splitters
    }
}



template <typename T> void CPUList<T>::broadcast_val(int start, int end)
{
    for(int idx = start; idx <= end; idx++)
    {
        int vertex = idx;
        int sublist = vert_list[vertex];


        if(this->infinite[vertex] == -1) // Odd wins this vertex, ignore it
        {
            continue;
        }

        if(sublist == -1) // Vertex was not touched by any sublist
        {
            this->infinite[vertex] = 1;
            continue;
        }


        if(split_inf[sublist] == 0)
        {
            this->vals[vertex].copy_and_add(split_val_sum[sublist], val_scratch[vertex]);
            this->infinite[vertex] = 0;

        }
        else
        {
            this->infinite[vertex] = 1;
        }
    }
}


template <typename T> void CPUList<T>::compute_valuation()
{
    launch(&CPUList<T>::init_list, n+1);
    launch(&CPUList<T>::build_list, n);

    launch(&CPUList<T>::pick_splitters, s);
    launch(&CPUList<T>::traverse_sublists, s);


    process_reduced();

    launch(&CPUList<T>::broadcast_val, n);

}

template <typename T> void CPUList<T>::compute_first_val()
{
    CPUList<T>::compute_valuation();
}

template <typename T> void CPUList<T>::switch_vertices(int start, int end)
{
    int total = 0;
    for(int idx = start; idx <= end; idx++)
        total += this->switch_vertex(idx, player);
    total_switched += total;
}

template <typename T> int CPUList<T>::switch_strategy(int player)
{
    this->player = player;
    total_switched = 0;
    launch(&CPUList<T>::switch_vertices, n);
    return total_switched;
}

template class CPUList<ArrayVal>;
template class CPUList<MapVal>;
