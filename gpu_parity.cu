#include "gpu_parity.h"
#include "gpu_util.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

/* This is an implementation that computes valuations using a tree vartiation of
 * Wiley's algorithm. It is not reported on in the paper, because the
 * Helman-Jaja list ranking approach is always faster. See gpu_list_ranking.cu
 * for that algorithm.
 */

using namespace std;

GPUParity::GPUParity(Game& g, GPUGame& g_dev) 
    : g(g), g_dev(g_dev)
{
    // Allocate memory used by the algorithm
    int n = g.get_n_vertices();
    init_zero(&strategy, n);
    init_zero(&infinite, n);
    init_zero(&solved, n);

    int mp = g.get_max_pri();
    val_size = n * (mp+1);

    init_zero(&val1, val_size);
    init_zero(&val2, val_size);
    init_zero(&next1, n);
    init_zero(&next2, n);
    init_zero(&done, 1);

}

/*************************  Init strat ***************************************/

__global__ void d_init_strat(int* strategy, int* player, int* vertices, int* edges, int n)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= n)
        return;

    if(player[idx] == 0)
        strategy[idx] = -1; // The sink
    else
        strategy[idx] = edges[vertices[idx]]; // An arbitrary edge

}

void GPUParity::init_strat()
{
    int n = g.get_n_vertices();
    d_init_strat <<< nblocks(n), BLOCK_SIZE >>>(strategy, g_dev.player,
            g_dev.vertices, g_dev.edges, g.get_n_vertices());
}

/*************************  Compute valuation *******************************/

// Initailize val and next 
__global__ void d_init_val(int* val, int* next, int* strategy, int* priority,
        int val_size, int pri_mem)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= val_size)
        return;

    int vertex = idx / pri_mem;
    int p = idx % pri_mem;

    if(p == 0)
        next[vertex] = strategy[vertex];

    val[idx] = 0;
    if(p == priority[vertex])
        val[idx] = 1;
}

__global__ void d_val_step(int* val_in, int* val_out, int* next_in, int*
        next_out, int val_size, int pri_mem)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= val_size)
        return;


    int vertex = idx / pri_mem;
    int p = idx % pri_mem;

    int next_vertex = next_in[vertex];
    if(next_vertex == -1)
    {
        // Vertex is done, just copy the valuation
        val_out[idx] = val_in[idx]; 
        if(p == 0)
            next_out[vertex] = next_vertex;
        return;
    }


    int target_index = next_vertex * pri_mem + p; // Where we should look

    val_out[idx] = val_in[idx] + val_in[target_index];
    if(p == 0)
        next_out[vertex] = next_in[next_vertex];
}

__device__ int max_diff(int* val, int val_idx, int pri_mem)
{
    int max_pri = -1;
    for(int i = pri_mem - 1; i >= 0; i--)
    {
        if(val[val_idx + i] != 0)
        {
            max_pri = i;
            break;
        }
    }
    return max_pri;
}

__global__ void d_mark_infinite(int* val, int* next, int* infinite, int n, int
        pri_mem)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= n)
        return;

    // Check if the vertex is inifinite
    if(next[idx] == -1)
    {
        infinite[idx] = 0;
        return;
    }

    // The vertex in next[idx] is definitely on the cycle, so find the largest
    // non-zero value in that vertex's valuation
    int cycle_vert = next[idx];
    int val_idx = cycle_vert * pri_mem;

    int max_pri = max_diff(val, val_idx, pri_mem);


    if(max_pri % 2 == 0) // Cycle is even
        infinite[idx] = 1;
    else
        infinite[idx] = -1;

}

void GPUParity::compute_valuation()
{
    int* val = val1;
    int* next = next1;

    int* val_other = val2;
    int* next_other = next2;

    int pri_mem = g.get_max_pri() + 1;

    d_init_val <<< nblocks(val_size), BLOCK_SIZE >>>
        (val, next, strategy, g_dev.priority, val_size, pri_mem);

    int step = g_dev.num_vert;
    while(step > 0)
    {
        // We will do the loop log n times
        step = step >> 1;

        d_val_step <<< nblocks(val_size), BLOCK_SIZE >>>
            (val, val_other, next, next_other, val_size, pri_mem);


        // Swap the val and temp arrays
        int* val_tmp = val; val = val_other; val_other = val_tmp;
        int* next_tmp = next; next = next_other; next_other = next_tmp;


    }

    d_mark_infinite <<< nblocks(g.get_n_vertices()), BLOCK_SIZE >>>
        (val, next, infinite, g.get_n_vertices(), pri_mem);

    // Save the valuation for use in other functions
    current_val = val;

}

/************************* Switch strategy  *******************************/

__device__ inline int compare_valuation(int v, int u, int* vals, int* infinite, int pri_mem)
{
    // First check if either valuation is infinte
    int inf_v;
    if (v != -1)
       inf_v = infinite[v];
    int inf_u;
    if (u != -1)
       inf_u = infinite[u];

    if(v != -1 and u != -1 and inf_v and inf_u)
    {
        if(inf_v == 1 and inf_u == -1)
            return 1;
        if(inf_v == -1 and inf_u == 1)
            return -1;
        return 0;
    }
    if((v != -1 and inf_v) and (u == -1 or not inf_u))
    {
        if(inf_v == 1)
            return 1;
        else
            return -1;
    }
    if((u != -1 and inf_u) and (v == -1 or not inf_v))
    {
        if(inf_u == 1)
            return -1;
        else
            return 1;
    }

    // Both valuations are finite

    // Find the maxdiff
    int maxdiff = -1;
    int v_idx = v * pri_mem;
    int u_idx = u * pri_mem;
    for(int i = pri_mem - 1; i >= 0; i--)
    {
        int diff;
        if(u == -1)
            diff = vals[v_idx+i];
        else
        {
            diff = vals[v_idx+i] - vals[u_idx+i];
        }

        if(diff != 0)
        {
            maxdiff = i;
            break;
        }
    }

    if(maxdiff == -1) // Both valuations were identical
        return 0;

    int v_count = vals[v_idx + maxdiff];
    int u_count;
    if(u == -1)
        u_count = 0;
    else
        u_count = vals[u_idx + maxdiff];


    if(maxdiff % 2 == 0) // Maxdiff is even
        return v_count - u_count;
    else // Maxdiff is odd
        return u_count - v_count;
}

__global__ void d_switch_strategy(int* vals, int* strategy, int* infinite, int*
        vertices, int* edges, int* player, int n, int player_switch, int
        pri_mem, int* done)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= n)
        return;

    if(player[idx] != player_switch)
        return;

    int edge_start = vertices[idx];
    // valid even for the final vertex, see constructor of GPUGame
    int edge_end = vertices[idx+1]; 

    int current = strategy[idx];
    int switched = false;
    for(int i = edge_start; i < edge_end; i++)
    {
        int u = edges[i];
        int comparison = compare_valuation(u, current, vals, infinite, pri_mem);

        if((player_switch == 0 and comparison > 0) or (player_switch == 1 and comparison < 0))
        {
            current = u;
            switched = true;
        }

    }

    if(switched)
    {
        //printf("idx %d switched to %d\n", idx, current);
        strategy[idx] = current;
        *done = 1; // It does not matter who wins the race
    }
}


int GPUParity::switch_strategy(int player)
{
    int pri_mem = g.get_max_pri() + 1;

    gpu_assert(cudaMemset(done, 0, sizeof(int)));

    d_switch_strategy <<< nblocks(g.get_n_vertices()), BLOCK_SIZE >>>
        (current_val, strategy, infinite, g_dev.vertices, g_dev.edges,
         g_dev.player, g.get_n_vertices(), player, pri_mem, done);

    int h_done = 0;
    gpu_assert(cudaMemcpy(&h_done, done, sizeof(int), cudaMemcpyDeviceToHost));

    return h_done;
}

/***************************** Mark solved ***********************************/

__global__ void d_mark_solved(int* solved, int* infinite, int player, int n)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= n)
        return;

    if(player == 0 and infinite[idx] == 1)
        solved[idx] = 1;
    else if(player == 1 and infinite[idx] == -1)
        solved[idx] = -1;
}

void GPUParity::mark_solved(int player)
{
    // Not used as it makes the algorithm slower
}

/***************************** get_winning *********************************/

void GPUParity::get_winning(std::vector<int>& out, int player)
{
    int n = g.get_n_vertices();
    int* h_infinite = (int*)malloc(n * sizeof(int));
    gpu_assert(cudaMemcpy(h_infinite, infinite, n * sizeof(int), cudaMemcpyDeviceToHost));

    for(int i = 0; i < n; i++)
    {
        if(h_infinite[i] == 1 and player == 0)
            out.push_back(i);
        else if(h_infinite[i] != 1 and player == 1)
            out.push_back(i);
    }
    free(h_infinite);
}

