#include "gpu_list_ranking.h"
#include "gpu_util.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 128

using namespace std;


GPUList::GPUList(Game& g, GPUGame& g_dev) 
    : GPUParity(g, g_dev), g(g), g_dev(g_dev)
{
    int n = g.get_n_vertices();
    l = 2*n + 2; // The length of the list
    s = (l - 1) / 64 + 1; // The number of splitters 
    init_zero(&succ, l);
    init_zero(&scratch, s);

    init_zero(&split_succ, s);
    h_split_succ = (int*)malloc(s * sizeof(int));

    init_zero(&split_val, s * (g.get_max_pri() + 1));
    h_split_val = (int*)malloc(s * (g.get_max_pri() + 1) * sizeof(int));
    h_cum_val = (int*)malloc(s * (g.get_max_pri() + 1) * sizeof(int));

    init_zero(&split_inf, s);
    h_split_inf = (int*)malloc(s * sizeof(int));

    init_zero(&vert_list, n+1);


}

/************************ BUILD LIST ***************************************/

/*
 * Initalizes the succ array for d_build_list.
 *
 *  Node i has two entries:
 *  - succ[2i] holds the "down" direction
 *  - succ[2i + 1] holds the "up" direction
 *
 * The sink is represented by succ[2n] (down) and succ[2n+1] (up)
 *
 * Each "up" edge points to the corresponding "down" edge. The down edges will
 * be filled in by d_build_list.
 */
__global__ void d_init_list(int* succ, int* vert_list, int n)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= n+1)
        return;

    succ[2*idx + 1] = 2*idx;

    if(idx == n)
        succ[2*n] = -1; // The head of the list
    else
        vert_list[idx] = -1; // Initialize vert_list for later
        
}

/*
 *  Fill the succ array with a linked list that is an euler tour of the (pseudo)
 *  forest defined by strategy.
 *  
 */
__global__ void d_build_list(int* strategy, int* succ, int n)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= n)
        return;

    int next = strategy[idx];

    // Compute the address of the "up" node for next
    int next_up;
    if(next == -1)
        next_up = 2 * n + 1;
    else
        next_up = 2 * next + 1;

    int my_up = 2 * idx + 1;
    int my_down = 2*idx;

    int new_down = atomicExch(&succ[next_up], my_up);
    succ[my_down] = new_down;
}

/********************************* PICK SPLITTERS **************************/

__global__ void d_pick_splitters(int* succ, int* scratch, int l, int s)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= s)
        return;

    int splitter;
    if(idx == s-1)
        splitter = l - 1; // The head of the list is always a splitter
    else
        splitter = (int)(((float)(l-1) / (float)s) * (float)idx);

    scratch[idx] = succ[splitter];
    succ[splitter] = -100 -idx;
}

/****************************** TRAVERSE SUBLISTS **************************/

__device__ void inline d_init_val(int* val, int n_pris, int split, int thread_pri)
{
    int base_addr = split * n_pris;
    //if(thread_pri == 0)
        //for(int i = 0; i < n_pris; i++)
            //val[base_addr + i] = 0;
    val[base_addr + thread_pri] = 0;
}

__device__ void inline d_update_val(int* val, int* priority, int* val_scratch,
        int* vert_list, int current, int n_pris, int n_vertices, int split, int
        thread_pri)
{
    int vertex = current / 2;

    if(vertex == n_vertices or current == -1) // Sink vertex has no priority
        return;

    // Direction is -1 for down and +1 for up
    int direction = ((current % 2) * 2 - 1);

    // Add in this vertex's priority
    int p = priority[vertex];
    if(thread_pri == p)
    {
        *val += direction;
    }

    // If current is an up edge, update val_scratch
    if(current % 2 == 1)
    {
        val_scratch[vertex*n_pris + thread_pri] = *val;

        // If this is the up edge, store the location of the list
        if(thread_pri == 0)
            vert_list[vertex] = split;
    }
}

__global__ void d_traverse_sublists(int* succ, int* scratch,
        int* split_succ, int* val, int* priority, int* vert_list, int*
        val_scratch, int s, int n_pris, int n_pris_rounded, int n_vertices, int threads)
{

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= threads)
        return;


    int split_index = idx/n_pris_rounded;
    int thread_pri = idx % n_pris_rounded;

    if(thread_pri >= n_pris)
        return;

    while(split_index < s)
    {

        int cum_val = 0;

        // Get the first step of the list
        int current = scratch[split_index]; 

        while(current >= 0)
        {
            d_update_val(&cum_val, priority, val_scratch, vert_list, current, n_pris,
                    n_vertices, split_index, thread_pri);

            current = succ[current];
        }

        int base_addr = split_index * n_pris;
        val[base_addr + thread_pri] = cum_val;

        if(thread_pri == 0)
            split_succ[split_index] = current;

        split_index += threads/n_pris;
    }
}

/********************* PROCESS REDUCED ***********************************/

void process_reduced(int* split_succ, int* split_val, 
         int* cum_val, int* split_inf, int n_pris, int s)
{
    // Initialize the infinite array
    for(int i = 0; i < s; i++)
        split_inf[i] = 1;

    // Initialize cumval
    for(int i = 0; i < s * n_pris; i++)
        cum_val[i] = 0;

    // Walk the reduced list from the start
    int current = s-1; // Up edge of the sink

    int val[n_pris];
    for(int i = 0; i < n_pris; i++)
        val[i] = 0;

    while(current != -1)
    {
        // Copy the current val into the out array
        for(int i = 0; i < n_pris; i++)
        {
            cum_val[current*n_pris + i] = val[i];
        }

        // Add in the next segment
        for(int i = 0; i < n_pris; i++)
        {
            val[i] += split_val[current*n_pris + i];

        }

        // This vertex is not infinite
        split_inf[current] = 0;

        // Move to the next vertex
        int next = split_succ[current];
        if(next == -1) // The sink
            break;
        current = -(next+100); // -100 offset was used in pick_splitters
    }

}

/************************** BROADCAST VAL *********************************/

__global__ void d_broadcast_val(int* val, int* split_val, int* val_scratch, int*
        vert_list, int* split_infinite, int* infinite, int n_pris, int pri_mem)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= pri_mem)
        return;

    int vertex = idx / n_pris;
    int sublist = vert_list[vertex];

    if(infinite[vertex] == -1) // Odd wins this vertex, ignore it
        return;

    if(sublist == -1) // Vertex was not touched by any sublist
    {
        infinite[vertex] = 1;
        return;
    }

    int pri = idx % n_pris;

    if(split_infinite[sublist] == 0)
    {
        val[idx] = val_scratch[idx] + split_val[sublist*n_pris + pri];
        infinite[vertex] = 0;
        //printf("vertex %d, sublist %d\n", vertex, sublist);

    }
    else
    {
        infinite[vertex] = 1;
    }

}

void GPUList::compute_valuation()
{
    current_val = val1;

    int n = g.get_n_vertices();
    int n_pris = g.get_max_pri() + 1;

    int round_boundary = 4;
    int n_pris_rounded;
    if(n_pris % round_boundary == 0)
        n_pris_rounded = n_pris;
    else
        n_pris_rounded = (n_pris/round_boundary + 1) * round_boundary;


    // Build the successor array
    d_init_list <<< nblocks(n+1), BLOCK_SIZE >>> (succ, vert_list, n);
    d_build_list <<< nblocks(n), BLOCK_SIZE >>> (strategy, succ, n);

    // Pick the splitters
    d_pick_splitters <<< nblocks(s), BLOCK_SIZE >>> (succ, scratch, l, s);

    // Reduce the list
    int traverse_threads = 4* 4096 * 16 * n_pris_rounded;

    d_traverse_sublists <<< nblocks(traverse_threads), BLOCK_SIZE >>> (succ,
            scratch, split_succ, split_val, g_dev.priority,
            vert_list, val2, s, n_pris, n_pris_rounded, n, traverse_threads);


    // Copy reduced list to host and process
    int split_mem = s * sizeof(int);
    int split_pri_mem = s * n_pris * sizeof(int);

    gpu_assert(cudaMemcpy(h_split_succ, split_succ, split_mem, cudaMemcpyDeviceToHost));
    gpu_assert(cudaMemcpy(h_split_val, split_val, split_pri_mem, cudaMemcpyDeviceToHost));

    process_reduced(h_split_succ, h_split_val, h_cum_val, h_split_inf, n_pris, s);

    gpu_assert(cudaMemcpy(split_val, h_cum_val, split_pri_mem, cudaMemcpyHostToDevice));
    gpu_assert(cudaMemcpy(split_inf, h_split_inf, split_mem, cudaMemcpyHostToDevice));

    // Broadcast values out to the vertices
    int pri_mem = n * n_pris;
    d_broadcast_val <<< nblocks(pri_mem), BLOCK_SIZE >>> (val1, split_val,
     val2, vert_list, split_inf, infinite,  n_pris, pri_mem);

}

void GPUList::compute_first_val()
{
    GPUParity::compute_valuation();
}
