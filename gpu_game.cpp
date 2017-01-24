#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

#include "gpu_game.h"
#include "gpu_util.h"
#include "game.h"

using namespace std;

GPUGame::GPUGame(Game& g) : g(g)
{

    // Game data
    init_from_vector(&priority, g.priority);
    init_from_vector(&player, g.player);

    // Construct vertex and edge arrays
    vector<int> host_vertices;
    vector<int> host_edges;

    for(int v = 0; v < g.get_n_vertices(); v++)
    {
        host_vertices.push_back(host_edges.size()); // Location of first edge 

        for(int e : g.get_edges(v))
        {
            host_edges.push_back(e);
        }
    }
    // One last entry to give the end of the array. 
    // Makes it much easier to loop between vertices[i] and vertices[i+1] to 
    // get all edges from vertex[i]
    host_vertices.push_back(host_edges.size()); 

    init_from_vector(&vertices, host_vertices);
    init_from_vector(&edges, host_edges);

    num_vert = g.get_n_vertices();
    num_edge = host_edges.size();
}

GPUGame::~GPUGame()
{
    cudaFree(player);
    cudaFree(priority);
}


