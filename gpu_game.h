#ifndef GPU_GAME
#define GPU_GAME

#include "game.h"

class GPUGame
{
public:
    Game& g;

    int num_vert;
    int num_edge;

    int* player;
    int* priority;
    int* vertices;
    int* edges;         // The target of each edge

    GPUGame(Game& g);
    ~GPUGame();
};

#endif
