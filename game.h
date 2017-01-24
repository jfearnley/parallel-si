#ifndef GAME_H
#define GAME_H

#include <vector>
#include <string>

class GPUGame;

class Game
{
    friend class GPUGame;

    int n_vertices;
    int max_pri;
    std::vector< std::vector<int> > edges;
    std::vector< int > player;
    std::vector< int > priority;
public:
    Game();

    int add_vertex(int player, int priority);
    void add_edge(int from, int to);
    void unique_priority_transform();

    int get_max_pri() {return max_pri;};
    int get_n_vertices() {return n_vertices;};
    int get_player(int v) {return player[v];};
    int get_priority(int v) {return priority[v];};
    std::vector<int>& get_edges(int v) {return edges[v];};


    void parse_pgsolver(std::string filename);
};

#endif
