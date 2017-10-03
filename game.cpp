#include <fstream>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <sstream>
#include <string.h>

#include "game.h"

using namespace std;

Game::Game()
{
    n_vertices = 0;
    max_pri = 0;
}

int Game::add_vertex(int player, int priority)
{
    int new_id = n_vertices;
    n_vertices++;

    edges.push_back( vector<int>() );
    this->player.push_back(player);
    this->priority.push_back(priority);

    if(priority > max_pri)
        max_pri = priority;

    return new_id;
}

void Game::add_edge(int from, int to)
{
    edges[from].push_back(to);
}

void Game::parse_pgsolver(string filename)
{
	ifstream in(filename.c_str(), ifstream::in);

	if(!in)
	{
		cout << "Unable to open file: " << filename << endl;
		exit(1);
	}

    parse_pgsolver(in);
}

void Game::parse_pgsolver(istream &in)
{
    string line;

    // discard parity line
    if (!getline(in, line)) return;

    while (getline(in, line)) {
        stringstream ss(line);
        string token;

        // ignore empty line
        if (!(ss >> token)) continue;

        // parse id, priority, owner
        int id = stoi(token);
        if (id < 0) {
            throw "invalid id";
        }

        int priority;
        if (!(ss >> priority)) {
            throw "missing priority";
        }

        int player;
        if (!(ss >> player)) {
            throw "missing player";
        }

        if (player != 0 && player != 1) {
            throw "invalid player";
        }

        int assigned_id = add_vertex(player, priority);
		if(id != assigned_id) {
            throw "vertices must be listed in order starting at 0!";
		}

        // parse successors and optional label
        for (;;) {
            int to;
            if (!(ss >> to)) throw "missing successor";

            add_edge(id, to);

            char ch;
            if (!(ss >> ch)) throw "missing ; to end line";

            if (ch != ',') break;
        }

        if(edges[id].size() == 0) {
            throw "sinks are not allowed!";
        }
    }
}

// Transform the game so that every priority is assigned to at most one vertex
void Game::unique_priority_transform()
{
    int current_pri = 0; // An even number that is larger than all priorities assigend so far

    for(int i = 0; i < max_pri; i++)
    {
        // Process the vertices with original-priority i
        for(int v = 0; v < n_vertices; v++)
        {
            int p = priority[v];
            if(p == i)
            {
                if(i % 2 == 0)
                    priority[v] = current_pri;
                else
                    priority[v] = current_pri + 1;

                current_pri += 2;
                
            }
        }
    }

    max_pri = current_pri;
}
