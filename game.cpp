#include <fstream>
#include <iostream>
#include <vector>
#include <stdlib.h>
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


	char buffer[1024*32];
	in.getline(buffer, 1024); // Discard "Parity" line

	while(in)
	{
		in.getline(buffer, 1024, '\n');
		if(!in)
			break;


        char* token = strtok(buffer, " ");
		int id = atoi(token);


        token = strtok(NULL, " ");
		int priority = atoi(token);
        if(priority < 0)
        {
			cout << "Priorities must be non-negative!" << endl;
			exit(1);
        }


        token = strtok(NULL, " ");
		int player = atoi(token);


        int assigned_id = add_vertex(player, priority);
		if(id != assigned_id)
		{
			cout << "Vertices must be listed in order starting at 0!" << endl;
			exit(1);
		}


        // Get edges
		char* edge = strtok(NULL, ""); // The rest of the buffer
        
        // Find the first ' ' or ';' or \0
        char* current = edge;
        while(true)
        {
            char c = *current;
            if(c == '\"' or c == ';' or c == 0)
            {
                *current = 0;
                break;
            }
            current++;
        }


        token = strtok(edge, ",");
		while(token != NULL)
		{
			int targetid = atoi(token);
			add_edge(assigned_id, targetid);
			token = strtok(NULL, ",");
		}

        if(edges[id].size() == 0)
        {
            cout << "Sinks are not allowed!" << endl;
            exit(1);
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
