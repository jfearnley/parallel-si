#ifndef VALUATION_H
#define VALUATION_H

#include <string>
#include <vector>
#include <map>

#include "game.h"

class ArrayVal 
{
    Game* g;
    std::vector<int> val;
public:
    ArrayVal(Game& g); // Initialize the valuation to 0

    void zero();
    void add_to_pri(int x, int p);
    void init_from_vertex(int v);
    void add_vertex(int v);
    void add_val(ArrayVal& other);
    void copy_and_add(const ArrayVal& other, int v);
    void copy_and_add(ArrayVal& other, ArrayVal& other2);
    std::string to_string();

    int compare_valuation(const ArrayVal& other);

    bool operator ==(const ArrayVal& other);
    bool operator !=(const ArrayVal& other) {return !(*this == other);};
};

class MapVal
{
    Game* g;
    std::map<int, int> val;
public:
    MapVal(Game& g); 

    void zero();
    void add_to_pri(int x, int p);
    void init_from_vertex(int v);
    void add_vertex(int v);
    void add_val(MapVal& other);
    void copy_and_add(const MapVal& other, int v);
    void copy_and_add(MapVal& other, MapVal& other2);
    std::string to_string();

    int compare_valuation(MapVal& other);

    bool operator ==(const MapVal& other);
    bool operator !=(const MapVal& other) {return !(*this == other);};
};


template<typename T> class VJVal 
{
    Game* g;
    int p;
    T s;
    int d;
public:
    VJVal(Game& g);

    void copy_p(const VJVal& other);
    void set_p(int p) { this->p = p; };
    int get_p() { return p; };

    void zero_sd() { s.zero(); d=0; };
    void copy_and_add(const VJVal& other, int v);


    std::string to_string();

    int compare_to(VJVal& other);
};


#endif 
