
#include <iostream>

#include "valuation.h"

using namespace std;

/********************************** ArrayVal ********************************/

ArrayVal::ArrayVal(Game& g) : g(&g)
{
    int p = g.get_max_pri() + 1;
    for(int i = 0; i < p; i++)
        val.push_back(0);
}

void ArrayVal::zero()
{
    for(int i = 0; i < val.size(); i++)
        val[i] = 0;
}

void ArrayVal::add_to_pri(int x, int p)
{
    if(p > val.size() or p < 0)
        cout << "priority is " << p << endl;
    val[p] += x;
}

void ArrayVal::init_from_vertex(int v)
{
    for(int i = 0; i < val.size(); i++)
        val[i] = 0;
    val[g->get_priority(v)] = 1;
}

void ArrayVal::add_vertex(int v)
{
    val[g->get_priority(v)] += 1;
}

void ArrayVal::add_val(ArrayVal& other)
{
    for(int i = 0; i < val.size(); i++)
        val[i] += other.val[i];
}

void ArrayVal::copy_and_add(const ArrayVal& other, int v)
{
    for(int i = 0; i < val.size(); i++)
        val[i] = other.val[i];
    add_vertex(v);
}

void ArrayVal::copy_and_add(ArrayVal& other, ArrayVal& other2)
{
    for(int i = 0; i < val.size(); i++)
        val[i] = other.val[i] + other2.val[i];
}

string ArrayVal::to_string()
{
    string out = "[";
    for(int i : val)
        out += std::to_string(i) + " ";
    out += "]";
    return out;
}

int inline compare_valuation_counts(int maxdiff, int v_count, int u_count)
{
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

int ArrayVal::compare_valuation(const ArrayVal& other)
{
    auto& other_val = other.val;

    // Find the maxdiff
    int maxdiff = -1;
    for(int i = val.size() - 1; i >= 0; i--)
    {
        if(val[i] != other_val[i])
        {
            maxdiff = i;
            break;
        }
    }

    if(maxdiff == -1) // Both valuations were identical
        return 0;

    int v_count = val[maxdiff];
    int u_count = other_val[maxdiff];

    return compare_valuation_counts(maxdiff, v_count, u_count);
}

bool ArrayVal::operator==(const ArrayVal& other)
{
    return val == other.val;
}

/***************************** MAP VAL **************************************/


MapVal::MapVal(Game& g) : g(&g)
{
}

void MapVal::zero()
{
    val.clear();
}

void MapVal::add_to_pri(int x, int p)
{
    val[p] += x;
}

void MapVal::init_from_vertex(int v)
{
    val.clear();
    val[g->get_priority(v)] = 1;
}

void MapVal::add_vertex(int v)
{
    val[g->get_priority(v)] += 1;
}

void MapVal::add_val(MapVal& other)
{
    for(auto& i : other.val)
        val[i.first] += other.val[i.second];
}

void MapVal::copy_and_add(const MapVal& other, int v)
{
    val = other.val;
    add_vertex(v);
}

void MapVal::copy_and_add(MapVal& other, MapVal& other2)
{
    val = other.val;

    for(auto& i : other2.val)
        val[i.first] += i.second;
}

string MapVal::to_string()
{
    string out = "[";
    for(auto& i : val)
        out += std::to_string(i.first) + ": " +  std::to_string(i.second) + " ";
    out += "]";
    return out;
}

int MapVal::compare_valuation(MapVal& other)
{
    map<int, int>& other_val = other.val;

    // Find the maxdiff
    int maxdiff = -1;
    auto our_iter = val.rbegin();
    auto other_iter = other_val.rbegin();

    // Deal with the case where at least one of the valuations is empty
    if(val.size() == 0 and other_val.size() == 0)
        return 0;
    else if(other_iter == other_val.rend())
        maxdiff = our_iter->first;
    else if(our_iter == val.rend())
        maxdiff = other_iter->first;

    if (maxdiff == -1)
    {
        while(true)
        {
            int our_pri = our_iter->first; 
            int other_pri = other_iter->first;

            // Detect if one list has a priority that the other doesn't
            if(our_pri > other_pri)
            {
                maxdiff = our_pri; break;
            }
            if(other_pri > our_pri)
            {
                maxdiff = other_pri; break;
            }

            // Both priorities are the same, now check if they have the same count
            int our_count = our_iter->second;
            int other_count = other_iter->second;

            if(our_count > other_count)
            {
                maxdiff = our_pri; break;
            }
            if(other_count > our_count)
            {
                maxdiff = other_pri; break;
            }

            // Both counts are the same, advance the iterators
            our_iter++;
            other_iter++;

            // If one has ended, then the other is the maxdiff
            if(our_iter != val.rend() and other_iter == other_val.rend())
            {
                maxdiff = our_iter->first; break;
            }
            if(our_iter == val.rend() and other_iter != other_val.rend())
            {
                maxdiff = other_iter->first; break;
            }

            // If both have ended, then both valuations are the same
            if(our_iter == val.rend() and other_iter == other_val.rend())
                return 0;
        }
    }

    // Compare the counts
    int v_count = 0;
    int u_count = 0;
    if(val.count(maxdiff) == 1)
        v_count = val[maxdiff];
    if(other_val.count(maxdiff) == 1)
        u_count = other_val[maxdiff];

    return compare_valuation_counts(maxdiff, v_count, u_count);

}

bool MapVal::operator==(const MapVal& other)
{
    return val == other.val;
}

/*************************** VJ VAL ******************************************/

template <typename T> VJVal<T>::VJVal(Game& g) : g(&g), s(g)
{
    p = 0;
    d = 0;
}

template <typename T> void VJVal<T>::copy_p(const VJVal& other)
{
    p = other.p;
}

template <typename T> string VJVal<T>::to_string()
{
    string out = "{";
    out += std::to_string(p) + " ";
    out += s.to_string() + " ";
    out += std::to_string(d); 
    return out + "}";

}

template <typename T> void VJVal<T>::copy_and_add(const VJVal& other, int v)
{
    if(g->get_priority(v) > p)
        s.copy_and_add(other.s, v);
    else
        s = other.s;

    d = other.d + 1;
}

template <typename T> int VJVal<T>::compare_to(VJVal& other)
{
    int our_parity = p % 2;
    int other_parity = other.p % 2;

    // Compare p
    if (p != other.p)
    {
        if(our_parity == 0 and other_parity == 0)
        {
            if(p > other.p)
                return 1;
            if(p < other.p)
                return -1;
        }
        if(our_parity == 1 and other_parity == 1)
        {
            if(p > other.p)
                return -1;
            if(p < other.p)
                return 1;
        }
        if(our_parity == 0 and other_parity == 1)
            return 1;
        if(our_parity == 1 and other_parity == 0)
            return -1;
    }


    // Compare s
    if(s != other.s)
    {
        return s.compare_valuation(other.s);
    }


    // Compare d
    if(d == other.d)
        return 0;


    if(our_parity == 0)
    {
        if(d < other.d)
            return 1;
        if(other.d < d)
            return -1;
    }
    if(our_parity == 1)
    {
        if(d < other.d)
            return -1;
        if(other.d < d)
            return 1;
    }

}

template class VJVal<ArrayVal>;
template class VJVal<MapVal>;
