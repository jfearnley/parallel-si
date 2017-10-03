#include "game.h"
#include "gpu_game.h"
#include "gpu_parity.h"
#include "gpu_list_ranking.h"
#include "cpu_parity.h"
#include "cpu_bf.h"
#include "cpu_list_ranking.h"
#include "cpu_vj.h"
#include "cpu_bv.h"
#include "valuation.h"
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <stdlib.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>

using namespace std;

void solve_si(StrategyImprovement& si, int* br_count, int* iter_count, bool reset)
{
    si.init_strat();

    bool first = true;
    while(true)
    {
        if(not first and reset)
            si.reset_opponent_strategy();
        while(true)
        {
            if(first)
                si.compute_first_val();
            else
                si.compute_valuation();

            (*br_count)++;
            si.mark_solved(1);

            if(si.switch_strategy(1) == 0)
                break;
        }


        first = false;

        si.mark_solved(0);
        (*iter_count)++;

        int total_switched = si.switch_strategy(0);
        //cout << "Total switched: " << total_switched << endl;
        if(total_switched == 0)
            break;
    }

}

template <typename T> void solve_bf(CPU_BellmanFord<T>& si, int* br_count, int* iter_count, bool reset)
{
    si.init_strat();

    while(true)
    {

        si.best_response(br_count);

        si.mark_solved(0);

        (*iter_count)++;

        int total_switched = si.switch_strategy(0);
        //cout << "Total switched: " << total_switched << endl;
        if(total_switched == 0)
            break;
    }

    //cout << "Solved after " << iter_count << " iterations" << endl; 
}

// Run a timed test of a strategy improvement algorithm
template <typename T> double test(T& si, void (*f)(T&, int*, int*, bool), bool reset)
{
    timeval start;
    gettimeofday(&start, nullptr);
    int iter_count = 0;
    int br_count = 0;

    (*f)(si, &br_count, &iter_count, reset);

    timeval end;
    gettimeofday(&end, nullptr);

    double solve_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000000.0;

    cout << std::fixed << solve_time << " " << iter_count << " " << br_count << endl;
}

// Checks that alg1 and alg2 give the same answer
bool verify(StrategyImprovement& alg1, StrategyImprovement& alg2)
{
    for(int p = 0; p < 2; p++)
    {
        vector<int> alg1out;
        vector<int> alg2out;

        alg1.get_winning(alg1out, p);
        alg2.get_winning(alg2out, p);

        //cout << "winning set"; 
        //for(auto v : alg1out)
            //cout << v << " ";
        //cout << endl;


        if(alg1out.size() != alg2out.size())
        {
            return false;
        }

        for(int i = 0; i < alg1out.size(); i++)
            if(alg1out[i] != alg2out[i])
                return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        cout << "usage: " << argv[0] << " [algorithm] [game]" << endl << endl;
        cout << "    algorithms: cpu, cpubf, cpu-reset, cpubv, cpubv, gpu, gpulist" << endl << endl;
        return 1;
    }

    string algorithm = argv[1];
    string filename = argv[2];

    Game g;

    try {
        ifstream file(filename, ios_base::binary);
        boost::iostreams::filtering_istream in;
        if (boost::algorithm::ends_with(filename, ".bz2")) in.push(boost::iostreams::bzip2_decompressor());
        if (boost::algorithm::ends_with(filename, ".gz")) in.push(boost::iostreams::gzip_decompressor());
        in.push(file);
        g.parse_pgsolver(in);
        file.close();
    } catch (const char *err) {
        std::cerr << "parsing error: " << err << std::endl;
        return -1;
    }

    if(algorithm == "gpu")
    {
        GPUGame gpu_g(g);
        GPUParity gpu(g, gpu_g);
        test<StrategyImprovement>(gpu, &solve_si, false);
    }
    else if(algorithm == "gpulist")
    {
        GPUGame gpu_g(g);
        GPUList gpu_list(g, gpu_g);
        test<StrategyImprovement>(gpu_list, &solve_si, false);
    }
    else if(algorithm == "cpu")
    {
        CPUParity cpu = CPUParity(g);
        test<StrategyImprovement>(cpu, &solve_si, false);
    }
    else if(algorithm == "cpu-reset")
    {
        CPUParity cpu = CPUParity(g);
        test<StrategyImprovement>(cpu, &solve_si, true);
    }
    else if(algorithm == "cpubv")
    {
        CPUBV<ArrayVal> cpu = CPUBV<ArrayVal>(g);
        test<StrategyImprovement>(cpu, &solve_si, false);
    }
    //else if(algorithm == "cpubvmap")
    //{
        //CPUBV<MapVal> cpu = CPUBV<MapVal>(g);
        //test<StrategyImprovement>(cpu, &solve_si);
        ////CPUBV<ArrayVal> cpu2 = CPUBV<ArrayVal>(g);
        ////test<StrategyImprovement>(cpu2, &solve_si);
        ////cout << verify(cpu, cpu2) << endl;
    //}
    else if(algorithm == "cpubf")
    {
        CPU_BellmanFord<ArrayVal> cpu_bf = CPU_BellmanFord<ArrayVal>(g);
        test(cpu_bf, &solve_bf, false);

    }
    else if(algorithm == "cpulist")
    {
        CPUList<ArrayVal> cpu_list = CPUList<ArrayVal>(g, 8);
        test<StrategyImprovement>(cpu_list, &solve_si, false);


    }
    //else if(algorithm == "cpuvj")
    //{
        //Game g2 = g;
        //g2.unique_priority_transform();
        //CPUVJ<MapVal> cpu_vj = CPUVJ<MapVal>(g2);
        //test<StrategyImprovement>(cpu_vj, &solve_si);

        ////GPUGame gpu_g(g);
        ////GPUList gpu_list(g, gpu_g);
        ////test<StrategyImprovement>(gpu_list, &solve_si);
        ////cout << verify(cpu_vj, gpu_list) << endl;
    //}
    else
    {
        cout << "unknown algorithm: " << algorithm << endl;
    }

}
