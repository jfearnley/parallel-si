#ifndef CPU_LIST_RANKING
#define CPU_LIST_RANKING

#include "si.h"
#include "cpu_bv.h"

template <typename T> class CPUList : public CPUBV<T>
{
    int n; // Number of vertices
    int s;
    int l; // Length of the list
    int p;
    int threads; // Number of threads to use
    int player;
    int total_switched;

    // See gpu_list_ranking.h for documentation
    int* succ; 
    int* vert_list;
    int* scratch;
    int* split_succ;
    int* split_inf;
    std::vector<T> split_val;
    std::vector<T> split_val_sum;
    std::vector<T> val_scratch;

    int start_n(int n, int tid);
    int end_n(int n, int tid);
    void init_list(int start, int end);
    void build_list(int start, int end);
    void pick_splitters(int start, int end);
    void update_val(T& cum_val, int current, int split);
    void traverse_sublists(int start, int end);
    void process_reduced();
    void broadcast_val(int start, int end);
    void switch_vertices(int start, int end);
    void launch(void (CPUList<T>::*f)(int, int), int n); // Dispatch threads to member function
public:
    CPUList(Game& g, int threads);

    virtual void compute_valuation();
    virtual void compute_first_val();
    virtual int switch_strategy(int player);
};

#endif
