#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cstdlib>
#include <algorithm>
using namespace std;

// MPI tagy pro zprávy
#define TAG_TASK      0
#define TAG_RESULT    1
#define TAG_TERMINATE 2

// Povolení dolního ořezání
#define ENABLE_PRUNING

// Hloubka pro vytváření OpenMP tasků
const int TASK_CUTOFF = 10;

// Globální proměnné v slave pro ukládání nejlepšího řešení
static int slave_best_weight;
static vector<int> slave_best_assignment;

// Subproblem struct
struct Subproblem {
    int idx;
    int count0;
    int current_weight;
    vector<int> partial;
};

// Sekvenční DFS pro malé subproblémy
void dfs_sequential_MPI(int idx, int count0, int current_weight, vector<int>& part,
                        int n, int a, const vector<vector<int>>& graph,
                        int &best_cut_weight, vector<int>& best_solution,
                        unsigned long long &rec_calls) {
    rec_calls++;
    if (idx == n) {
        if (part[0] != 0) return;
        if (count0 == a || count0 == n - a) {
            if (current_weight < best_cut_weight) {
                best_cut_weight = current_weight;
                best_solution = part;
            }
        }
        return;
    }
#ifdef ENABLE_PRUNING
    if (count0 + (n - idx) < a) return;
    if (count0 > n - a) return;
    int lb = 0;
    for (int i = idx; i < n; ++i) {
        int addToX = 0, addToY = 0;
        for (int j = 0; j < idx; ++j) {
            if (part[j] == 0) addToY += graph[i][j];
            else              addToX += graph[i][j];
        }
        lb += min(addToX, addToY);
    }
    if (current_weight + lb >= best_cut_weight) return;
#endif
    // X branch
    {
        vector<int> p0 = part;
        p0[idx] = 0;
        int w0 = current_weight;
        for (int j = 0; j < idx; ++j)
            if (p0[j] == 1)
                w0 += graph[idx][j];
        dfs_sequential_MPI(idx+1, count0+1, w0, p0,
                           n, a, graph, best_cut_weight, best_solution, rec_calls);
    }
    // Y branch
    {
        vector<int> p1 = part;
        p1[idx] = 1;
        int w1 = current_weight;
        for (int j = 0; j < idx; ++j)
            if (p1[j] == 0)
                w1 += graph[idx][j];
        dfs_sequential_MPI(idx+1, count0, w1, p1,
                           n, a, graph, best_cut_weight, best_solution, rec_calls);
    }
}

// Generování subproblémů do fronty použitím DFS až do hloubky cutoff
void generate_subproblems(const vector<vector<int>>& graph,
                          int idx, int count0, int current_weight,
                          vector<int>& partial,
                          vector<Subproblem>& queue,
                          int cutoff, int n) {
    if (idx == cutoff) {
        Subproblem sp{idx, count0, current_weight, partial};
        queue.push_back(sp);
        return;
    }
    // X branch
    {
        vector<int> p0 = partial;
        p0[idx] = 0;
        int w0 = current_weight;
        for (int j = 0; j < idx; ++j)
            if (partial[j] == 1)
                w0 += graph[idx][j];
        generate_subproblems(graph, idx+1, count0+1, w0, p0,
                             queue, cutoff, n);
    }
    // Y branch
    {
        vector<int> p1 = partial;
        p1[idx] = 1;
        int w1 = current_weight;
        for (int j = 0; j < idx; ++j)
            if (partial[j] == 0)
                w1 += graph[idx][j];
        generate_subproblems(graph, idx+1, count0, w1, p1,
                             queue, cutoff, n);
    }
}

// Paralelní DFS s OpenMP tasky pro slave
void dfs_parallel(int idx, int count0, int current_weight,
                  vector<int> local_part,
                  int n, int a, const vector<vector<int>>& graph) {
    if (idx == n) {
        if (local_part[0] == 0 && (count0 == a || count0 == n - a)) {
            #pragma omp critical
            if (current_weight < slave_best_weight) {
                slave_best_weight = current_weight;
                slave_best_assignment = local_part;
            }
        }
        return;
    }
#ifdef ENABLE_PRUNING
    if (count0 + (n - idx) < a) return;
    if (count0 > n - a) return;
    int lb = 0;
    for (int i = idx; i < n; ++i) {
        int addX = 0, addY = 0;
        for (int j = 0; j < idx; ++j) {
            if (local_part[j] == 0) addY += graph[i][j];
            else                    addX += graph[i][j];
        }
        lb += min(addX, addY);
    }
    if (current_weight + lb >= slave_best_weight) return;
#endif
    if (idx < TASK_CUTOFF) {
        vector<int> p0 = local_part;
        p0[idx] = 0;
        int w0 = current_weight;
        for (int j = 0; j < idx; ++j)
            if (local_part[j] == 1)
                w0 += graph[idx][j];
        #pragma omp task firstprivate(p0, w0, idx, count0)
        dfs_parallel(idx+1, count0+1, w0, p0, n, a, graph);

        vector<int> p1 = local_part;
        p1[idx] = 1;
        int w1 = current_weight;
        for (int j = 0; j < idx; ++j)
            if (local_part[j] == 0)
                w1 += graph[idx][j];
        #pragma omp task firstprivate(p1, w1, idx, count0)
        dfs_parallel(idx+1, count0, w1, p1, n, a, graph);

        #pragma omp taskwait
    } else {
        local_part[idx] = 0;
        int w0 = current_weight;
        for (int j = 0; j < idx; ++j)
            if (local_part[j] == 1)
                w0 += graph[idx][j];
        dfs_parallel(idx+1, count0+1, w0, local_part, n, a, graph);

        local_part[idx] = 1;
        int w2 = current_weight;
        for (int j = 0; j < idx; ++j)
            if (local_part[j] == 0)
                w2 += graph[idx][j];
        dfs_parallel(idx+1, count0, w2, local_part, n, a, graph);
    }
}

// Wrapper pro spuštění paralelního DFS v slave
void handle_task_parallel(int idx, int count0, int current_weight,
                          vector<int>& partial,
                          int n, int a, const vector<vector<int>>& graph) {
    slave_best_weight = numeric_limits<int>::max();
    slave_best_assignment = partial;
    #pragma omp parallel
    #pragma omp single nowait
    dfs_parallel(idx, count0, current_weight, partial, n, a, graph);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n, a;
    vector<vector<int>> graph;

    if (my_rank == 0) {
        // Master: načtení dat
        if (argc < 2) {
            cerr << "Usage: " << argv[0] << " input_file [a_value]" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        ifstream infile(argv[1]);
        infile >> n;
        a = (argc >= 3 ? atoi(argv[2]) : 5);
        graph.assign(n, vector<int>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                infile >> graph[i][j];
    }

    // Broadcast vstupních dat
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&a, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank != 0) graph.assign(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        MPI_Bcast(graph[i].data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        // Master: počet procesorů
        cout << "Master: Pocet procesoru = " << num_procs << endl;

        // Generování subproblémů
        vector<int> initial(n, -1);
        bool symmetricCase = ((n % 2 == 0) && (a == n/2));
        if (symmetricCase) initial[0] = 0;
        int start_idx    = symmetricCase ? 1 : 0;
        int start_count0 = symmetricCase ? 1 : 0;
        vector<Subproblem> queue;
        generate_subproblems(graph, start_idx, start_count0, 0,
                             initial, queue, TASK_CUTOFF, n);
        cout << "Master: Vygenerovano " << queue.size()
             << " subproblemu k rozeslani." << endl;

        // Distribuce a sběr
        int num_tasks      = queue.size();
        int tasks_sent     = 0;
        int tasks_completed= 0;
        int global_best    = numeric_limits<int>::max();
        vector<int> global_best_assignment;
        MPI_Status status;
        
        // Pošleme první to každého slave
        for (int dest = 1; dest < num_procs && tasks_sent < num_tasks; ++dest) {
            auto &sp = queue[tasks_sent++];
            vector<int> msg(3 + n);
            msg[0] = sp.idx;
            msg[1] = sp.count0;
            msg[2] = sp.current_weight;
            for (int i = 0; i < n; ++i)
                msg[3 + i] = sp.partial[i];
            MPI_Send(msg.data(), 3 + n, MPI_INT,
                     dest, TAG_TASK, MPI_COMM_WORLD);
        }
        // Dynamická distribuce
        while (tasks_completed < num_tasks) {
            vector<int> res(1 + n);
            MPI_Recv(res.data(), 1 + n, MPI_INT,
                     MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
            int sender    = status.MPI_SOURCE;
            int local_best= res[0];
            if (local_best < global_best) {
                global_best = local_best;
                global_best_assignment.assign(res.begin()+1, res.end());
            }
            tasks_completed++;
            if (tasks_sent < num_tasks) {
                auto &sp = queue[tasks_sent++];
                vector<int> msg(3 + n);
                msg[0] = sp.idx;
                msg[1] = sp.count0;
                msg[2] = sp.current_weight;
                for (int i = 0; i < n; ++i)
                    msg[3 + i] = sp.partial[i];
                MPI_Send(msg.data(), 3 + n, MPI_INT,
                         sender, TAG_TASK, MPI_COMM_WORLD);
            }
        }
        // Signál ukončení
        vector<int> termMsg(3 + n, -1);
        for (int dest = 1; dest < num_procs; ++dest)
            MPI_Send(termMsg.data(), 3 + n, MPI_INT,
                     dest, TAG_TERMINATE, MPI_COMM_WORLD);

        // Výpis výsledků
        cout << "Master: Final best cut weight = " << global_best << endl;
        cout << "Master: Best partition = ";
        for (int v : global_best_assignment)
            cout << v << " ";
        cout << endl;
    } else {
        // Slave: příjem úloh a zpracování
        while (true) {
            MPI_Status status;
            vector<int> buf(3 + n);
            MPI_Recv(buf.data(), 3 + n, MPI_INT,
                     0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_TERMINATE) {
                cout << "Slave " << my_rank
                     << ": Received terminate, exiting." << endl;
                break;
            }
            // Deserializace
            int idx         = buf[0];
            int count0      = buf[1];
            int w           = buf[2];
            vector<int> partial(n);
            for (int i = 0; i < n; ++i)
                partial[i] = buf[3 + i];
            cout << "Slave " << my_rank
                 << ": Received task idx=" << idx
                 << " count0=" << count0
                 << " w=" << w << endl;
            // Paralelní DFS
            handle_task_parallel(idx, count0, w,
                                 partial, n, a, graph);
            cout << "Slave " << my_rank
                 << ": Completed local best = " << slave_best_weight
                 << endl;
            // Odeslání výsledku
            vector<int> res(1 + n);
            res[0] = slave_best_weight;
            for (int i = 0; i < n; ++i)
                res[1 + i] = slave_best_assignment[i];
            MPI_Send(res.data(), 1 + n, MPI_INT,
                     0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    return 0;
}

