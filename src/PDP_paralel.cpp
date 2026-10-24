#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <ctime>
#include <algorithm>
#include <cstdlib>
#include <omp.h>
using namespace std;

#define ENABLE_PRUNING

// Globalni vystupni promenne
double seq_time = 0.0;               // Sekvencni cas [s]
unsigned long long rec_calls = 0;    // Pocet volani rekurzivni funkce
int best_cut_weight = numeric_limits<int>::max(); // Minimalni vaha rezu
unsigned int num_solutions = 0;      // Pocet nalezenych reseni s minimalni vahou

// Globalni promenne pro graf a parametry ulohy
int n;      // pocet uzlu
int a;      // velikost mensi mnoziny
vector<vector<int>> graph;
vector<int> part; // part[i] = 0 => uzel v mnozine X, 1 => v mnozine Y

// Ulozeni vsech reseni s minimalni vahou (kopie pole prirazeni)
vector< vector<int> > best_solutions;

// Konstantni cutoff, do ktereho se vytvari OpenMP tasky (experimentujte s hodnotou)
const int TASK_CUTOFF = 10;

// Funkce pro vypocet poctu hran rezu a jejich celkove vahy pro dane reseni
void compute_cut_details(const vector<int>& sol, int &edgeCount, int &cutWeight) {
    edgeCount = 0;
    cutWeight = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if(sol[i] != sol[j]) { // hrana je mezi ruznymi skupinami
                edgeCount++;
                cutWeight += graph[i][j];
            }
        }
    }
}

// Paralelni DFS – rekurzivni prohledavani s OpenMP tasky
// idx        : index aktualne zpracovavaneho uzlu
// count0     : pocet uzlu, ktere jsou v X (0)
// current_weight : aktualni soucet vah rezu
// local_part : lokální kopie prirazeni uzlu
void dfs_parallel(int idx, int count0, int current_weight, vector<int> local_part) {
    #pragma omp atomic
    rec_calls++;

    // Základní pripad: vsechny uzly jsou prirazeny
    if(idx == n) {
        // Odstraneni duplikaci: ulozime pouze reseni, kdy je uzel 0 v X
        if(local_part[0] != 0)
            return;
        if(count0 == a || count0 == n - a) {
            #pragma omp critical
            {
                if(current_weight < best_cut_weight) {
                    best_cut_weight = current_weight;
                    num_solutions = 1;
                    best_solutions.clear();
                    best_solutions.push_back(local_part);
                } else if(current_weight == best_cut_weight) {
                    num_solutions++;
                    best_solutions.push_back(local_part);
                }
            }
        }
        return;
    }

#ifdef ENABLE_PRUNING
    // Pruning podminky: pokud zbyle uzly nemohou dosahnout potrebne velikosti nebo pokud
    // dolni odhad dodatecne vahy rezu znizuje smysl prohledavani
    if(count0 + (n - idx) < a) return;
    if(count0 > n - a) return;
    
    int lb = 0;
    for (int i = idx; i < n; i++) {
        int addToX = 0, addToY = 0;
        for (int j = 0; j < idx; j++) {
            if(local_part[j] == 0)
                addToY += graph[i][j];
            else if(local_part[j] == 1)
                addToX += graph[i][j];
        }
        lb += min(addToX, addToY);
    }
    if(current_weight + lb >= best_cut_weight)
        return;
#endif

    // Pokud jsme nad urcenou hranici, vytvori se nove tasky, jinak se DFS provadi sekvenicne
    if(idx < TASK_CUTOFF) {
        // Vetev: prirazeni uzlu idx do X (0)
        vector<int> local_part_copy = local_part;
        local_part_copy[idx] = 0;
        int addWeight = 0;
        for (int j = 0; j < idx; j++) {
            if(local_part[j] == 1)
                addWeight += graph[idx][j];
        }
        #pragma omp task firstprivate(idx, count0, current_weight, local_part_copy)
        {
            dfs_parallel(idx + 1, count0 + 1, current_weight + addWeight, local_part_copy);
        }

        // Vetev: prirazeni uzlu idx do Y (1)
        local_part_copy = local_part;
        local_part_copy[idx] = 1;
        addWeight = 0;
        for (int j = 0; j < idx; j++) {
            if(local_part[j] == 0)
                addWeight += graph[idx][j];
        }
        #pragma omp task firstprivate(idx, count0, current_weight, local_part_copy)
        {
            dfs_parallel(idx + 1, count0, current_weight + addWeight, local_part_copy);
        }
        #pragma omp taskwait
    } else {
        // Sekvenční rekurze na nižších úrovních
        local_part[idx] = 0;
        int addWeight = 0;
        for (int j = 0; j < idx; j++) {
            if(local_part[j] == 1)
                addWeight += graph[idx][j];
        }
        dfs_parallel(idx + 1, count0 + 1, current_weight + addWeight, local_part);
        
        local_part[idx] = 1;
        addWeight = 0;
        for (int j = 0; j < idx; j++) {
            if(local_part[j] == 0)
                addWeight += graph[idx][j];
        }
        dfs_parallel(idx + 1, count0, current_weight + addWeight, local_part);
    }
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        cerr << "Usage: " << argv[0] << " input_file [a_value]" << endl;
        return 1;
    }
    
    ifstream infile(argv[1]);
    if(!infile) {
        cerr << "Could not open file: " << argv[1] << endl;
        return 1;
    }
    
    infile >> n;
    if(argc >= 3)
        a = atoi(argv[2]);
    else
        a = 5;
    
    part.resize(n, -1);
    graph.assign(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            infile >> graph[i][j];
    infile.close();
    
    // Pro symetricke pripady (kdy je a == n/2) fixujeme uzel 0 v X
    bool symmetricCase = ((n % 2 == 0) && (a == n/2));
    vector<int> local_part = part;
    if(symmetricCase)
        local_part[0] = 0;
    
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            if(symmetricCase)
                dfs_parallel(1, 1, 0, local_part);
            else
                dfs_parallel(0, 0, 0, local_part);
        }
    }
    double end = omp_get_wtime();
    seq_time = end - start;
    
    cout << "Sekvencni cas [s]: " << seq_time << endl;
    cout << "# volani rekurzivni funkce: " << rec_calls << endl;
    cout << "Minimalni vaha rezu: " << best_cut_weight << endl;
    cout << "# reseni: " << num_solutions << endl;
    
    cout << "\nVypis vsech hranovych rezu s minimalni vahou:" << endl;
    for (size_t sol = 0; sol < best_solutions.size(); sol++) {
        vector<int>& solAssignment = best_solutions[sol];
        vector<int> X, Y;
        for (int i = 0; i < n; i++) {
            if(solAssignment[i] == 0)
                X.push_back(i);
            else
                Y.push_back(i);
        }
        int edgeCount = 0, computedCutWeight = 0;
        compute_cut_details(solAssignment, edgeCount, computedCutWeight);
        
        cout << "Reseni " << sol + 1 << ":" << endl;
        cout << "  X: ";
        for (int node : X)
            cout << node << " ";
        cout << "\n  Y: ";
        for (int node : Y)
            cout << node << " ";
        cout << "\n  Pocet hran rezu: " << edgeCount << endl;
        cout << "  Vaha rezu: " << computedCutWeight << endl << endl;
    }
    
    return 0;
}
