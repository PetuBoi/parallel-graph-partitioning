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

// Globální proměnné
double par_time = 0.0;               // Měřený paralelní čas [s]
unsigned long long rec_calls = 0;    // Počet volání rekurzivní funkce
int best_cut_weight = numeric_limits<int>::max(); // Nejmenší váha řezu
unsigned int num_solutions = 0;      // Počet nalezených řešení s minimální vahou

int n;      // Počet uzlů
int a;      // Velikost menší množiny
vector<vector<int>> graph;
vector<vector<int>> best_solutions;

// Struktura pro uchování subproblému
struct Subproblem {
    int idx;              // Aktuální index, od kterého pokračujeme
    int count0;           // Počet uzlů v množině X (0) dosud
    int current_weight;   // Dosavadní váha řezu
    vector<int> partial;  // Částečné přiřazení uzlů
};

// Funkce pro výpočet detailů řezu (počet hran a celková váha)
void compute_cut_details(const vector<int>& sol, int &edgeCount, int &cutWeight) {
    edgeCount = 0;
    cutWeight = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if(sol[i] != sol[j]) { // hrana mezi různými skupinami
                edgeCount++;
                cutWeight += graph[i][j];
            }
        }
    }
}

// Sekvenční DFS, která dokončí prohledávání od zadaného subproblému
void dfs_sequential(int idx, int count0, int current_weight, vector<int>& partial) {
    #pragma omp atomic
    rec_calls++;

    if(idx == n) {
        // Odstranění duplikací: uložíme pouze řešení, kdy je uzel 0 v X
        if(partial[0] != 0)
            return;
        if(count0 == a || count0 == n - a) {
            #pragma omp critical
            {
                if(current_weight < best_cut_weight) {
                    best_cut_weight = current_weight;
                    num_solutions = 1;
                    best_solutions.clear();
                    best_solutions.push_back(partial);
                } else if(current_weight == best_cut_weight) {
                    num_solutions++;
                    best_solutions.push_back(partial);
                }
            }
        }
        return;
    }

#ifdef ENABLE_PRUNING
    if(count0 + (n - idx) < a) return;
    if(count0 > n - a) return;
    int lb = 0;
    for (int i = idx; i < n; i++) {
        int addToX = 0, addToY = 0;
        for (int j = 0; j < idx; j++) {
            if(partial[j] == 0)
                addToY += graph[i][j];
            else if(partial[j] == 1)
                addToX += graph[i][j];
        }
        lb += min(addToX, addToY);
    }
    if(current_weight + lb >= best_cut_weight)
        return;
#endif

    // Větev: přiřazení uzlu idx do X (0)
    {
        vector<int> local = partial;
        local[idx] = 0;
        int addWeight = 0;
        for (int j = 0; j < idx; j++) {
            if(local[j] == 1)
                addWeight += graph[idx][j];
        }
        dfs_sequential(idx + 1, count0 + 1, current_weight + addWeight, local);
    }

    // Větev: přiřazení uzlu idx do Y (1)
    {
        vector<int> local = partial;
        local[idx] = 1;
        int addWeight = 0;
        for (int j = 0; j < idx; j++) {
            if(local[j] == 0)
                addWeight += graph[idx][j];
        }
        dfs_sequential(idx + 1, count0, current_weight + addWeight, local);
    }
}

// Generuje subproblémy až do daného cutoff indexu
void generate_subproblems(int idx, int count0, int current_weight, vector<int>& partial, vector<Subproblem>& subproblems, int cutoff) {
    if(idx == cutoff) {
        subproblems.push_back({idx, count0, current_weight, partial});
        return;
    }
    // Větev: přiřazení uzlu idx do X (0)
    {
        vector<int> local = partial;
        local[idx] = 0;
        int addWeight = 0;
        for (int j = 0; j < idx; j++) {
            if(local[j] == 1)
                addWeight += graph[idx][j];
        }
        generate_subproblems(idx + 1, count0 + 1, current_weight + addWeight, local, subproblems, cutoff);
    }
    // Větev: přiřazení uzlu idx do Y (1)
    {
        vector<int> local = partial;
        local[idx] = 1;
        int addWeight = 0;
        for (int j = 0; j < idx; j++) {
            if(local[j] == 0)
                addWeight += graph[idx][j];
        }
        generate_subproblems(idx + 1, count0, current_weight + addWeight, local, subproblems, cutoff);
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
    
    graph.assign(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            infile >> graph[i][j];
        }
    }
    infile.close();
    
    // Pro symetrické případy (kdy je a == n/2) fixujeme uzel 0 v X
    bool symmetricCase = ((n % 2 == 0) && (a == n/2));
    vector<int> initial(n, -1);
    if(symmetricCase)
        initial[0] = 0;
    
    // Vybereme cutoff pro generování subproblémů
    const int CUTOFF = 10; 
    vector<Subproblem> subproblems;
    // Pokud fixujeme první uzel, začneme od indexu 1
    generate_subproblems(symmetricCase ? 1 : 0, symmetricCase ? 1 : 0, 0, initial, subproblems, CUTOFF);
    
    // Nastavení počtu vláken automaticky
    int nThreads = omp_get_num_procs();
    omp_set_dynamic(0); 
    omp_set_nested(1); // Vnořená paralelizace
    omp_set_num_threads(nThreads);
    
    double start = omp_get_wtime();
    // Paralelní zpracování generovaných subproblémů pomocí datového paralelismu (paralelní smyčka)
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < subproblems.size(); i++) {
        dfs_sequential(subproblems[i].idx,
                       subproblems[i].count0,
                       subproblems[i].current_weight,
                       subproblems[i].partial);
    }
    double end = omp_get_wtime();
    par_time = end - start;
    
    cout << "Paralelni cas [s]: " << par_time << endl;
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
