#define TEST 0
#define MEASURE 1
#define DETAILED_MEASURE 0
#define CONVERGENCE 0


#include <iostream>
#include "src/sequentialTSP.h"
#include "src/parallelTSP.h"
#include "src/fastflowTSP.h"


int main(int argc, char *argv[]) {
    // Reading arguments from parameters
    if (argc < 7) {
        std::cout << "Usage is " << argv[0]
                  << " nCities populationSize generations mutationProbability crossoverProbability nWorkers [seed]"
                  << std::endl;
        return (-1);
    }

    int nCities = std::atoi(argv[1]);
    int populationSize = std::atoi(argv[2]);
    int generations = std::atoi(argv[3]);
    double mutationProbability = std::atof(argv[4]);
    double crossoverProbability = std::atof(argv[5]);
    unsigned int nWorkers = std::atoi(argv[6]);
    int seed = argv[7] ? std::atoi(argv[7]) : 35412;

#if TEST == true
    nCities = 5000;
    populationSize = 20000;
    generations = 1;
    mutationProbability = 0.01;
    crossoverProbability = 0.1;
    nWorkers = 4;
#endif

    auto seqTSP = SequentialTSP();
    auto parTSP = ParallelTSP();
    auto ffTSP = FastflowTSP();

    // Sequential execution
    auto statusSeq = seqTSP.run(nCities, populationSize, generations, mutationProbability, crossoverProbability, seed);
    if (statusSeq) return -1;

    // STD threads execution
    auto statusPar = parTSP.run(nCities, populationSize, generations, mutationProbability, crossoverProbability, nWorkers, seed);
    if (statusPar) return -1;

    // FastFlow execution
    auto statusFF = ffTSP.run(nCities, populationSize, generations, mutationProbability, crossoverProbability,
                              nWorkers, seed);
    if (statusFF) return -1;
}

