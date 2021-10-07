#define TEST false

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <ctime>
#include "Point.h"
#include <chrono>

using namespace std;

struct Eval {
    int dstMatrixIndex;
    double score;
};

Point *generateCities(const int nCities, double min, double max, unsigned int seed) {
    auto cities = new Point[nCities];
    std::mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(min, max);

    for (int i = 0; i < nCities; ++i) {
        Point p(round(dis(gen)), round(dis(gen)));
        cities[i] = p;
    }
    return cities;
}

/**
 * returns a shuffled array starting from arr
 * */
int *shuffle_array(const int arr[], int n, const unsigned int seed) {
    int *copy = new int[n];
    for (int i = 0; i < n; ++i) {
        copy[i] = arr[i];
    }
    // Shuffling our array
    shuffle(copy, copy + n,
            default_random_engine(seed));
    return copy;
}

double *evaluate(int **population, Point *cities, const int populationSize, const int nCities) {
#if TEST == true
    auto start = std::chrono::system_clock::now();
#endif
    auto evaluation = new double[populationSize];
    double totalDistanceOfChromosome;
    for (int i = 0; i < populationSize; ++i) {
        totalDistanceOfChromosome = 0;
        for (int j = 0; j < nCities - 1; ++j) {
            totalDistanceOfChromosome += cities[population[i][j]].dist(cities[population[i][j + 1]]);
        }
        // Evaluation score of i-th chromosome
        evaluation[i] = totalDistanceOfChromosome;
        // cout << "Total distance for chromosome " << i << ": " << totalDistanceOfChromosome << endl;
    }
#if TEST == true
    auto end = std::chrono::system_clock::now();
    std::cout << "Evaluation - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return evaluation;
}

double *calculateFitness(const double *evaluation, const int populationSize, const double lowerBound) {
#if TEST == true
    auto start = std::chrono::system_clock::now();
#endif
    auto fitness = new double[populationSize];
    for (int i = 0; i < populationSize; ++i) {
        fitness[i] = lowerBound / evaluation[i];
    }
#if TEST == true
    auto end = std::chrono::system_clock::now();
    std::cout << "Fitness - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return fitness;
}

int pickOne(const double *fitness, std::mt19937 &gen, std::uniform_real_distribution<> dis) {
    // Get distribution in 0...fitnessSum
    double r = dis(gen);
    // cout << "Random value: " << r << endl;
    int i = 0;
    // sum(fitness) is >= r so I don't need to check if I go out of bound
    while (r > 0) {
        r -= fitness[i];
        //    cout << "Fitness: " << fitness[i] << endl;
        i++;
    }
    return --i;
}

int **selection(double *fitness, int **population, const int populationSize, unsigned int seed) {
#if TEST == true
    auto start = std::chrono::system_clock::now();
#endif

    int **selection = new int *[populationSize];
    double fitnessSum = 0;
    for (int i = 0; i < populationSize; ++i) {
        fitnessSum += fitness[i];
    }
    std::mt19937 gen(seed); //Standard mersenne_twister_engine
    std::uniform_real_distribution<> dis(0, fitnessSum);

    // int timesPicked[populationSize];
    /*
    for (int i = 0; i < populationSize; ++i) {
        timesPicked[i] = 0;
    }
     */
    int pickedIndex;
    for (int i = 0; i < populationSize; ++i) {
        pickedIndex = pickOne(fitness, gen, dis);
        // timesPicked[pickedIndex]++;
        selection[i] = population[pickedIndex];
    }
    /*
    for (int j = 0; j < populationSize; ++j) {
        cout << "Population " << j << "-th picked " << timesPicked[j] << " times" << endl;
    }
     */
#if TEST == true
    auto end = std::chrono::system_clock::now();
    std::cout << "Selection - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return selection;
}

bool isContained(int el, const int *array, int nEl) {
    for (int i = 0; i < nEl; ++i) {
        if (array[i] == el) {
            return true;
        }
    }
    return false;
}

int *recombine(const int *chromosomeA, int *chromosomeB, int nEl) {
    int *combination = new int[nEl];
    // Pick two rand indexes
    int indexA = rand() % nEl;
    int indexB = rand() % nEl;

    if (indexB < indexA) {
        int aux = indexA;
        indexA = indexB;
        indexB = aux;
    }

    // cout << "IndexA: " << indexA << " ";
    // cout << "IndexB: " << indexB << " " << endl;

    // First, put chromosomeA[indexA..indexB (included)] into combination[0..indexB-indexA]
    int i;
    int chromosomeIndex = indexA;
    for (i = 0; i < indexB - indexA + 1; ++i) {
        combination[i] = chromosomeA[chromosomeIndex];
        chromosomeIndex++;
    }

    int combinationIndex = i;
    chromosomeIndex = 0;

    // Let's combine the rest of the elements

    while (combinationIndex < nEl) {
        if (!isContained(chromosomeB[chromosomeIndex], combination, combinationIndex)) {
            combination[combinationIndex] = chromosomeB[chromosomeIndex];
            combinationIndex++;
        }
        chromosomeIndex++;
    }

    /*cout << "Combination: ";
    for (int j = 0; j < nEl; ++j) {
        cout << combination[j] << " ";
    }
    cout << endl;*/

    return combination;
}

int **crossover(int **population, const int populationSize, const int nCities, const double crossoverRate) {
#if TEST == true
    auto start = std::chrono::system_clock::now();
#endif
    int **selection = new int *[populationSize];
    double r;
    for (int i = 0; i < populationSize; ++i) {
        r = (double) rand() / RAND_MAX;
        if (r >= crossoverRate) {
            selection[i] = population[i];
            continue;
        }
        int indexA = rand() % populationSize;
        int indexB = rand() % populationSize;
        int *mateA = population[indexA];
        int *mateB = population[indexB];
        /*
        cout << "Mate A: ";
        for (int j = 0; j < nCities; ++j) {
            cout << mateA[j] << " ";
        }
        cout << endl;

        cout << "Mate B: ";
        for (int j = 0; j < nCities; ++j) {
            cout << mateB[j] << " ";
        }
        cout << endl;
        */
        selection[i] = recombine(mateA, mateB, nCities);
    }
#if TEST == true
    auto end = std::chrono::system_clock::now();
    std::cout << "Crossover - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return selection;
}

void printPopulation(int *const *intermediatePopulation, const int populationSize, const int nCities) {
    for (int i = 0; i < populationSize; ++i) {
        cout << "Chromosome " << i << ":";
        for (int j = 0; j < nCities; ++j) {
            cout << " " << intermediatePopulation[i][j];
        }
        cout << endl;
    }
}

void swapTwo(int *arr, const int nEl) {
    int indexA = rand() % nEl;
    int indexB = rand() % nEl;
    while (indexA == indexB) {
        indexB = rand() % nEl;
    }
    int aux = arr[indexA];
    arr[indexA] = arr[indexB];
    arr[indexB] = aux;
}

// swaps two elements inside each chromosome with certain probability
void mutate(int **population, const int populationSize, const int nCities, const double probability) {
#if TEST == true
    auto start = std::chrono::system_clock::now();
#endif
    int *populationToMutate;
    for (int i = 0; i < populationSize; ++i) {
        populationToMutate = population[i];
        double r = (float) rand() / RAND_MAX;
        if (r < probability) {
            swapTwo(populationToMutate, nCities);
        }
    }
#if TEST == true
    auto end = std::chrono::system_clock::now();
    std::cout << "Mutation - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
}

bool compareEvaluations(Eval a, Eval b) {
    return a.score < b.score;
}

double findBestDistance(const double *distances, int n) {
    double bestSoFar = MAXFLOAT;
    for (int i = 0; i < n; ++i) {
        if (distances[i] < bestSoFar) {
            bestSoFar = distances[i];
        }
    }
    return bestSoFar;
}

double calculateLowerBound(int **population, Point *cities, const int populationSize, const int nCities) {
#if TEST == true
    auto start = std::chrono::system_clock::now();
#endif
    auto evaluation = new Eval[populationSize];
    // Matrix initialization
    auto dstMatrix = new double *[populationSize];
    for (int i = 0; i < populationSize; ++i) {
        dstMatrix[i] = new double[nCities - 1];
    }
    double totalDistanceOfChromosome;
    double dst;
    double lowerBound = 0;
    for (int i = 0; i < populationSize; ++i) {
        totalDistanceOfChromosome = 0;
        for (int j = 0; j < nCities - 1; ++j) {
            dst = cities[population[i][j]].dist(cities[population[i][j + 1]]);
            dstMatrix[i][j] = dst;
            totalDistanceOfChromosome += dst;
        }
        // Evaluation score of i-th chromosome
        evaluation[i].score = totalDistanceOfChromosome;
        evaluation[i].dstMatrixIndex = i;

        // cout << "Total distance for chromosome " << i << ": " << totalDistanceOfChromosome << endl;
    }

    // Sort evaluation by score
    std::sort(evaluation, evaluation + populationSize, compareEvaluations);

    // pick ncities-1 distances, one for each of the first ncities-1 dstMatrix rows. Pick every time the smallest of the row
    for (int i = 0; i < nCities - 1; ++i) {
        double bestDistance = findBestDistance(dstMatrix[evaluation[i].dstMatrixIndex], nCities - 1);
        lowerBound += bestDistance;
    }
    delete[] evaluation;
    delete[] dstMatrix;
#if TEST == true
    auto end = std::chrono::system_clock::now();
    std::cout << "Lower bound calculation - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return lowerBound;
}


void run(const int nCities, const int populationSize, const int generations, const double min, const double max, const unsigned int seed, const double mutationProbability){
    srand(seed);
    ofstream outFile;
    outFile.open ("data.txt");

#if TEST == true
    // start timer to record the initial population generation
    auto start = std::chrono::system_clock::now();
#endif
    //generate the cities in the array cities
    Point *cities = generateCities(nCities, min, max, seed);
    /* for (int i = 0; i < nCities; ++i) {
        cities[i].print(cout);
    }
    cout << endl; */

    // Defining the matrix where I store population of orders:
    int **population;
    population = new int *[populationSize];
    for (int i = 0; i < populationSize; i++)
        population[i] = new int[nCities];

    // I now need a population
    // The population is made of arrays of indexes, aka orders with which I visit the cities
    // The order changes, the cities array remain the same

    // I create an initial order which is 0,1,...,nCities-1, then I shuffle this initial order to obtain other populationSize-1 orders, forming the initial population

    for (int i = 0; i < nCities; ++i) {
        population[0][i] = i;
    }

    // First chromosome has been populated. Now let's create the rest of the initial population
    for (int i = 1; i < populationSize; ++i) {
        // Every time I shuffle the previous chromosome
        int *a = shuffle_array(population[i - 1], nCities, 0);
        for (int j = 0; j < nCities; ++j) {
            population[i][j] = a[j];
        }
        // delete[] a;
    }

    // printPopulation(population, populationSize, nCities);

#if TEST == true
    auto end = std::chrono::system_clock::now();
    std::cout << "Initial population generation - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    // Now we have an initial population ready

    double bestGlobalFitness = 0;
    double bestLocalFitness;
    double bestLocalEval;
    double avgEval;
    int globalBestIndex;
    int localBestIndex;




    for (int iter = 0; iter < generations; ++iter) {

        double lb = calculateLowerBound(population, cities, populationSize, nCities);
        // cout << "Lower bound: " << lb << endl;

        // Let's calculate the evaluation score
        double *evaluation = evaluate(population, cities, populationSize, nCities);

        // Now calculate fitness (a percentage) based on the evaluation

        double *fitness = calculateFitness(evaluation, populationSize, lb);

        avgEval = 0;
        bestLocalEval = 0;
        for (int i = 0; i < populationSize; ++i) {
            avgEval += evaluation[i];
            if (evaluation[i] > bestLocalEval) {
                bestLocalEval = evaluation[i];
                //localBestIndex = i;
            }
        }
        avgEval = avgEval / populationSize;
        /*
        if (bestLocalFitness > bestGlobalFitness) {
            bestGlobalFitness = bestLocalFitness;
            globalBestIndex = localBestIndex;
        }
         */
        outFile << iter << "\t" << avgEval << endl;

        // It's time for reproduction!

        // Let's find the intermediate population, aka chromosomes that will be recombined (and then mutated) to be the next generation

        // Select populationSize elements so that the higher the fitness the higher the probability to be selected


        int **intermediatePopulation = selection(fitness, population, populationSize, seed);

        // printPopulation(intermediatePopulation, populationSize, nCities);

        // With the intermediate population I can now crossover the elements


        // printPopulation(intermediatePopulation, populationSize, nCities);
        double crossoverRate = 0.1;
        int **nextGen = crossover(intermediatePopulation, populationSize, nCities, crossoverRate);
        // printPopulation(nextGen, populationSize, nCities);

        mutate(nextGen, populationSize, nCities, mutationProbability);

        // printPopulation(nextGen, populationSize, nCities);

        delete[] intermediatePopulation;
        delete[] fitness;
        delete[] evaluation;
        for (int k = 0; k < populationSize; ++k) {
            // delete[] population[k];
        }
        delete[] population;
        population = nextGen;
    }

    delete[] cities;
    outFile.close();
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cout << "Usage is " << argv[0]
                  << " nCities populationSize generations mutationProbability [seed]"
                  << std::endl;
        return (-1);
    }

    const int nCities = std::atoi(argv[1]);
    int const populationSize = std::atoi(argv[2]);
    int const generations = std::atoi(argv[3]);
    const double mutationProbability = std::atof(argv[4]);
    int seed = 35412;
    const double min = 0;
    const double max = 100;

    if (argv[5]) {
        seed = std::atoi(argv[5]);
    }

    run(nCities, populationSize, generations, min, max, seed, mutationProbability);
}


#pragma clang diagnostic pop