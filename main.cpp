#define TEST true

#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include <random>
#include <algorithm>
#include "Point.h"
#include <future>

struct Eval {
    int dstMatrixIndex;
    double score;
};

std::vector<Point> generateCities(const int nCities, double min, double max, unsigned int seed) {
    std::vector<Point> cities;
    cities.reserve(nCities);

    std::mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(min, max);

    for (int i = 0; i < nCities; ++i) {
        cities.emplace_back(round(dis(gen)), round(dis(gen)));
    }
    return cities;
}

/**
 * Returns a shuffled vector starting from vec
 * */
std::vector<int> shuffle_vector(std::vector<int>& vec, const unsigned int seed) {
    std::vector<int> copy(vec);
    auto rng = std::default_random_engine(seed);
    std::shuffle(std::begin(copy), std::end(copy), rng);
    return copy;
}

/*
 * Computes the evaluation (total trip distance) for each chromosome
 * */
std::vector<double> evaluate(std::vector<std::vector<int>>& population, std::vector<Point>& cities, const int & populationSize, const int & nCities) {
#if TEST == true
    auto start = std::chrono::system_clock::now();
#endif
    std::vector<double> evaluation(populationSize);
    double totalDistanceOfChromosome;
    for (int i = 0; i < populationSize; ++i) {
        totalDistanceOfChromosome = 0;
        for (int j = 0; j < nCities - 1; ++j) {
            totalDistanceOfChromosome += cities[population[i][j]].dist(cities[population[i][j + 1]]);
        }
        // Evaluation score of i-th chromosome
        evaluation[i] = totalDistanceOfChromosome;
    }
#if TEST == true
    auto end = std::chrono::system_clock::now();
    std::cout << "Evaluation - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return evaluation;
}

/*
 * Computes fitness value for each chromosome, using the lowerBound estimation
 * */
std::vector<double> calculateFitness(std::vector<double>& evaluation, const int & populationSize, const double & lowerBound) {
#if TEST == true
    auto start = std::chrono::system_clock::now();
#endif
    std::vector<double> fitness(populationSize);
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

int pickOne(const std::vector<double>& fitness, std::mt19937 &gen, std::uniform_real_distribution<> dis) {
    // Get distribution in 0...fitnessSum
    double r = dis(gen);
    int i = 0;
    // sum(fitness) >= r so I don't need to check if I go out of bound
    while (r > 0) {
        r -= fitness[i];
        i++;
    }
    return --i;
}

std::vector<std::vector<int>> selection(const std::vector<double> & fitness, const std::vector<std::vector<int>> & population, const int populationSize, unsigned int seed) {
#if TEST == true
    auto start = std::chrono::system_clock::now();
#endif

    std::vector<std::vector<int>> selection(populationSize);
    double fitnessSum = 0;
    for (int i = 0; i < populationSize; ++i) {
        fitnessSum += fitness[i];
    }
    std::mt19937 gen(seed); //Standard mersenne_twister_engine
    std::uniform_real_distribution<> dis(0, fitnessSum);

    int pickedIndex;
    for (int i = 0; i < populationSize; ++i) {
        pickedIndex = pickOne(fitness, gen, dis);
        // timesPicked[pickedIndex]++;
        selection[i] = population[pickedIndex];
    }

#if TEST == true
    auto end = std::chrono::system_clock::now();
    std::cout << "Selection - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return selection;
}

std::vector<int> recombine(std::vector<int>& chromosomeA, std::vector<int>& chromosomeB, int nEl) {
    std::vector<int> combination(nEl);
    // Pick two rand indexes
    int indexA = rand() % nEl;
    int indexB = rand() % nEl;

    if (indexB < indexA) {
        std::swap(indexA, indexB);
    }

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

    while (combinationIndex < nEl && chromosomeIndex < nEl) {
        if(std::find(combination.begin(), combination.end(), chromosomeB[chromosomeIndex]) == combination.end()){
            combination[combinationIndex] = chromosomeB[chromosomeIndex];
            combinationIndex++;
        }
        chromosomeIndex++;
    }
    return combination;
}

std::vector<std::vector<int>> crossover(std::vector<std::vector<int>>& population, const int populationSize, const int nCities, const double crossoverRate) {
#if TEST == true
    auto start = std::chrono::system_clock::now();
#endif
    std::vector<std::vector<int>> selection(populationSize);
    double r;
    for (int i = 0; i < populationSize; ++i) {
        r = (double) rand() / RAND_MAX;
        if (r >= crossoverRate) {
            selection[i] = population[i];
            continue;
        }
        int indexA = rand() % populationSize;
        int indexB = rand() % populationSize;
        std::vector<int> mateA = population[indexA];
        std::vector<int> mateB = population[indexB];

        selection[i] = recombine(mateA, mateB, nCities);
    }
#if TEST == true
    auto end = std::chrono::system_clock::now();
    std::cout << "Crossover - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return selection;
}

/*
 * Swaps two random elements of array vec
 * */
void swapTwo(std::vector<int>& vec, const int nEl) {
    int indexA = rand() % nEl;
    int indexB = rand() % nEl;
    while (indexA == indexB) {
        indexB = rand() % nEl;
    }
    int aux = vec[indexA];
    vec[indexA] = vec[indexB];
    vec[indexB] = aux;
}

/*
 * Swaps two elements inside each chromosome with certain probability
 * */
void mutate(std::vector<std::vector<int>>& population, const int populationSize, const int nCities, const double probability) {
#if TEST == true
    auto start = std::chrono::system_clock::now();
#endif
    std::vector<int> populationToMutate;
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

double calculateLowerBound(std::vector<std::vector<int>>& population, std::vector<Point>&cities, const int & populationSize, const int & nCities) {
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
    }

    // Sort evaluation by score
    std::sort(evaluation, evaluation + populationSize, compareEvaluations);

    // pick ncities-1 distances, one for each of the first ncities-1 dstMatrix rows. Pick every time the smallest of the row
    for (int i = 0; i < nCities - 1; ++i) {
        double bestDistance = findBestDistance(dstMatrix[evaluation[i].dstMatrixIndex], nCities - 1);
        lowerBound += bestDistance;
    }
    delete[] evaluation;
    for (int k = 0; k < populationSize; ++k) {
        delete[] dstMatrix[k];
    }
    delete[] dstMatrix;
#if TEST == true
    auto end = std::chrono::system_clock::now();
    std::cout << "Lower bound calculation - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return lowerBound;
}


void run(const int nCities, const int populationSize, const int generations, const double min, const double max,
         const unsigned int seed, const double mutationProbability, const double crossoverProbability, unsigned int nWorkers) {
    srand(seed);
    std::ofstream outFile;
    outFile.open ("data.txt");

#if TEST == true
    // start timer to record the initial population generation
    auto start = std::chrono::system_clock::now();
#endif
    //generate the cities in the array cities
    std::vector<Point> cities = generateCities(nCities, min, max, seed);

    // Defining the matrix where I store population of orders:
    std::vector<std::vector<int>> population(populationSize);

    for (int i = 0; i < populationSize; i++)
        population[i].resize(nCities);

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
        population[i].reserve(nCities);
        population[i] = shuffle_vector(population[i - 1], seed);
    }

#if TEST == true
    auto end = std::chrono::system_clock::now();
    std::cout << "Initial population generation - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    // Now we have an initial population ready

    double bestLocalEval;
    double avgEval;

    for (int iter = 0; iter < generations; ++iter) {

        double lb = calculateLowerBound(population, cities, populationSize, nCities);

        // Let's calculate the evaluation score
        std::vector<double> evaluation = evaluate(population, cities, populationSize, nCities);

        // Now calculate fitness (a percentage) based on the evaluation

        auto fitness = calculateFitness(evaluation, populationSize, lb);

        avgEval = 0;
        bestLocalEval = 0;
        for (int i = 0; i < populationSize; ++i) {
            avgEval += evaluation[i];
            if (evaluation[i] > bestLocalEval) {
                bestLocalEval = evaluation[i];
            }
        }
        avgEval = avgEval / populationSize;

        outFile << iter << "\t" << avgEval << std::endl;

        // It's time for reproduction!

        // Let's find the intermediate population, aka chromosomes that will be recombined (and then mutated) to be the next generation

        // Select populationSize elements so that the higher the fitness the higher the probability to be selected


        std::vector<std::vector<int>> intermediatePopulation = selection(fitness, population, populationSize, seed);

        // printPopulation(intermediatePopulation, populationSize, nCities);

        // With the intermediate population I can now crossover the elements


        // printPopulation(intermediatePopulation, populationSize, nCities);
        std::vector<std::vector<int>> nextGen = crossover(intermediatePopulation, populationSize, nCities, crossoverProbability);

        mutate(nextGen, populationSize, nCities, mutationProbability);

        intermediatePopulation.clear();
        fitness.clear();
        evaluation.clear();
        population.clear();
        population = std::move(nextGen);

    }
    population.clear();

    // delete[] cities;
    outFile.close();
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"

int main(int argc, char *argv[]) {

    // Reading arguments from parameters
    if (argc < 7) {
        std::cout << "Usage is " << argv[0]
                  << " nCities populationSize generations mutationProbability crossoverProbability nWorkers[seed]"
                  << std::endl;
        return (-1);
    }

    int nCities = std::atoi(argv[1]);
    int populationSize = std::atoi(argv[2]);
    int generations = std::atoi(argv[3]);
    double mutationProbability = std::atof(argv[4]);
    double crossoverProbability = std::atof(argv[5]);
    unsigned int nWorkers = std::atoi(argv[6]);
    const double min = 0;
    const double max = 100;
    int seed = argv[7] ? std::atoi(argv[7]) : 35412;

#if TEST == true
    nCities = 1000;
    populationSize = 2000;
    generations = 1;
    mutationProbability = 0.01;
    crossoverProbability = 0.1;
#endif

    std::vector<std::future<void>> workers;
    std::vector<std::pair<int, int>> chunks;

    // setting the right amount of workers
    auto const hardware_threads=
            std::thread::hardware_concurrency();

    auto properWorkers = std::min(hardware_threads-1!=0?hardware_threads-1:2,nWorkers);

    if (properWorkers > populationSize) {
        properWorkers = populationSize;
    }

    workers.resize(properWorkers);
    chunks.resize(properWorkers);

    int workerCellsToCompute = std::floor(populationSize / properWorkers);
    auto remainedElements = populationSize % properWorkers;
    int index = 0;

    // determine chunks to be assigned
    std::for_each(chunks.begin(), chunks.end(), [&](std::pair<int, int> &chunk) {
        chunk.first = index;
        if (remainedElements) {
            chunk.second = chunk.first + workerCellsToCompute;
            remainedElements--;
        } else {
            chunk.second = chunk.first + workerCellsToCompute - 1;
        }
        index = chunk.second + 1;
    });



    run(nCities, populationSize, generations, min, max, seed, mutationProbability, crossoverProbability, nWorkers);

}


#pragma clang diagnostic pop