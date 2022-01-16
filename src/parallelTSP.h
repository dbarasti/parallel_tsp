//
// Created by dbara on 13/10/21.
//

#ifndef TSP_GA_PARALLELTSP_H
#define TSP_GA_PARALLELTSP_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include <random>
#include <algorithm>
#include "lib/Point.h"
#include "lib/utils.h"
#include <future>


class ParallelTSP{
public:
    int run(int nCities, unsigned int populationSize, int generations, double mutationProbability, double crossoverProbability, unsigned int nWorkers, int seed);
private:
    const double MIN = 0;
    const double MAX = 100;
    std::vector<Point> cities;
    std::vector<std::vector<int>> population;
    std::vector<std::future<void>> workers;
    std::vector<std::pair<int, int>> chunks;

    void setup(unsigned int nWorkers, unsigned int populationSize, unsigned int nCities, int seed);

    std::vector<Point> generateCities(int nCities, int seed);

    void generateInitialPopulation(int seed);

    double calculateLowerBound();

    std::vector<double> evaluate();

    std::vector<double> calculateFitness(std::vector<double>& evaluation, const double & lowerBound);

    void selection(const std::vector<double> & fitness, unsigned int seed);

    void crossover(double crossoverRate);

    void mutate(double probability);

    static std::vector<int> recombine(std::vector<int>& chromosomeA, std::vector<int>& chromosomeB, int nEl);

    static std::vector<int> shuffle_vector(std::vector<int> &vec, int seed);

    static int pickOne(const std::vector<double>& fitness, std::mt19937 &gen, std::uniform_real_distribution<> dis);
};

int ParallelTSP::run(int nCities, unsigned int populationSize, int generations, double mutationProbability, double crossoverProbability, unsigned int nWorkers, int seed) {
    if (nCities > populationSize){
        std::cout << "[ERROR] nCities cannot be greater than populationSize" << std::endl;
        return 1;
    }
    std::ofstream outFile;
    outFile.open ("data_par.txt", std::ofstream::trunc);

    setup(nWorkers, populationSize, nCities, seed);

    //generate the cities in the array cities
    // ONE TIME ONLY. DONE SEQUENTIALLY
    cities = generateCities(nCities, seed);

    // I now need a population
    // ONE TIME ONLY. DONE SEQUENTIALLY
    generateInitialPopulation(seed);

#if MEASURE == true
    auto start = std::chrono::system_clock::now();
#endif
    // Now we have an initial population ready
    // we can start with the iterations
    for (int iter = 0; iter < generations; ++iter) {
        double lb = calculateLowerBound();

        // Let's calculate the evaluation score
        auto evaluation = evaluate();

        // Now calculate fitness (a percentage) based on the evaluation
        auto fitness = calculateFitness(evaluation, lb);

#if TEST == true
        utils::computeAvgEval(outFile, iter, evaluation);
#endif
        // It's time for reproduction!
        // Let's find the intermediate population, aka chromosomes that will be recombined (and then mutated) to be the next generation
        // Select populationSize elements so that the higher the fitness the higher the probability to be selected
        selection(fitness, seed);

        crossover(crossoverProbability);

        mutate(mutationProbability);

        fitness.clear();
        evaluation.clear();

        // if (iter % 100 == 0) std::cout << "iter " << iter + 1 << std::endl;
    }
#if MEASURE == true
    auto end = std::chrono::system_clock::now();
    std::cout << nWorkers << " " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
#endif
    population.clear();
    outFile.close();
    return 0;
}

inline void ParallelTSP::setup(unsigned int nWorkers, unsigned int populationSize, unsigned int nCities, int seed) {
    srand(seed);

    // setting the right amount of workers
    auto const hardware_threads =
            std::thread::hardware_concurrency();

    auto properWorkers = std::min(hardware_threads-1!=0?hardware_threads-1:2, nWorkers);

    if (properWorkers > populationSize) {
        properWorkers = populationSize;
    }

    this->workers.resize(properWorkers);
    this->chunks.resize(properWorkers);

    int workerCellsToCompute = std::floor(populationSize / properWorkers);
    auto remainedElements = populationSize % properWorkers;
    int index = 0;

    // determine chunks to be assigned
    std::for_each(this->chunks.begin(), this->chunks.end(), [&](std::pair<int, int> &chunk) {
        chunk.first = index;
        if (remainedElements) {
            chunk.second = chunk.first + workerCellsToCompute;
            remainedElements--;
        } else {
            chunk.second = chunk.first + workerCellsToCompute - 1;
        }
        index = chunk.second + 1;
    });

    // resizing population
    population.resize(populationSize);

    for (auto & chromosome : population)
        chromosome.resize(nCities);
}

inline std::vector<Point> ParallelTSP::generateCities(int nCities, int seed) {
    cities.reserve(nCities);

    std::mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(MIN, MAX);

    for (int i = 0; i < nCities; ++i) {
        cities.emplace_back(round(dis(gen)), round(dis(gen)));
    }
    return cities;
}

inline std::vector<int> ParallelTSP::shuffle_vector(std::vector<int> &vec, int seed) {
    std::vector<int> copy(vec);
    auto rng = std::default_random_engine(seed);
    std::shuffle(std::begin(copy), std::end(copy), rng);
    return copy;
}

inline std::vector<double> ParallelTSP::evaluate() {
#if TEST
    auto start = std::chrono::system_clock::now();
#endif
    std::vector<double> evaluation(population.size());
#if DETAILED_MEASURE == true
    auto start = std::chrono::system_clock::now();
#endif

    auto chunksIt = chunks.begin();
    std::for_each(workers.begin(), workers.end(), [&](std::future<void> &worker) {
        worker = std::async(std::launch::async, [&, start = chunksIt->first, end = chunksIt->second] {
            double totalDistanceOfChromosome;
            for (unsigned int i = start; i <= end; ++i) {
                totalDistanceOfChromosome = 0;
                for (unsigned int j = 0; j < cities.size() - 1; ++j) {
                    totalDistanceOfChromosome += cities[population[i][j]].dist(cities[population[i][j + 1]]);
                }
                // Evaluation score of i-th chromosome
                evaluation[i] = totalDistanceOfChromosome;
            }
        });
        // iterate over chunks data structure to assign indexes to the workers
        chunksIt++;
    });

    std::for_each(workers.begin(), workers.end(), [&](std::future<void> &worker) { worker.get(); });
#if DETAILED_MEASURE == true
    auto end = std::chrono::system_clock::now();
    std::cout << "parallel evaluate computation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
#endif

#if TEST
    auto end = std::chrono::system_clock::now();
    std::cout << "Evaluation - parallel time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return evaluation;
}

inline std::vector<double> ParallelTSP::calculateFitness(std::vector<double> &evaluation, const double &lowerBound) {
#if TEST
    auto start = std::chrono::system_clock::now();
#endif
    std::vector<double> fitness(evaluation.size());
    for (unsigned int i = 0; i < evaluation.size(); ++i) {
        fitness[i] = lowerBound / evaluation[i];
    }

#if TEST
    auto end = std::chrono::system_clock::now();
    std::cout << "Fitness - parallel time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return fitness;
}

inline int ParallelTSP::pickOne(const std::vector<double> &fitness, std::mt19937 &gen, std::uniform_real_distribution<> dis) {
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

inline void ParallelTSP::selection(const std::vector<double> &fitness, unsigned int seed) {
#if TEST
    auto start = std::chrono::system_clock::now();
#endif

    std::vector<std::vector<int>> selection(population.size());
    double fitnessSum = 0;
    for (unsigned long i = 0; i < population.size(); ++i) {
        fitnessSum += fitness[i];
    }
    std::mt19937 gen(seed); //Standard mersenne_twister_engine
    std::uniform_real_distribution<> dis(0, fitnessSum);

#if DETAILED_MEASURE == true
    auto start = std::chrono::system_clock::now();
#endif
    auto chunksIt = chunks.begin();
    std::for_each(workers.begin(), workers.end(), [&](std::future<void> &worker) {
        worker = std::async(std::launch::async, [&, start = chunksIt->first, end = chunksIt->second] {
            int pickedIndex;
            for (unsigned long i = start; i <= end; ++i) {
                pickedIndex = pickOne(fitness, gen, dis);
                // timesPicked[pickedIndex]++;
                selection[i] = population[pickedIndex];
            }
        });
        // iterate over chunks data structure to assign indexes to the workers
        chunksIt++;
    });

    std::for_each(workers.begin(), workers.end(), [&](std::future<void> &worker) { worker.get(); });
#if DETAILED_MEASURE == true
    auto end = std::chrono::system_clock::now();
    std::cout << "parallel selection computation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
#endif
    population = std::move(selection);

#if TEST
    auto end = std::chrono::system_clock::now();
    std::cout << "Selection - parallel time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
}

inline std::vector<int> ParallelTSP::recombine(std::vector<int> &chromosomeA, std::vector<int> &chromosomeB, int nEl) {
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

inline void ParallelTSP::crossover(const double crossoverRate) {
#if TEST
    auto start = std::chrono::system_clock::now();
#endif
    std::vector<std::vector<int>> newPopulation(population.size());
#if DETAILED_MEASURE == true
    auto start = std::chrono::system_clock::now();
#endif
    auto chunksIt = chunks.begin();
    std::for_each(workers.begin(), workers.end(), [&](std::future<void> &worker) {
        worker = std::async(std::launch::async, [&, start = chunksIt->first, end = chunksIt->second] {
            double r;
            for (unsigned long i = start; i <= end; ++i) {
                r = (double) rand() / RAND_MAX;
                if (r >= crossoverRate) {
                    newPopulation[i] = population[i];
                    continue;
                }
                int indexA = rand() % population.size();
                int indexB = rand() % population.size();
                std::vector<int> mateA = population[indexA];
                std::vector<int> mateB = population[indexB];

                newPopulation[i] = recombine(mateA, mateB, cities.size());
            }
        });
        // iterate over chunks data structure to assign indexes to the workers
        chunksIt++;
    });

    std::for_each(workers.begin(), workers.end(), [&](std::future<void> &worker) { worker.get(); });
#if DETAILED_MEASURE == true
    auto end = std::chrono::system_clock::now();
    std::cout << "parallel crossover computation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
#endif
    population = std::move(newPopulation);

#if TEST
    auto end = std::chrono::system_clock::now();
    std::cout << "Crossover - parallel time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
}


inline void ParallelTSP::mutate(const double probability) {
#if TEST
    auto start = std::chrono::system_clock::now();
#endif
    for (auto & toMutate : population) {
        double r = (float) rand() / RAND_MAX;
        if (r < probability) {
            utils::swapTwo(toMutate);
        } // else, do nothing, aka do not mutate
    }

#if TEST
    auto end = std::chrono::system_clock::now();
    std::cout << "Mutation - parallel time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
}

inline double ParallelTSP::calculateLowerBound() {
#if TEST
    auto start = std::chrono::system_clock::now();
#endif
    std::vector<utils::Eval> evaluation(population.size());

    // distance matrix initialization
    std::vector<std::vector<double>> dstMatrix(population.size());
    for (unsigned int i = 0; i < population.size(); ++i) {
        dstMatrix[i] = std::vector<double>(cities.size() - 1);
    }
#if DETAILED_MEASURE == true
    auto start = std::chrono::system_clock::now();
#endif
    auto chunksIt = chunks.begin();
    std::for_each(workers.begin(), workers.end(), [&](std::future<void> &worker) {
        worker = std::async(std::launch::async, [&, start = chunksIt->first, end = chunksIt->second] {
            double dst;
            double totalDistanceOfChromosome;
            for (unsigned int i = start; i <= end; ++i) {
                totalDistanceOfChromosome = 0;
                for (unsigned int j = 0; j < cities.size() - 1; ++j) {
                    dst = cities[population[i][j]].dist(cities[population[i][j + 1]]);
                    dstMatrix[i][j] = dst;
                    totalDistanceOfChromosome += dst;
                }
                // Evaluation score of i-th chromosome
                evaluation[i].score = totalDistanceOfChromosome;
                evaluation[i].dstMatrixIndex = i;
            }
        });
        // iterate over chunks data structure to assign indexes to the workers
        chunksIt++;
    });

    std::for_each(workers.begin(), workers.end(), [&](std::future<void> &worker) { worker.get(); });
#if DETAILED_MEASURE == true
    auto end = std::chrono::system_clock::now();
    std::cout << "parallel lb computation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
#endif
    // Sort evaluation by score
    std::sort(evaluation.begin(), evaluation.end(), utils::compareEvaluations);

    double lowerBound = 0;

    // pick nCities-1 distances, one for each of the first ncities-1 dstMatrix rows. Pick every time the smallest of the row
    for (unsigned int i = 0; i < cities.size() - 1; ++i) {
        double bestDistance = utils::findBestDistance(dstMatrix[evaluation[i].dstMatrixIndex]);
        lowerBound += bestDistance;
    }
#if TEST == true
    auto end = std::chrono::system_clock::now();
    std::cout << "Lower bound calculation - parallel time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return lowerBound;
}

inline void ParallelTSP::generateInitialPopulation(int seed) {
#if TEST
    // start timer to record the initial population generation
    auto start = std::chrono::system_clock::now();
#endif
    // The population is made of arrays of indexes, aka orders with which I visit the cities
    // The order changes, the cities array remains the same
    // the initial ordering can be generated with
    for (unsigned long i = 0; i < cities.size(); ++i) {
        population[0][i] = i;
    }

    // ...

    // First chromosome has been populated. Now let's create the rest of the initial population
    for (unsigned long i = 1; i < population.size(); ++i) {
        // Every time I shuffle the previous chromosome
        population[i].reserve(cities.size());
        population[i] = shuffle_vector(population[i - 1], seed);
    }

#if TEST
    auto end = std::chrono::system_clock::now();
    std::cout << "Initial population generation - parallel time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
}


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"



#endif //TSP_GA_PARALLELTSP_H
