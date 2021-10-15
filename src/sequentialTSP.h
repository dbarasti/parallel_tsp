//
// Created by dbara on 14/10/21.
//

#ifndef TSP_GA_SEQUENTIALTSP_H
#define TSP_GA_SEQUENTIALTSP_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include <random>
#include <algorithm>
#include "lib/Point.h"
#include "lib/utils.h"
#include <future>


class SequentialTSP{
public:
    void run(int nCities, unsigned int populationSize, int generations, double mutationProbability, double crossoverProbability, int seed);
private:
    const double MIN = 0;
    const double MAX = 100;
    std::vector<Point> cities;
    std::vector<std::vector<int>> population;

    std::vector<Point> generateCities(int nCities, int seed);

    double calculateLowerBound();

    std::vector<double> evaluate();

    std::vector<double> calculateFitness(std::vector<double>& evaluation, const double & lowerBound);

    std::vector<std::vector<int>> selection(const std::vector<double> & fitness, const std::vector<std::vector<int>> & population, unsigned int seed);

    std::vector<std::vector<int>> crossover(std::vector<std::vector<int>>& population, int nCities, double crossoverRate);

    std::vector<int> recombine(std::vector<int>& chromosomeA, std::vector<int>& chromosomeB, int nEl);

    void mutate(int nCities, double probability);

    static std::vector<int> shuffle_vector(std::vector<int> &vec, int seed);

    int pickOne(const std::vector<double>& fitness, std::mt19937 &gen, std::uniform_real_distribution<> dis);

    void swapTwo(std::vector<int>& vec, int nEl);

    static double findBestDistance(const double *distances, unsigned long n);

};

void SequentialTSP::run(int nCities, unsigned int populationSize, int generations, double mutationProbability, double crossoverProbability, int seed) {
    srand(seed);
    std::ofstream outFile;
    outFile.open ("data.txt");

#if TEST
    // start timer to record the initial population generation
    auto start = std::chrono::system_clock::now();
#endif

    population.resize(populationSize);
    for (unsigned int i = 0; i < populationSize; i++)
        population[i].resize(nCities);

    //generate the cities in the array cities
    // ONE TIME ONLY. DONE SEQUENTIALLY
    cities = generateCities(nCities, seed);


    // I now need a population
    // The population is made of arrays of indexes, aka orders with which I visit the cities
    // The order changes, the cities array remain the same

    // FOR EACH CHUNK, GENERATE POPULATION IN PARALLEL, ALWAYS STARTING FROM AN INITIAL ORDER
    // the initial ordering can be generated with

    for (int i = 0; i < nCities; ++i) {
        population[0][i] = i;
    }

    // ...

    // First chromosome has been populated. Now let's create the rest of the initial population
    for (int i = 1; i < populationSize; ++i) {
        // Every time I shuffle the previous chromosome
        population[i].reserve(nCities);
        population[i] = shuffle_vector(population[i - 1], seed);
    }

#if TEST
    auto end = std::chrono::system_clock::now();
    std::cout << "Initial population generation - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif

    // Now we have an initial population ready
    double bestLocalEval;
    double avgEval;

    for (int iter = 0; iter < generations; ++iter) {

        double lb = calculateLowerBound();

        // Let's calculate the evaluation score
        std::vector<double> evaluation = evaluate();

        // Now calculate fitness (a percentage) based on the evaluation

        auto fitness = calculateFitness(evaluation, lb);

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


        std::vector<std::vector<int>> intermediatePopulation = selection(fitness, population, seed);

        // printPopulation(intermediatePopulation, populationSize, nCities);

        // With the intermediate population I can now crossover the elements


        // printPopulation(intermediatePopulation, populationSize, nCities);
        std::vector<std::vector<int>> nextGen = crossover(intermediatePopulation, nCities, crossoverProbability);

        mutate(nCities, mutationProbability);

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

std::vector<Point> SequentialTSP::generateCities(int nCities, int seed) {
    cities.reserve(nCities);

    std::mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(MIN, MAX);

    for (int i = 0; i < nCities; ++i) {
        cities.emplace_back(round(dis(gen)), round(dis(gen)));
    }
    return cities;
}

std::vector<int> SequentialTSP::shuffle_vector(std::vector<int> &vec, int seed) {
    std::vector<int> copy(vec);
    auto rng = std::default_random_engine(seed);
    std::shuffle(std::begin(copy), std::end(copy), rng);
    return copy;
}

std::vector<double> SequentialTSP::evaluate() {
#if TEST
    auto start = std::chrono::system_clock::now();
#endif
    std::vector<double> evaluation(population.size());
    double totalDistanceOfChromosome;
    for (unsigned int i = 0; i < population.size(); ++i) {
        totalDistanceOfChromosome = 0;
        for (unsigned int j = 0; j < cities.size() - 1; ++j) {
            totalDistanceOfChromosome += cities[population[i][j]].dist(cities[population[i][j + 1]]);
        }
        // Evaluation score of i-th chromosome
        evaluation[i] = totalDistanceOfChromosome;
    }
#if TEST
    auto end = std::chrono::system_clock::now();
    std::cout << "Evaluation - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return evaluation;
}

std::vector<double> SequentialTSP::calculateFitness(std::vector<double> &evaluation, const double &lowerBound) {
#if TEST
    auto start = std::chrono::system_clock::now();
#endif
    std::vector<double> fitness(evaluation.size());
    for (unsigned int i = 0; i < evaluation.size(); ++i) {
        fitness[i] = lowerBound / evaluation[i];
    }
#if TEST
    auto end = std::chrono::system_clock::now();
    std::cout << "Fitness - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return fitness;
}

int SequentialTSP::pickOne(const std::vector<double> &fitness, std::mt19937 &gen, std::uniform_real_distribution<> dis) {
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

std::vector<std::vector<int>>
SequentialTSP::selection(const std::vector<double> &fitness, const std::vector<std::vector<int>> &population,
                         unsigned int seed) {
#if TEST
    auto start = std::chrono::system_clock::now();
#endif

    std::vector<std::vector<int>> selection(population.size());
    double fitnessSum = 0;
    for (int i = 0; i < population.size(); ++i) {
        fitnessSum += fitness[i];
    }
    std::mt19937 gen(seed); //Standard mersenne_twister_engine
    std::uniform_real_distribution<> dis(0, fitnessSum);

    int pickedIndex;
    for (int i = 0; i < population.size(); ++i) {
        pickedIndex = pickOne(fitness, gen, dis);
        // timesPicked[pickedIndex]++;
        selection[i] = population[pickedIndex];
    }

#if TEST
    auto end = std::chrono::system_clock::now();
    std::cout << "Selection - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return selection;
}

std::vector<int> SequentialTSP::recombine(std::vector<int> &chromosomeA, std::vector<int> &chromosomeB, int nEl) {
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

std::vector<std::vector<int>>
SequentialTSP::crossover(std::vector<std::vector<int>> &population, const int nCities, const double crossoverRate) {
#if TEST
    auto start = std::chrono::system_clock::now();
#endif
    std::vector<std::vector<int>> selection(population.size());
    double r;
    for (int i = 0; i < population.size(); ++i) {
        r = (double) rand() / RAND_MAX;
        if (r >= crossoverRate) {
            selection[i] = population[i];
            continue;
        }
        int indexA = rand() % population.size();
        int indexB = rand() % population.size();
        std::vector<int> mateA = population[indexA];
        std::vector<int> mateB = population[indexB];

        selection[i] = recombine(mateA, mateB, nCities);
    }
#if TEST
    auto end = std::chrono::system_clock::now();
    std::cout << "Crossover - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
    return selection;
}

void SequentialTSP::swapTwo(std::vector<int> &vec, const int nEl) {
    int indexA = rand() % nEl;
    int indexB = rand() % nEl;
    while (indexA == indexB) {
        indexB = rand() % nEl;
    }
    int aux = vec[indexA];
    vec[indexA] = vec[indexB];
    vec[indexB] = aux;
}

void SequentialTSP::mutate(const int nCities, const double probability) {
#if TEST
    auto start = std::chrono::system_clock::now();
#endif
    std::vector<int> populationToMutate;
    for (int i = 0; i < population.size(); ++i) {
        populationToMutate = population[i];
        double r = (float) rand() / RAND_MAX;
        if (r < probability) {
            swapTwo(populationToMutate, nCities);
        }
    }
#if TEST
    auto end = std::chrono::system_clock::now();
    std::cout << "Mutation - sequential time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
#endif
}



double SequentialTSP::findBestDistance(const double *distances, unsigned long n) {
    double bestSoFar = MAXFLOAT;
    for (unsigned long i = 0; i < n; ++i) {
        if (distances[i] < bestSoFar) {
            bestSoFar = distances[i];
        }
    }
    return bestSoFar;
}

double SequentialTSP::calculateLowerBound() {
#if TEST
    auto start = std::chrono::system_clock::now();
#endif
    auto evaluation = new utils::Eval[population.size()];
    // Matrix initialization
    auto dstMatrix = new double *[population.size()];
    for (unsigned int i = 0; i < population.size(); ++i) {
        dstMatrix[i] = new double[cities.size() - 1];
    }
    double totalDistanceOfChromosome;
    double dst;
    double lowerBound = 0;
    for (unsigned int i = 0; i < population.size(); ++i) {
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

    // Sort evaluation by score
    std::sort(evaluation, evaluation + population.size(), utils::compareEvaluations);

    // pick ncities-1 distances, one for each of the first ncities-1 dstMatrix rows. Pick every time the smallest of the row
    for (unsigned int i = 0; i < cities.size() - 1; ++i) {
        double bestDistance = findBestDistance(dstMatrix[evaluation[i].dstMatrixIndex], cities.size() - 1);
        lowerBound += bestDistance;
    }
    delete[] evaluation;
    for (unsigned int k = 0; k < population.size(); ++k) {
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


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"

#endif //TSP_GA_SEQUENTIALTSP_H
