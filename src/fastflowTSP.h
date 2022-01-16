//
// Created by dbara on 09/01/22.
//
#ifndef TSP_GA_FASTFLOWTSP_H
#define TSP_GA_FASTFLOWTSP_H

#include "ff/ff.hpp"
#include <memory>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include "lib/Point.h"
#include "lib/utils.h"

class FastflowTSP {
private:
    const double MIN = 0;
    const double MAX = 100;
    std::vector<Point> cities;
    std::vector<std::vector<int>> population;
    std::vector<std::pair<int, int>> chunks;

    void setup(unsigned int nWorkers, unsigned int populationSize, unsigned int nCities, int seed);

    std::vector<Point> generateCities(int nCities, int seed);

    static std::vector<int> shuffle_vector(std::vector<int> &vec, int seed);

    void generateInitialPopulation(int seed);

    static int pickOne(const std::vector<double> &fitness, std::mt19937 &gen, std::uniform_real_distribution<> dis);

    static std::vector<int> recombine(std::vector<int>& chromosomeA, std::vector<int>& chromosomeB, int nEl);

    // FF nodes
    struct LbEmitter : ff::ff_node_t<std::pair<int, int>> {
    private:
        std::vector<std::pair<int, int>> &chunks;
        int generations;
    public:
        LbEmitter(std::vector<std::pair<int, int>> &chunks, int generations) : chunks(chunks),
                                                                               generations(generations) {}

        std::pair<int, int> *svc(std::pair<int, int> *chunk) override {
            if (generations < 1) {
                return this->EOS;
            }
            std::for_each(chunks.begin(), chunks.end(), [&](std::pair<int, int> chunk) {
                this->ff_send_out(new std::pair<int, int>(chunk.first, chunk.second));
            });
            generations--;
            delete chunk;
            return this->GO_ON;
        }
    };


    struct LbWorker : ff::ff_node_t<std::pair<int, int>> {
    private:
        std::vector<utils::Eval> &evaluation;
        std::vector<std::vector<double>> &dstMatrix;
        std::vector<Point> &cities;
        std::vector<std::vector<int>> &population;

    public:
        explicit LbWorker(std::vector<utils::Eval> &evaluation,
                          std::vector<std::vector<double>> &dstMatrix, std::vector<Point> &cities, std::vector<std::vector<int>> &population):
                          evaluation(evaluation),
                          dstMatrix(dstMatrix),
                          cities(cities),
                          population(population) {}

        std::pair<int, int> *svc(std::pair<int, int> *chunk) override {
            int start = chunk->first;
            int end = chunk->second;
            delete chunk;
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

            this->ff_send_out(new std::pair<int, int>(start, end));
            return this->GO_ON;
        };
    };

    struct LbCollectorEvalEmitter : ff::ff_minode_t<std::pair<int, int>> {
    private:
        std::vector<std::pair<int, int>> &chunks;
       double &lb;
        std::vector<Point> &cities;
        std::vector<utils::Eval> &evaluation;
        std::vector<std::vector<double>> &dstMatrix;

        int chunksToArrive;

    public:
        LbCollectorEvalEmitter(std::vector<std::pair<int, int>> &chunks, double &lb, std::vector<Point> &cities, std::vector<utils::Eval> &evaluation, std::vector<std::vector<double>> &dstMatrix) : chunks(chunks), lb(lb), cities(cities), evaluation(evaluation), dstMatrix(dstMatrix), chunksToArrive(chunks.size()){}

        std::pair<int, int> *svc(std::pair<int, int> *chunk) override {
            chunksToArrive--;
            if (chunksToArrive > 0) {
                return this->GO_ON;
            }
            // Sort evaluation by score
            std::sort(evaluation.begin(), evaluation.end(), utils::compareEvaluations);

            lb = 0;
            // pick nCities-1 distances, one for each of the first ncities-1 dstMatrix rows. Pick every time the smallest of the row
            for (unsigned int i = 0; i < cities.size() - 1; ++i) {
                double bestDistance = utils::findBestDistance(dstMatrix[evaluation[i].dstMatrixIndex]);
                lb += bestDistance;
            }

            std::for_each(chunks.begin(), chunks.end(), [&](std::pair<int, int> chunk) {
                this->ff_send_out(new std::pair<int, int>(chunk.first, chunk.second));
            });
            chunksToArrive = chunks.size();
            delete chunk;
            return this->GO_ON;
        }
    };

    struct EvaluateWorker : ff::ff_node_t<std::pair<int, int>> {
    private:
        std::vector<double> &evaluation;
        std::vector<Point> &cities;
        std::vector<std::vector<int>> &population;

    public:
        explicit EvaluateWorker(std::vector<double> &evaluation,
                          std::vector<Point> &cities, std::vector<std::vector<int>> &population):
                evaluation(evaluation),
                cities(cities),
                population(population) {}

        std::pair<int, int> *svc(std::pair<int, int> *chunk) override {
            int start = chunk->first;
            int end = chunk->second;
            delete chunk;
            double totalDistanceOfChromosome;

            for (unsigned int i = start; i <= end; ++i) {
                totalDistanceOfChromosome = 0;
                for (unsigned int j = 0; j < cities.size() - 1; ++j) {
                    totalDistanceOfChromosome += cities[population[i][j]].dist(cities[population[i][j + 1]]);
                }
                // Evaluation score of i-th chromosome
                evaluation[i] = totalDistanceOfChromosome;
            }

            this->ff_send_out(new std::pair<int, int>(start, end));
            return this->GO_ON;
        };
    };

    struct EvaluateCollectorFitnessSelectionEmitter : ff::ff_minode_t<std::pair<int, int>> {
    private:
        std::vector<std::pair<int, int>> &chunks;
        std::vector<double> &evaluation;
        double &lb;
        std::vector<double> &fitness;
        double &fitnessSum;
        int chunksToArrive;

    public:
        EvaluateCollectorFitnessSelectionEmitter(std::vector<std::pair<int, int>> &chunks, std::vector<double> &evaluation, double &lb, std::vector<double> &fitness, double & fitnessSum) : chunks(chunks), evaluation(evaluation), lb(lb), fitness(fitness), fitnessSum(fitnessSum), chunksToArrive(chunks.size()){}

        std::pair<int, int> *svc(std::pair<int, int> *chunk) override {
            chunksToArrive--;
            if (chunksToArrive > 0) {
                return this->GO_ON;
            }

            for (unsigned int i = 0; i < evaluation.size(); ++i) {
                fitness[i] = lb / evaluation[i];
            }
            fitnessSum = 0;
            for (double fitnessValue : fitness) {
                fitnessSum += fitnessValue;
            }


            std::for_each(chunks.begin(), chunks.end(), [&](std::pair<int, int> chunk) {
                this->ff_send_out(new std::pair<int, int>(chunk.first, chunk.second));
            });
            chunksToArrive = chunks.size();
            delete chunk;
            return this->GO_ON;
        }
    };

    struct SelectionWorker : ff::ff_node_t<std::pair<int, int>> {
    private:
        std::vector<std::vector<int>> &population;
        std::vector<double> &fitness;
        double &fitnessSum;
        int seed;
        std::vector<std::vector<int>> &selection;

    public:
        explicit SelectionWorker(std::vector<std::vector<int>> &population, std::vector<double> &fitness, double &fitnessSum,
                                int seed, std::vector<std::vector<int>> &selection):
                population(population),
                fitness(fitness),
                fitnessSum(fitnessSum),
                seed(seed),
                selection(selection) {}

        std::pair<int, int> *svc(std::pair<int, int> *chunk) override {
            int start = chunk->first;
            int end = chunk->second;
            delete chunk;
            std::mt19937 gen(seed); //Standard mersenne_twister_engine
            std::uniform_real_distribution<> dis(0, fitnessSum);
            int pickedIndex;

            for (unsigned long i = start; i <= end; ++i) {
                pickedIndex = pickOne(fitness, gen, dis);
                selection[i] = population[pickedIndex];
            }

            this->ff_send_out(new std::pair<int, int>(start, end));
            return this->GO_ON;
        };
    };

    struct SelectionCollectorCrossoverMutateEmitter : ff::ff_minode_t<std::pair<int, int>> {
    private:
        std::vector<std::pair<int, int>> &chunks;
        std::vector<std::vector<int>> &population;
        std::vector<std::vector<int>> &selection;

        int chunksToArrive;

    public:
        SelectionCollectorCrossoverMutateEmitter(std::vector<std::pair<int, int>> &chunks, std::vector<std::vector<int>> &population, std::vector<std::vector<int>> &selection) : chunks(chunks), population(population), selection(selection), chunksToArrive(chunks.size()){}

        std::pair<int, int> *svc(std::pair<int, int> *chunk) override {
            chunksToArrive--;
            if (chunksToArrive > 0) {
                return this->GO_ON;
            }
            population = std::move(selection);
            selection = std::vector<std::vector<int>>(population.size());

            std::for_each(chunks.begin(), chunks.end(), [&](std::pair<int, int> chunk) {
                this->ff_send_out(new std::pair<int, int>(chunk.first, chunk.second));
            });
            chunksToArrive = chunks.size();
            delete chunk;
            return this->GO_ON;
        }
    };

    struct CrossoverMutateWorker : ff::ff_node_t<std::pair<int, int>> {
    private:
        double crossoverRate;
        double mutationRate;
        std::vector<std::vector<int>> &population;
        std::vector<std::vector<int>> &newPopulation;
        std::vector<Point> &cities;

    public:
        explicit CrossoverMutateWorker(double crossoverProbability, double mutationProbability, std::vector<std::vector<int>> &population, std::vector<std::vector<int>> &newPopulation, std::vector<Point> &cities):
                crossoverRate(crossoverProbability),
                mutationRate(mutationProbability),
                population(population),
                newPopulation(newPopulation),
                cities(cities){}

        std::pair<int, int> *svc(std::pair<int, int> *chunk) override {
            int start = chunk->first;
            int end = chunk->second;
            delete chunk;
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

                // Now Mutate
                r = (float) rand() / RAND_MAX;
                if (r < mutationRate) {
                    utils::swapTwo(newPopulation[i]);
                } // else, do nothing, aka do not mutate
            }

            this->ff_send_out(new std::pair<int, int>(start, end));
            return this->GO_ON;
        };
    };

    struct CrossoverMutateCollector : ff::ff_minode_t<std::pair<int, int>> {
    private:
        std::vector<std::pair<int, int>> &chunks;
        std::vector<std::vector<int>> &population;
        std::vector<std::vector<int>> &newPopulation;

        int chunksToArrive;

    public:
        CrossoverMutateCollector(std::vector<std::pair<int, int>> &chunks, std::vector<std::vector<int>> &population, std::vector<std::vector<int>> &newPopulation) : chunks(chunks), population(population), newPopulation(newPopulation), chunksToArrive(chunks.size()){}

        std::pair<int, int> *svc(std::pair<int, int> *chunk) override {
            chunksToArrive--;
            if (chunksToArrive > 0) {

                return this->GO_ON;
            }
            population = std::move(newPopulation);
            newPopulation = std::vector<std::vector<int>>(population.size());
            this->ff_send_out(new std::pair<int, int>(chunk->first, chunk->second));
            chunksToArrive = chunks.size();
            delete chunk;
            return this->GO_ON;
        }
    };

    public:
        int run(int nCities, unsigned int populationSize, int generations, double mutationProbability,
                double crossoverProbability, unsigned int nWorkers, int seed);
};

inline void FastflowTSP::setup(unsigned int nWorkers, unsigned int populationSize, unsigned int nCities, int seed) {
    srand(seed);

    // setting the right amount of workers
    auto const hardware_threads =
            std::thread::hardware_concurrency();

    auto properWorkers = std::min(hardware_threads-1!=0?hardware_threads-1:2, nWorkers);

    if (properWorkers > populationSize) {
        properWorkers = populationSize;
    }

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

inline std::vector<Point> FastflowTSP::generateCities(int nCities, int seed) {
    cities.reserve(nCities);

    std::mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(MIN, MAX);

    for (int i = 0; i < nCities; ++i) {
        cities.emplace_back(round(dis(gen)), round(dis(gen)));
    }
    return cities;
}

inline std::vector<int> FastflowTSP::shuffle_vector(std::vector<int> &vec, int seed) {
    std::vector<int> copy(vec);
    auto rng = std::default_random_engine(seed);
    std::shuffle(std::begin(copy), std::end(copy), rng);
    return copy;
}

inline void FastflowTSP::generateInitialPopulation(int seed) {
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

inline int FastflowTSP::pickOne(const std::vector<double> &fitness, std::mt19937 &gen, std::uniform_real_distribution<> dis) {
    // Get distribution in 0...fitnessSum
    double r = dis(gen);
    int i = 0;
    // sum(fitness) >= r so I don't need to check if I go out of bound
    while (r > 0) {
        r -= fitness[i];
        i++;
    }
    return i==0?0:--i;
}

inline std::vector<int> FastflowTSP::recombine(std::vector<int> &chromosomeA, std::vector<int> &chromosomeB, int nEl) {
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

int FastflowTSP::run(int nCities, unsigned int populationSize, int generations, double mutationProbability,
                     double crossoverProbability, unsigned int nWorkers, int seed) {
    if (nCities > populationSize){
        std::cout << "[ERROR] nCities cannot be greater than populationSize" << std::endl;
        return 1;
    }
    std::ofstream outFile;
    outFile.open ("data_ff.txt", std::ofstream::trunc);

    setup(nWorkers, populationSize, nCities, seed);

    //generate the cities in the array cities
    // ONE TIME ONLY. DONE SEQUENTIALLY
    cities = generateCities(nCities, seed);

    // I now need a population
    // ONE TIME ONLY. DONE SEQUENTIALLY
    generateInitialPopulation(seed);

    double lb = 0;
    LbEmitter lbEmitter(chunks, generations);

    std::vector<utils::Eval> lbEvaluation(population.size());

    // distance matrix initialization
    std::vector<std::vector<double>> dstMatrix(population.size());
    for (unsigned int i = 0; i < population.size(); ++i) {
        dstMatrix[i] = std::vector<double>(cities.size() - 1);
    }

    std::vector<std::unique_ptr<ff::ff_node>> lbWorkers;
    for (int i = 0; i < nWorkers; ++i) {
        lbWorkers.push_back(std::make_unique<LbWorker>(lbEvaluation, dstMatrix, cities, population));
    }

    ff::ff_Farm<std::pair<int, int>> lbFarm(std::move(lbWorkers), lbEmitter);
    lbFarm.remove_collector();

    LbCollectorEvalEmitter lbCollectorEvalEmitter(chunks, lb, cities, lbEvaluation, dstMatrix);

    std::vector<double> evaluation(population.size());
    std::vector<std::unique_ptr<ff::ff_node>> evaluateWorkers;
    for (int i = 0; i < nWorkers; ++i) {
        evaluateWorkers.push_back(std::make_unique<EvaluateWorker>(evaluation, cities, population));
    }

    ff::ff_Farm<std::pair<int, int>> evaluateFarm(std::move(evaluateWorkers), lbCollectorEvalEmitter);
    evaluateFarm.remove_collector();

    std::vector<double> fitness(evaluation.size());
    double fitnessSum = 0;
    EvaluateCollectorFitnessSelectionEmitter evaluateCollectorFitnessSelectionEmitter(chunks, evaluation, lb, fitness, fitnessSum);


    std::vector<std::vector<int>> selection(population.size());
    std::vector<std::unique_ptr<ff::ff_node>> selectionWorkers;
    for (int i = 0; i < nWorkers; ++i) {
        selectionWorkers.push_back(std::make_unique<SelectionWorker>(population, fitness, fitnessSum, seed, selection));
    }

    ff::ff_Farm<std::pair<int, int>> selectionFarm(std::move(selectionWorkers), evaluateCollectorFitnessSelectionEmitter);
    selectionFarm.remove_collector();

    SelectionCollectorCrossoverMutateEmitter selectionCollectorCrossoverMutateEmitter(chunks, population, selection);

    std::vector<std::vector<int>> newPopulation(population.size());

    std::vector<std::unique_ptr<ff::ff_node>> crossoverMutateWorkers;
    for (int i = 0; i < nWorkers; ++i) {
        crossoverMutateWorkers.push_back(std::make_unique<CrossoverMutateWorker>(crossoverProbability, mutationProbability, population, newPopulation, cities));
    }

    CrossoverMutateCollector crossoverMutateCollector(chunks, population, newPopulation);

    ff::ff_Farm<std::pair<int, int>> crossoverMutateFarm(std::move(crossoverMutateWorkers), selectionCollectorCrossoverMutateEmitter, crossoverMutateCollector);

    ff::ff_Pipe<std::pair<int, int>> pipe(lbFarm, evaluateFarm, selectionFarm, crossoverMutateFarm);
    pipe.wrap_around();

#if MEASURE == true
    auto start = std::chrono::system_clock::now();
#endif
    if (pipe.run_and_wait_end() < 0) {
        ff::error("running pipe");
        return 1;
    }
#if MEASURE == true
    auto end = std::chrono::system_clock::now();
    std::cout << nWorkers << " " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
#endif
    return 0;

}

#endif //TSP_GA_FASTFLOWTSP_H
