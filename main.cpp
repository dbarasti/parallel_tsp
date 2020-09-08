#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <ctime>

using namespace std;

// Class to represent points.
class Point {
private:
    double xval, yval;
public:
    // Constructor uses default arguments to allow calling with zero, one,
    // or two values.
    Point(double x = 0.0, double y = 0.0) {
        xval = x;
        yval = y;
    }

    // Extractors.
    double x() { return xval; }

    double y() { return yval; }

    // Distance to another point.  Pythagorean thm.
    double dist(Point other) {
        double xd = xval - other.xval;
        double yd = yval - other.yval;
        return sqrt(xd * xd + yd * yd);
    }

    // Add or subtract two points.
    Point add(Point b) {
        return Point(xval + b.xval, yval + b.yval);
    }

    Point sub(Point b) {
        return Point(xval - b.xval, yval - b.yval);
    }

    // Move the existing point.
    void move(double a, double b) {
        xval += a;
        yval += b;
    }

    // Print the point on the stream.  The class ostream is a base class
    // for output streams of various types.
    void print(ostream &strm) {
        strm << "(" << xval << "," << yval << ")";
    }
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

struct Eval {
    int dstMatrixIndex;
    double score;
};

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
    return evaluation;
}

double *calculateFitness(const double *evaluation, const int populationSize) {
    auto fitness = new double[populationSize];
    double totalInverse = 0;
    for (int i = 0; i < populationSize; ++i) {
        fitness[i] = 1 / (pow(evaluation[i], 8) + 1);
        totalInverse += fitness[i];
    }
//    double totalFitness = 0;
    for (int i = 0; i < populationSize; ++i) {
        fitness[i] = fitness[i] / totalInverse;
//        totalFitness += fitness[i];
//        cout << "Fitness for chromosome " << i << ": " << fitness[i] << endl;
    }
//    cout << "Fitness sum: " << totalFitness << endl;
    return fitness;
}

int pickOne(const double *fitness, std::mt19937 &gen, std::uniform_real_distribution<> dis) {
    // Get distribution in 0...fitnessSum
    double r = dis(gen);
    int i = 0;
    // sum(fitness) is >= r so I don't need to check if I go out of bound
    while (r > 0) {
        r -= fitness[i];
        i++;
    }
    return --i;
}

int **selection(double *fitness, int **population, const int populationSize, unsigned int seed) {
    int **selection = new int *[populationSize];
//    double fitnessSum = 0;
//    for (int i = 0; i < populationSize; ++i) {
//        fitnessSum += fitness[i];
//    }
    std::mt19937 gen(seed); //Standard mersenne_twister_engine
    std::uniform_real_distribution<> dis(0, 1);

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

int **crossover(int **population, const int populationSize, const int nCities) {
    int **selection = new int *[populationSize];
    for (int i = 0; i < populationSize; ++i) {
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

void mutate(int **population, const int populationSize, const int nCities, const double probability) {
    int *populationToMutate;
    for (int i = 0; i < populationSize; ++i) {
        populationToMutate = population[i];
        double r = (float) rand() / RAND_MAX;
        if (r < probability) {
            swapTwo(populationToMutate, nCities);
        }
    }
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
    return lowerBound;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"

int main() {
    const int nCities = 15;
    int const populationSize = 300;
    const double min = 0;
    const double max = 100;
    const unsigned int seed = 35412;
    const double mutationProbability = 0.01;
    srand(seed);

    //generate the cities in the array cities
    Point *cities = generateCities(nCities, min, max, seed);
    for (int i = 0; i < nCities; ++i) {
        cities[i].print(cout);
    }
    cout << endl;

    //Defining the matrix where I store population of orders:

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

    // Now we have an initial population ready

    double bestGlobalFitness = 0;
    double bestLocalFitness;
    int globalBestIndex;
    int localBestIndex;

    double lb = calculateLowerBound(population, cities, populationSize, nCities);
    cout << "Lower bound: " << lb << endl;


    while (true) {

        // Let's calculate the evaluation score

        double *evaluation = evaluate(population, cities, populationSize, nCities);

        // Now calculate fitness (a percentage) based on the evaluation

        double *fitness = calculateFitness(evaluation, populationSize);

        bestLocalFitness = 0;
        for (int i = 0; i < populationSize; ++i) {
            if (fitness[i] > bestLocalFitness) {
                bestLocalFitness = fitness[i];
                localBestIndex = i;
            }
        }

        if (bestLocalFitness > bestGlobalFitness) {
            bestGlobalFitness = bestLocalFitness;
            globalBestIndex = localBestIndex;
            auto now = std::chrono::system_clock::now();
            std::time_t now_time = std::chrono::system_clock::to_time_t(now);
            cout << "new best fitness score: " << bestGlobalFitness << " with evaluation of "
                 << evaluation[globalBestIndex]
                 << " found at " << std::ctime(&now_time) << endl;
        }

        // It's time for reproduction!

        // Let's find the intermediate population, aka chromosomes that will be recombined (and then mutated) to be the next generation

        // Select populationSize elements so that the higher the fitness the higher the probability to be selected

        int **intermediatePopulation = selection(fitness, population, populationSize, seed);

        // printPopulation(intermediatePopulation, populationSize, nCities);

        // With the intermediate population I can now crossover the elements
        int **nextGen = crossover(intermediatePopulation, populationSize, nCities);

        mutate(nextGen, populationSize, nCities, mutationProbability);

        // printPopulation(nextGen, populationSize, nCities);

        delete[] intermediatePopulation;
        delete[] fitness;
        delete[] evaluation;
        for (int k = 0; k < populationSize; ++k) {
            delete[] population[k];
        }
        delete[] population;
        population = nextGen;
    }

    delete[] cities;
}


#pragma clang diagnostic pop