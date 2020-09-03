#include <iostream>
#include <cmath>
#include <random>
#include <functional>

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
        return sqrt(xd*xd + yd*yd);
    }

    // Add or subtract two points.
    Point add(Point b)
    {
        return Point(xval + b.xval, yval + b.yval);
    }
    Point sub(Point b)
    {
        return Point(xval - b.xval, yval - b.yval);
    }

    // Move the existing point.
    void move(double a, double b)
    {
        xval += a;
        yval += b;
    }

    // Print the point on the stream.  The class ostream is a base class
    // for output streams of various types.
    void print(ostream &strm)
    {
        strm << "(" << xval << "," << yval << ")";
    }
};

Point* generateCities(const int nCities, double min, double max, unsigned int seed) {
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

    // To obtain a time-based seed
    int * copy = new int[n];
    for (int i = 0; i < n; ++i) {
        copy[i] = arr[i];
    }
    // Shuffling our array
    shuffle(copy, copy + n,
            default_random_engine(seed));
    return copy;
}

double * evaluate(int ** population, Point * cities, const int populationSize, const int nCities){
    auto evaluation = new double[populationSize];
    double totalDistance;
    for (int i = 0; i < populationSize; ++i) {
        totalDistance = 0;
        for (int j = 0; j < nCities-1; ++j) {
            totalDistance +=  cities[population[i][j]].dist(cities[population[i][j+1]]);
        }
        evaluation[i] = totalDistance;
        cout << "Total distance for chromosome " << i << ": " << totalDistance << endl;
    }
    return evaluation;
}

double *calculateFitness(const double *evaluation, const int populationSize) {
    auto fitness = new double[populationSize];
    double totalInverse = 0;
    for (int i = 0; i < populationSize; ++i) {
        fitness[i] = 1/evaluation[i];
        totalInverse += fitness[i];
    }
    double totalFitness = 0;
    for (int i = 0; i < populationSize; ++i) {
        fitness[i] = fitness[i] / totalInverse;
        cout << "Fitness for chromosome " << i << ": " << fitness[i] << endl;
        totalFitness += fitness[i];
    }
    cout << "Total fitness: " << totalFitness << endl;
    return fitness;
}

int pickOne(const double * fitness){
    // Get rnd number 0..1
    double r = (float) rand()/RAND_MAX;
    int i = 0;
    // sum(fitness) is == 1 so I don't need to check if I go out of bound
    while(r > 0){
        r -= fitness[i];
        i++;
    }
    return --i;
}

int **selection(double *fitness, int** population, const int populationSize) {
    int ** selection = new int*[populationSize];
    for (int i = 0; i < populationSize; ++i) {
        selection[i] = population[pickOne(fitness)];
    }
    return selection;
}

int main() {
    const int nCities = 5;
    const double min = 0;
    const double max = 100;
    const unsigned int seed = 35412;

    //generate the cities in the array cities
    Point* cities = generateCities(nCities, min, max, seed);
    for (int i = 0; i < nCities; ++i) {
        cities[i].print(cout);
    }
    std::cout << endl;

    int const populationSize = 5;

    int **population;
    population = new int *[populationSize];
    for(int i = 0; i <populationSize; i++)
        population[i] = new int[nCities];

    // I now need a population
    // The population is made of arrays of indexes, aka orders with which I visit the cities
    // The order changes, the cities remain the same

    // I create an initial order which is 0,1,...,nCities-1, then I shuffle this initial order to obtain other populationSize-1 orders.

    for (int i = 0; i < nCities; ++i) {
        population[0][i] = i;
    }

    // First chromosome has been populated. Now let's create the rest of the initial population

    for (int i = 1; i < populationSize; ++i) {
        int* a = shuffle_array(population[i - 1], nCities, seed);
        for (int j = 0; j < nCities; ++j) {
            population[i][j] = a[j];
        }
    }

    for (int i = 0; i < populationSize; ++i) {
        // Printing our array
        for (int j = 0; j < nCities; ++j)
            cout << population[i][j] << " ";
        cout << endl;
    }

    // Now we have an initial population ready

    // Let's calculate the fitness

    double * evaluation = evaluate(population, cities, populationSize, nCities);

    // Now calculate fitness (a percentage) based on the evaluation

    double * fitness = calculateFitness(evaluation, populationSize);

    // It's time for reproduction!

    // Let's find the intermediate population, aka chromosomes that will be recombined (and then mutated) to be the next generation

    // Select populationSize elements so that the higher the fitness the higher the probability to be selected

    int ** intermediatePopulation = selection(fitness, population, populationSize);

    for (int i = 0; i < populationSize; ++i) {
        cout << "Intermediate chromosome " << i << ":";
        for (int j = 0; j < nCities; ++j) {
            cout << " " << intermediatePopulation[i][j];
        }
        cout << endl;
    }

    // With the intermediate population I can now crossover the elements


    return 0;
}