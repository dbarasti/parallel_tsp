# Parallelizig the Genetic Travelling Salesperson Problem
The Travelling Salesperson Problem (TSP) is a well-known optimization problem requiring to find the shortest possible path visiting a list of nodes exactly once. Being the TSP a NP-hard problem, a common solution is to find a “good enough” path using some optimization algorithm. One of the flavours of the TSP is to use a genetic algorithm to find a solution. I developed and described two implementations of the genetic TSP, parallelizing the stages of each generation of the algorithm: the first parallel implementation uses unstructured,  low-level mechanisms of C++ (\textit{futures} and \textit{async}), whereas the second exploits the FastFlow library, approaching the problem in a more structured way. As a baseline for the performance evaluation, I developed a sequential version of the genetic algorithm as well. After a description of the implementations, I used a performance model to evaluate and compare the proposed solutions.


## Building and Running
The project can be compiled and executed using Cmake or gcc.

If CMake is installed type:
`mkdir build && cd build `


Now to build the project type: 
`cmake --build .` 

Otherwise, to use GCC use: 

`g++ -std=c++17 -O3 main.cpp -o tsp-gen -pthread -DBLOCKING_MODE`

To run the program correctly, you need to pass to the executable the following parameters: 

```./tsp-gen nCities populationSize generations mutationProbability crossoverProbability nWorkers [seed]```

## Acknowledgments
This project was developed for the course of [Parallel and Distributed Systems](http://didawiki.di.unipi.it/doku.php/magistraleinformaticanetworking/spm/sdpm09support) at University of Pisa under the guide of [Prof. Marco Danelutto](http://calvados.di.unipi.it/paragroup/danelutto/) and [Prof. Massimo Torquati](http://calvados.di.unipi.it/paragroup/torquati/).
