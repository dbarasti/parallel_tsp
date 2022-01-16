//
// Created by dbara on 14/10/21.
//

#ifndef TSP_GA_UTILS_H
#define TSP_GA_UTILS_H

namespace utils{
    struct Eval {
        unsigned long dstMatrixIndex;
        double score;
    };

    inline bool compareEvaluations(utils::Eval a, utils::Eval b);
    inline void swapTwo(std::vector<int> &vec);
    inline void computeAvgEval(std::ofstream &outFile, int iter, const std::vector<double> &evaluation);
    inline double findBestDistance(const std::vector<double>& distances);
}

bool utils::compareEvaluations(utils::Eval a, utils::Eval b) {
    return a.score < b.score;
}

void utils::swapTwo(std::vector<int> &vec) {
    int indexA = rand() % vec.size();
    int indexB = rand() % vec.size();
    while (indexA == indexB) {
        indexB = rand() % vec.size();
    }
    int aux = vec[indexA];
    vec[indexA] = vec[indexB];
    vec[indexB] = aux;
}

void utils::computeAvgEval(std::ofstream &outFile, int iter, const std::vector<double> &evaluation){
    double bestLocalEval;
    double avgEval;
    avgEval = 0;
    bestLocalEval = 0;
    for (double eval : evaluation) {
        avgEval += eval;
        if (eval > bestLocalEval) {
            bestLocalEval = eval;
        }
    }
    avgEval = avgEval / evaluation.size();

    outFile << iter << "\t" << avgEval << std::endl;
}

inline double utils::findBestDistance(const std::vector<double>& distances) {
    double bestSoFar = 1000000;
    for (double distance : distances) {
        if (distance < bestSoFar) {
            bestSoFar = distance;
        }
    }
    return bestSoFar;
}


#endif //TSP_GA_UTILS_H
