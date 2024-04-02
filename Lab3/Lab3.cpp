#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

using namespace std;

const int N = 3000;
vector<double> solution(N);

int get_random_number(int min, int max) {
    return rand() % (max - min + 1) + min;
}

void solve_gauss(double** matrix) {
    #pragma omp parallel
    for (int i = 0; i < N - 1; i++)
    {
        #pragma omp for
        for (int j = i + 1; j < N; j++) {
            double coeff = matrix[j][i] / matrix[i][i];
            for (int k = i; k < N + 1; k++) {
                matrix[j][k] = matrix[j][k] - coeff * matrix[i][k];
            }
        }
    }

    for (int i = N - 1; i >= 0; i--) {
        solution[i] = matrix[i][N];
        for (int j = i + 1; j < N; j++) {
            solution[i] -= matrix[i][j] * solution[j];
        }
        solution[i] = solution[i] / matrix[i][i];
    }

}

int main() {

    double** matrix = new double* [N];
    for (int i = 0; i < N; ++i) {
        matrix[i] = new double[N + 1];
        for (int j = 0; j < N + 1; ++j) {
            matrix[i][j] = get_random_number(1, 20);
        }
    }

    auto begin = std::chrono::steady_clock::now();
    solve_gauss(matrix);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> solving_time = end - begin;
    cout << "Solution:\n";
    for (int i = 0; i < N; ++i) {
        cout << "x" << i + 1 << " = " << solution[i] << '\n';
    }
    cout << "Solution time: " << solving_time.count() << endl;
    
    for (int i = 0; i < N; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;

    return 0;
}