#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (int i = 0; i < row.size(); i++) {
            if (i < row.size() - 1) {
                if (row.at(i) >= 0)
                    cout << " + " << row.at(i) << " * x" << i + 1 << '\t';
                else
                    cout << " - " << abs(row.at(i)) << " * x" << i + 1 << '\t';
            }
            else
                cout << " = " << row.at(i) << '\t';

        }
        cout << '\n';
    }
}

void solveByGaussian(vector<vector<double>>& matrix) {
    const int n = matrix.size();

    for (int i = 0; i < n - 1; ++i) {
        #pragma omp parallel for shared(matrix) default(none) schedule(static)
        for (int k = i + 1; k < n; ++k) {
            double factor = matrix[k][i] / matrix[i][i];
            for (int j = i; j < n + 1; ++j) {
                matrix[k][j] -= factor * matrix[i][j];
            }
        }
    }

    vector<double> solution(n);
    for (int i = n - 1; i >= 0; --i) {
        solution[i] = matrix[i][n];
        #pragma omp parallel for shared(matrix, solution) default(none) schedule(static)
        for (int j = i + 1; j < n; ++j) {
            solution[i] -= matrix[i][j] * solution[j];
        }
        solution[i] /= matrix[i][i];
    }

    cout << "Solution:\n";
    for (int i = 0; i < n; ++i) {
        cout << "x" << i + 1 << " = " << solution[i] << '\n';
    }
}

int main() {

    vector<vector<vector<double>>> matrix = {
        {
            {2, 1, -1, 8},
            {-3, -1, 2, -11},
            {-2, 1, 2, -3}
        },
        {
            {-2, 8, 9, -12, 34},
            {8, 11, 3, 1, 39},
            {31, -10, 2, -5, 109},
            {60, -15, -9, -6, 111}
}
    };

    for (int i = 0; i < matrix.size(); i++) {
        cout << '\n' << '#' << i + 1 << "\n\n";
        cout << "Original Matrix:\n";
        printMatrix(matrix[i]);
        solveByGaussian(matrix[i]);
    }

    return 0;
}