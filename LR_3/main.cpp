#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <fstream>
#include <cstdio>
#include <climits>

using namespace std;

void Task14(int** matrix, int matrix_size, int* out_matrix, int result_matrix_size) {
    int min = INT_MAX;
    bool stop;

    for (int i = 0; i < matrix_size - result_matrix_size; i++) {
        stop = true;

        int* row = new int[result_matrix_size + 1];
        row[0] = i;
        for (int k = 1; k < result_matrix_size; k++) row[k] = (row[k - 1] + 1);

        while (stop) {
            int* clm = new int[result_matrix_size];
            bool stop2 = true;
            int elem = result_matrix_size - 1;
            int lastElem = 0;

            for (int j = 0; j < result_matrix_size; j++) clm[j] = j;

            while (stop2) {
                int minMost = 0;
                for (int r = 0; r < result_matrix_size; r++) {
                    for (int c = 0; c < r + 1; c++) {
                        minMost += matrix[row[r]][clm[c]];
                        if (minMost > min)break;
                    }
                    if (minMost > min) break;
                }

                if (minMost < min) {
                    min = minMost;
                    int idxs = 0;
                    for (int m = 0; m < result_matrix_size; m++) {
                        for (int v = 0; v < m + 1; v++) {
                            out_matrix[idxs] = row[m];
                            out_matrix[idxs + 1] = clm[v];
                            idxs += 2;
                        }
                    }
                }

                int elemT = result_matrix_size - 1;
                int lastElemT = 0;
                clm[elemT]++;

                while (clm[elemT] >= (matrix_size - lastElemT)) {
                    if (elemT <= 0 && clm[0] <= (matrix_size - lastElemT)) {
                        stop2 = false;
                        break;
                    }
                    clm[elemT - 1]++;
                    for (int k = elemT; k < result_matrix_size; k++) clm[k] = clm[k - 1] + 1;
                    elemT--;
                    lastElemT++;
                }
            }

            row[elem]++;
            while (row[elem] == (matrix_size - lastElem)) {
                if (elem <= 1 && row[1] <= (matrix_size - lastElem)) {
                    stop = false;
                    break;
                }
                row[elem - 1]++;
                for (int k = elem; k < result_matrix_size; k++) row[k] = row[k - 1] + 1;
                elem--;
                lastElem++;
            }
        }
    }
}

bool ComparisonResults(int* single_mass, int* parallel_mass, int size_array) {
    for (int i = 0; i < size_array; i++)
        if (single_mass[i] != parallel_mass[i])
            return false;
    return true;
}

int main(int argc, char* argv[]) {
    srand((unsigned)time(NULL));

    int mpiReturn = MPI_Init(&argc, &argv);
    if (mpiReturn) {
        printf("\nError MPI\n");
        MPI_Abort(MPI_COMM_WORLD, mpiReturn);
    }

    int matrix_size, triangle_matrix_size, MPI_rank;
    double parallel_time, single_time;

    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);

    if (MPI_rank == 0) {

        cout << "Enter input matrix size - ";
        cin >> matrix_size;
        cout << "Enter triangle matrix size - ";
        cin >> triangle_matrix_size;
    }

    int** matrix = new int* [matrix_size];

    for (int i = 0; i < matrix_size; i++)
        matrix[i] = new int[matrix_size];

    for (int i = 0; i < matrix_size; i++)
        for (int j = 0; j < matrix_size; j++)
            matrix[i][j] = rand() % (10);

    int out_size = 0;
    for (int i = 0; i <= triangle_matrix_size; i++) out_size += i;

    int* matrix_single = new int[out_size * 2];
    int* matrix_parallel = new int[out_size * 2];

    if (MPI_rank == 0) {
        parallel_time = omp_get_wtime();
        single_time = omp_get_wtime();
    }

    Task14(matrix, matrix_size, matrix_parallel, triangle_matrix_size);

    if (!MPI_rank) parallel_time = omp_get_wtime() - parallel_time;

    MPI_Finalize();

    if (!MPI_rank) {
        Task14(matrix, matrix_size, matrix_single, triangle_matrix_size);

        single_time = omp_get_wtime() - single_time;

        printf("Time of single:   %f\n", single_time);
        printf("Time of parallel: %f\n", parallel_time);

        if (ComparisonResults(matrix_parallel, matrix_single, out_size))
            cout << "Is equal" << endl;
        else
            cout << "Is not equal" << endl;

        bool save;

        cout << "Save result to file? (0 or 1) - ";
        cin >> save;

        if (save == true) {
            ofstream fout("Result.txt", ios::app);
            fout << endl << "=====================================================" << endl << endl;
            fout << "Matrix:" << endl << endl;
            for (int i = 0; i < matrix_size; i++) {
                for (int j = 0; j < matrix_size; j++)
                    fout << matrix[i][j] << "   ";
                fout << endl;
            }
            fout << endl << "Triangle matrix index (single):" << endl << endl;
            for (int i = 0; i < out_size * 2; i += 2) {
                fout << matrix_single[i] << " " << matrix_single[i + 1] << "   ";
                if (matrix_single[i] != matrix_single[i + 2] && i < (out_size * 2) - 2) fout << endl;
            }
            fout << endl << endl << "Triangle matrix(single):" << endl << endl;
            for (int i = 0; i < out_size * 2; i += 2) {
                fout << matrix[matrix_single[i]][matrix_single[i + 1]] << "   ";
                if (matrix_single[i] != matrix_single[i + 2] && i < (out_size * 2) - 2) fout << endl;
            }
            fout << endl << endl << "Triangle matrix index (parallel):" << endl << endl;
            for (int i = 0; i < out_size * 2; i += 2) {
                fout << matrix_parallel[i] << " " << matrix_parallel[i + 1] << "   ";
                if (matrix_parallel[i] != matrix_parallel[i + 2] && i < (out_size * 2) - 2) fout << endl;
            }
            fout << endl << endl << "Triangle matrix(parallel):" << endl << endl;
            for (int i = 0; i < out_size * 2; i += 2) {
                fout << matrix[matrix_parallel[i]][matrix_parallel[i + 1]] << "   ";
                if (matrix_parallel[i] != matrix_parallel[i + 2] && i < (out_size * 2) - 2) fout << endl;
            }
            fout.close();
        }
    }
    return 0;
}