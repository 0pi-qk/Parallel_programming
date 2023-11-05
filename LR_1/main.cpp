#include <iostream>
#include <cstdlib>
#include "omp.h"

using namespace std;

void merge(int* X, int n, int* tmp) {
    int i = 0;
    int j = n / 2;
    int ti = 0;

    while (i < n / 2 && j < n) {
        if (X[i] < X[j]) {
            tmp[ti] = X[i];
            ti++; i++;
        }
        else {
            tmp[ti] = X[j];
            ti++;
            j++;
        }
    }
    while (i < n / 2) {
        tmp[ti] = X[i];
        ti++;
        i++;
    }
    while (j < n) {
        tmp[ti] = X[j];
        ti++;
        j++;
    }

    memcpy(X, tmp, n * sizeof(int));
}

void mergesort_single(int* X, int n, int* tmp){
    if (n < 2) return;
    mergesort_single(X, n / 2, tmp);
    mergesort_single(X + (n / 2), n - (n / 2), tmp);
    merge(X, n, tmp);
}

bool comparison_of_results(int* single_mass, int* parallel_mass, int n) {
    for (int i = 0; i < n; i++)
        if (single_mass[i] != parallel_mass[i])
            return false;
    return true;
}

void mergesort_parallel (int a[], int size, int temp[], int threads) {
    if (threads == 1) mergesort_single(a, size, temp);
    else if (threads > 1) {
#pragma omp parallel sections
        {
#pragma omp section
            mergesort_parallel(a, size / 2, temp, threads / 2);
#pragma omp section
            mergesort_parallel(a + size / 2, size - size / 2,
                temp + size / 2, threads - threads / 2);
        }
        merge(a, size, temp);
    }
}

void start_test(int n) {
    double single_time, parallel_time;

    int* data = (int*)malloc(n * sizeof(int) + 1);
    int* data1 = (int*)malloc(n * sizeof(int) + 1);
    int* tmp = (int*)malloc(n * sizeof(int) + 1);
    int* tmp1 = (int*)malloc(n * sizeof(int) + 1);

    for (int i = 0; i < n; i++) {
        data1[i] = data[i] = rand() % n;
    }

    parallel_time = omp_get_wtime();
    mergesort_parallel(data, n, tmp, omp_get_max_threads());
    parallel_time = omp_get_wtime() - parallel_time;

    single_time = omp_get_wtime();
    mergesort_single(data1, n, tmp1);
    single_time = omp_get_wtime() - single_time;

    cout << "Single time:   " << single_time << endl;
    cout << "Parallel time:   " << parallel_time << endl;

    if (comparison_of_results(data, data1, n))
        cout << "Is equal" << endl;
    else
        cout << "Is not equal" << endl;

    free(tmp);
    free(tmp1);
    free(data);
    free(data1);
}


int main() {
    bool menu = true;
    int key, n;
    while (menu) {
        cout << "1|  Manual test mode" << endl;
        cout << "2|  Automatic test mode" << endl;
        cout << "3|  Out" << endl;
        cout << "-> ";
        cin >> key;
        cout << "==================================================";
        switch (key) {
        case 1: {
            cout << endl << "Enter list size - ";
            cin >> n;
            start_test(n);
            cout << "==================================================" << endl;
            break;
        }
        case 2: {
            for (int nn = 10; nn < 9000001;) {
                cout << endl << "List size - " << nn << endl;
                start_test(nn);
                cout << "==================================================";
                if (nn < 1000000) nn *= 10;
                else nn += 1000000;
            }
            cout << endl;
            break;
        }
        case 3: {
            menu = false;
            break;
        }
        default:
            break;
        }
    }
}

