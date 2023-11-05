#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <omp.h>

using namespace std;

void SaveFileData(int* data, int size_array) {
	ofstream fout("Result.txt", ios::app);
	fout << "___";
	for (int i = 0; i < size_array; i++)
		fout << data[i] << " ";
	fout << endl;
	fout.close();
}

void SaveFileSeparator() {
	ofstream fout("Result.txt", ios::app);
	fout << endl << "====== 1 - Initial; 2 - Parallel; 3 - Single ======" << endl;
	fout.close();
}

__device__ void ParallelSwap(int* a, int* b) {
	const int t = *a;
	*a = *b;
	*b = t;
}

__device__ void ParallelHeapify(int* maxHeap, int heapSize, int idx) {
	int largest = idx;
	int left = 2 * idx + 1;
	int right = 2 * idx + 2;
	if (left < heapSize && maxHeap[left] > maxHeap[largest]) {
		largest = left;
	}

	if (right < heapSize && maxHeap[right] > maxHeap[largest]) {
		largest = right;
	}

	if (largest != idx) {
		ParallelSwap(&maxHeap[idx], &maxHeap[largest]);
		ParallelHeapify(maxHeap, heapSize, largest);
	}
}

__global__ void ParallelSort(int* iA, const int size_array) {
	iA = iA + blockIdx.x * size_array;

	__syncthreads();

	if (threadIdx.x == 0) {
		for (int i = size_array / 2 - 1; i >= 0; i--)
			ParallelHeapify(iA, size_array, i);

		for (int i = size_array - 1; i >= 0; i--) {
			ParallelSwap(&iA[0], &iA[i]);
			ParallelHeapify(iA, i, 0);
		}
	}
}

void SingleSwap(int* a, int* b) {
	const int t = *a;
	*a = *b;
	*b = t;
}

void SingleHeapify(int data[], int size_array, int i) {
	int largest = i;
	int l = 2 * i + 1;
	int r = 2 * i + 2;

	if (l < size_array && data[l] > data[largest])
		largest = l;

	if (r < size_array && data[r] > data[largest])
		largest = r;

	if (largest != i) {
		SingleSwap(&data[i], &data[largest]);
		SingleHeapify(data, size_array, largest);
	}
}

void SingleSort(int data[], int size_array) {
	for (int i = size_array / 2 - 1; i >= 0; i--)
		SingleHeapify(data, size_array, i);

	for (int i = size_array - 1; i >= 0; i--) {
		SingleSwap(&data[0], &data[i]);
		SingleHeapify(data, i, 0);
	}
}

bool ComparisonResults(int* single_mass, int* parallel_mass, int size_array) {
	for (int i = 0; i < size_array; i++)
		if (single_mass[i] != parallel_mass[i])
			return false;
	return true;
}

void StartTest(int size_array, bool save) {
	srand((unsigned)time(NULL));

	if (save == true) SaveFileSeparator();

	double parallel_time, single_time;

	int* data_parallel = (int*)malloc(size_array * sizeof(int));
	int* data_single = (int*)malloc(size_array * sizeof(int));

	int* temp_data_parallel = NULL;
	cudaMalloc((void**)&temp_data_parallel, size_array * sizeof(int));

	for (int i = 0; i < size_array; ++i)
		data_single[i] = data_parallel[i] = rand() % 100 + 1;

	if (save == true) SaveFileData(data_parallel, size_array);

	cudaMemcpy(temp_data_parallel, data_parallel, size_array * sizeof(int), cudaMemcpyHostToDevice);

	int threads, blocks;

	if (size_array < 256) {
		threads = 1024;
		blocks = 1024;
	}
	else {
		blocks = 262144 / size_array;
		if (blocks < 32) threads = 32;
		else threads = 262144 / size_array;
	}

	parallel_time = omp_get_wtime();
	ParallelSort << <blocks, threads >> > (temp_data_parallel, size_array);
	parallel_time = omp_get_wtime() - parallel_time;
	cudaDeviceSynchronize();

	cudaMemcpy(data_parallel, temp_data_parallel, size_array * sizeof(int), cudaMemcpyDeviceToHost);

	single_time = omp_get_wtime();
	SingleSort(data_single, size_array);
	single_time = omp_get_wtime() - single_time;

	if (ComparisonResults(data_parallel, data_single, size_array))
		cout << "Is equal" << endl;
	else
		cout << "Is not equal" << endl;

	printf("Time of single sorting:   %f\n", single_time);
	printf("Time of parallel sorting: %f\n", parallel_time);

	cudaFree(temp_data_parallel);

	if (save == true) {
		SaveFileData(data_parallel, size_array);
		SaveFileData(data_single, size_array);
	}

	free(data_parallel);
	free(data_single);

	cudaDeviceReset();
}

int main() {
    bool menu = true, save;
    int key, size_array;
    while (menu) {
        cout << "1|  Manual test mode " << endl;
        cout << "2|  Automatic test mode" << endl;
        cout << "3|  Out" << endl;
        cout << "-> ";
        cin >> key;
        cout << "==================================================";
        switch (key) {
        case 1: {
            cout << endl << "Enter list size - ";
            cin >> size_array;
            cout << "Save result to file? (0 or 1) - ";
            cin >> save;
            StartTest(size_array, save);
            cout << "==================================================" << endl;
            break;
        }
        case 2: {
            for (size_array = 10; size_array < 9000001;) {
                cout << endl << "List size - " << size_array << endl;
                StartTest(size_array, false);
                cout << "==================================================";
                if (size_array < 1000000) size_array *= 10;
                else size_array += 1000000;
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