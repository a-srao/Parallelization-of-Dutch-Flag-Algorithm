#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define THRESHOLD 128  // Increased threshold for insertion sort
#define SAMPLE_RATE 100  // Sample rate for median-of-medians pivot selection

// Optimized insertion sort with fewer comparisons and memory accesses
void insertion_sort(int *arr, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int key = arr[i];
        if (key < arr[i - 1]) {  // Only enter the loop if necessary
            int j = i - 1;
            do {
                arr[j + 1] = arr[j];
                j--;
            } while (j >= left && arr[j] > key);
            arr[j + 1] = key;
        }
    }
}

// Median of three values
static inline int median3(int a, int b, int c) {
    return (a < b) ? 
           (b < c ? b : (a < c ? c : a)) :
           (a < c ? a : (b < c ? c : b));
}

// Optimized pivot selection using median-of-medians approach
void select_pivots(int *arr, int low, int high, int *low_pivot, int *high_pivot) {
    if (high - low < 3) {
        *low_pivot = arr[low];
        *high_pivot = arr[high];
        return;
    }

    int len = high - low + 1;
    int sample_size = len / SAMPLE_RATE + 1;
    if (sample_size > 9) sample_size = 9;  // Limit sample size
    
    int samples[9];
    for (int i = 0; i < sample_size; i++) {
        int idx = low + (i * len / sample_size);
        samples[i] = arr[idx];
    }
    
    // Sort samples
    for (int i = 0; i < sample_size - 1; i++) {
        for (int j = 0; j < sample_size - i - 1; j++) {
            if (samples[j] > samples[j + 1]) {
                int temp = samples[j];
                samples[j] = samples[j + 1];
                samples[j + 1] = temp;
            }
        }
    }

    *low_pivot = samples[sample_size / 3];
    *high_pivot = samples[(2 * sample_size) / 3];
}

// Optimized 3-way partition with fewer swaps
void partition_3way(int *arr, int low, int high, int *low_pivot_idx, int *high_pivot_idx) {
    // Select better pivots
    int low_pivot, high_pivot;
    select_pivots(arr, low, high, &low_pivot, &high_pivot);
    
    // Initialize pointers
    int lt = low;      // Less than low pivot
    int i = low + 1;   // Current element
    int gt = high;     // Greater than high pivot
    
    // Place pivots at the ends
    arr[low] = low_pivot;
    arr[high] = high_pivot;
    
    // Three-way partitioning with fewer comparisons
    while (i <= gt) {
        if (arr[i] < low_pivot) {
            int temp = arr[i];
            arr[i++] = arr[++lt];
            arr[lt] = temp;
        }
        else if (arr[i] > high_pivot) {
            int temp = arr[i];
            arr[i] = arr[gt];
            arr[gt--] = temp;
        }
        else i++;
    }
    
    // Move pivots to final positions
    arr[low] = arr[lt];
    arr[lt] = low_pivot;
    arr[high] = arr[gt + 1];
    arr[gt + 1] = high_pivot;
    
    *low_pivot_idx = lt;
    *high_pivot_idx = gt + 1;
}

// Optimized quicksort with insertion sort for small arrays
void quicksort_3way(int *arr, int low, int high) {
    while (low < high) {
        if (high - low < THRESHOLD) {
            insertion_sort(arr, low, high);
            return;
        }
        
        int low_pivot_idx, high_pivot_idx;
        partition_3way(arr, low, high, &low_pivot_idx, &high_pivot_idx);
        
        // Tail recursion elimination for the largest partition
        if (low_pivot_idx - low <= high - high_pivot_idx) {
            quicksort_3way(arr, low, low_pivot_idx - 1);
            low = high_pivot_idx + 1;
        } else {
            quicksort_3way(arr, high_pivot_idx + 1, high);
            high = low_pivot_idx - 1;
        }
    }
}

// Optimized min-heap implementation
typedef struct {
    int value;
    int array_idx;
} HeapNode;

// Cache-friendly heap operations
static inline void sift_down(HeapNode *heap, int start, int end) {
    int root = start;
    while (root * 2 + 1 <= end) {
        int child = root * 2 + 1;
        if (child + 1 <= end && heap[child + 1].value < heap[child].value)
            child++;
        if (heap[root].value > heap[child].value) {
            HeapNode temp = heap[root];
            heap[root] = heap[child];
            heap[child] = temp;
            root = child;
        } else
            return;
    }
}
// Optimized k-way merge using buffer for better cache utilization
void k_way_merge(int *arr, int N, int num_procs, int *sendcounts, int *displs) {
    const int BUFFER_SIZE = 1024;
    int *buffer = (int *)malloc(BUFFER_SIZE * sizeof(int));
    int buffer_pos = 0;
    
    // Fixed: Correct allocation for HeapNode array
    HeapNode *heap = (HeapNode *)malloc(num_procs * sizeof(HeapNode));
    int *indices = (int *)calloc(num_procs, sizeof(int));
    int *output = (int *)malloc(N * sizeof(int));
    
    // Initialize heap
    for (int i = 0; i < num_procs; i++) {
        heap[i].value = (sendcounts[i] > 0) ? arr[displs[i]] : INT_MAX;
        heap[i].array_idx = i;
        indices[i] = displs[i] + 1;
    }
    
    // Heapify
    for (int i = (num_procs - 2) / 2; i >= 0; i--)
        sift_down(heap, i, num_procs - 1);
    
    // Merge with buffering
    for (int i = 0; i < N; i++) {
        HeapNode min = heap[0];
        buffer[buffer_pos++] = min.value;
        
        // Flush buffer if full
        if (buffer_pos == BUFFER_SIZE) {
            memcpy(output + i - BUFFER_SIZE + 1, buffer, BUFFER_SIZE * sizeof(int));
            buffer_pos = 0;
        }
        
        // Update heap
        if (indices[min.array_idx] < displs[min.array_idx] + sendcounts[min.array_idx]) {
            heap[0].value = arr[indices[min.array_idx]++];
        } else {
            heap[0].value = INT_MAX;
        }
        sift_down(heap, 0, num_procs - 1);
    }
    
    // Flush remaining buffer
    if (buffer_pos > 0) {
        memcpy(output + N - buffer_pos, buffer, buffer_pos * sizeof(int));
    }
    
    // Copy back to original array
    memcpy(arr, output, N * sizeof(int));
    
    free(buffer);
    free(heap);
    free(indices);
    free(output);
}
/*
int main(int argc, char *argv[]) {
    int rank, size, N;
    int *arr = NULL;
    double start_time, end_time;
    int *sendcounts = NULL;
    int *displs = NULL;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Root process reads input
    if (rank == 0) {
        FILE *input_file = fopen("input.txt", "r");
        if (!input_file) {
            fprintf(stderr, "Error opening input file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        fscanf(input_file, "%d", &N);
        arr = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) {
            fscanf(input_file, "%d", &arr[i]);
        }
        fclose(input_file);
        
        // Calculate distribution
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        
        int base = N / size;
        int remainder = N % size;
        
        displs[0] = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = base + (i < remainder);
            if (i > 0) displs[i] = displs[i-1] + sendcounts[i-1];
        }
    }
    
    // Broadcast array size
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Get local array size
    int local_N;
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Allocate and distribute local arrays
    int *local_arr = (int *)malloc(local_N * sizeof(int));
    MPI_Scatterv(arr, sendcounts, displs, MPI_INT, local_arr, local_N, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Sort local arrays
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    quicksort_3way(local_arr, 0, local_N - 1);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    // Gather sorted arrays
    MPI_Gatherv(local_arr, local_N, MPI_INT, arr, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Merge on root
    if (rank == 0) {
        k_way_merge(arr, N, size, sendcounts, displs);
        printf("Sorting time: %f seconds\n", (end_time - start_time)*1000);
    }
    
    // Cleanup
    free(local_arr);
    if (rank == 0) {
        free(arr);
        free(sendcounts);
        free(displs);
    }
    
    MPI_Finalize();
    return 0;
}*/
int main(int argc, char *argv[]) {
    int rank, size, N;
    int *arr = NULL;
    double start_time, end_time;
    int *sendcounts = NULL;
    int *displs = NULL;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Root process reads input
    if (rank == 0) {
        FILE *input_file = fopen("input.txt", "r");
        if (!input_file) {
            fprintf(stderr, "Error opening input file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        fscanf(input_file, "%d", &N);
        arr = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) {
            fscanf(input_file, "%d", &arr[i]);
        }
        fclose(input_file);
        
        // Calculate distribution
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        
        int base = N / size;
        int remainder = N % size;
        
        displs[0] = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = base + (i < remainder);
            if (i > 0) displs[i] = displs[i-1] + sendcounts[i-1];
        }
    }
    
    // Broadcast array size
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Get local array size
    int local_N;
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Allocate and distribute local arrays
    int *local_arr = (int *)malloc(local_N * sizeof(int));
    MPI_Scatterv(arr, sendcounts, displs, MPI_INT, local_arr, local_N, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Sort local arrays
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    quicksort_3way(local_arr, 0, local_N - 1);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    // Gather sorted arrays
    MPI_Gatherv(local_arr, local_N, MPI_INT, arr, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Merge on root
    if (rank == 0) {
        k_way_merge(arr, N, size, sendcounts, displs);
        
        // Write the sorted array to an output file in the root process
        FILE *output_file = fopen("output.txt", "w");
        if (!output_file) {
            fprintf(stderr, "Error opening output file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Write the number of elements
        fprintf(output_file, "%d\n", N);

        // Write the sorted array
        for (int i = 0; i < N; i++) {
            fprintf(output_file, "%d ", arr[i]);
        }

        fprintf(output_file, "\n");
        fclose(output_file);
        printf("Sorted data has been written to output.txt\n");
        
        printf("Sorting time: %f seconds\n", (end_time - start_time) * 1000);
    }
    
    // Cleanup
    free(local_arr);
    if (rank == 0) {
        free(arr);
        free(sendcounts);
        free(displs);
    }
    
    MPI_Finalize();
    return 0;
}

