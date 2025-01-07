#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define BLOCK_SIZE 256
#define THRESHOLD 1024
#define SHARED_SIZE 2048

// Device helper functions
__device__ inline void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

__device__ inline int median3(int a, int b, int c) {
    return (a < b) ? 
           (b < c ? b : (a < c ? c : a)) :
           (a < c ? a : (b < c ? c : b));
}

// Insertion sort for small segments
__device__ void insertion_sort(int* arr, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int key = arr[i];
        if (key < arr[i - 1]) {
            int j = i - 1;
            do {
                arr[j + 1] = arr[j];
                j--;
            } while (j >= left && arr[j] > key);
            arr[j + 1] = key;
        }
    }
}

// Three-way partition kernel
__global__ void partition_kernel(int* arr, int N, int* lt_counts, int* gt_counts, 
                               int low_pivot, int high_pivot) {
    extern __shared__ int shared_mem[];
    int* shared_lt = shared_mem;
    int* shared_gt = &shared_mem[blockDim.x];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize local counters
    shared_lt[tid] = 0;
    shared_gt[tid] = 0;
    
    // Count elements in thread's portion
    while (gid < N) {
        int val = arr[gid];
        if (val < low_pivot) shared_lt[tid]++;
        else if (val > high_pivot) shared_gt[tid]++;
        gid += gridDim.x * blockDim.x;
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_lt[tid] += shared_lt[tid + s];
            shared_gt[tid] += shared_gt[tid + s];
        }
        __syncthreads();
    }
    
    // Write block results to global memory
    if (tid == 0) {
        lt_counts[blockIdx.x] = shared_lt[0];
        gt_counts[blockIdx.x] = shared_gt[0];
    }
}

// Rearrange elements kernel
__global__ void rearrange_kernel(int* arr, int N, int* pos_lt, int* pos_gt,
                               int low_pivot, int high_pivot) {
    __shared__ int shared_pos_lt;
    __shared__ int shared_pos_gt;
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared positions for block
    if (tid == 0) {
        shared_pos_lt = atomicAdd(pos_lt, blockDim.x);
        shared_pos_gt = atomicAdd(pos_gt, -blockDim.x);
    }
    __syncthreads();
    
    while (gid < N) {
        int val = arr[gid];
        if (val < low_pivot) {
            int pos = atomicAdd(&shared_pos_lt, 1);
            if (pos < N) arr[pos] = val;
        } else if (val > high_pivot) {
            int pos = atomicAdd(&shared_pos_gt, -1);
            if (pos >= 0) arr[pos] = val;
        }
        gid += gridDim.x * blockDim.x;
    }
}

// Host function to perform one level of partitioning
void partition_step(int* d_arr, int N, int& lt_end, int& gt_start) {
    const int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate memory for counters
    int *d_lt_counts, *d_gt_counts;
    cudaMalloc(&d_lt_counts, num_blocks * sizeof(int));
    cudaMalloc(&d_gt_counts, num_blocks * sizeof(int));
    
    // Select pivots (using first, middle, last elements for simplicity)
    int h_samples[3];
    cudaMemcpy(h_samples, d_arr, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_samples[1], d_arr + N/2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_samples[2], d_arr + N-1, sizeof(int), cudaMemcpyDeviceToHost);
    
    int low_pivot = median3(h_samples[0], h_samples[1], h_samples[2]);
    int high_pivot = max(max(h_samples[0], h_samples[1]), h_samples[2]);
    
    // Count elements
    partition_kernel<<<num_blocks, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(int)>>>
        (d_arr, N, d_lt_counts, d_gt_counts, low_pivot, high_pivot);
    
    // Calculate positions
    int *d_pos_lt, *d_pos_gt;
    cudaMalloc(&d_pos_lt, sizeof(int));
    cudaMalloc(&d_pos_gt, sizeof(int));
    
    cudaMemset(d_pos_lt, 0, sizeof(int));
    cudaMemset(d_pos_gt, N-1, sizeof(int));
    
    // Rearrange elements
    rearrange_kernel<<<num_blocks, BLOCK_SIZE>>>
        (d_arr, N, d_pos_lt, d_pos_gt, low_pivot, high_pivot);
    
    // Get partition boundaries
    cudaMemcpy(&lt_end, d_pos_lt, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gt_start, d_pos_gt, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_lt_counts);
    cudaFree(d_gt_counts);
    cudaFree(d_pos_lt);
    cudaFree(d_pos_gt);
}

// Main sorting function
void cuda_quicksort(int* h_arr, int N) {
    // Allocate device memory
    int* d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Create work queue for partitions to be sorted
    struct Partition {
        int start, end;
        Partition(int s = 0, int e = 0) : start(s), end(e) {}
    };
    std::vector<Partition> work_queue;
    work_queue.push_back(Partition(0, N-1));
    
    // Process work queue
    while (!work_queue.empty()) {
        Partition current = work_queue.back();
        work_queue.pop_back();
        
        int len = current.end - current.start + 1;
        
        if (len <= THRESHOLD) {
            // Use thrust::sort for small partitions
            thrust::device_ptr<int> thrust_ptr(d_arr + current.start);
            thrust::sort(thrust_ptr, thrust_ptr + len);
            continue;
        }
        
        int lt_end, gt_start;
        partition_step(d_arr + current.start, len, lt_end, gt_start);
        
        // Add new partitions to work queue
        if (lt_end > 0) {
            work_queue.push_back(Partition(current.start, current.start + lt_end - 1));
        }
        if (gt_start < len-1) {
            work_queue.push_back(Partition(current.start + gt_start + 1, current.end));
        }
    }
    
    // Copy result back to host
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_arr);
}

int main() {
    // Read input
    FILE* input_file = fopen("input.txt", "r");
    if (!input_file) {
        fprintf(stderr, "Error opening input file\n");
        return 1;
    }
    
    int N;
    fscanf(input_file, "%d", &N);
    
    int* h_arr = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        fscanf(input_file, "%d", &h_arr[i]);
    }
    fclose(input_file);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start time
    cudaEventRecord(start);
    
    // Sort array
    cuda_quicksort(h_arr, N);
    
    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Sorting time: %f milliseconds\n", milliseconds);
    
    // Cleanup
    free(h_arr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}



nvcc -O3 quicksort.cu -o quicksort