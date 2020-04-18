#include <iostream>
#include <math.h>
#include <sys/time.h>
#include "cudaDmy.cuh"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <map>
#include <set> 
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <bits/stdc++.h> 
#include <assert.h>
#include <limits>
#include <string>
#include <sstream>

using std::ifstream;
using std::cout;
using std::endl;
using std::map;
using std::set;
using std::pair;
using std::make_pair;
using std::string;
using std::vector;
using std::advance;
using std::sort;
using std::accumulate;
using std::max_element;

struct pairCompareSecondDescending {
    bool operator() (const pair<int,int> &lhs, const pair<int,int> &rhs) const{
         return (lhs.second > rhs.second);
    }
};

struct timeval t1, t2;

__global__ void m_find_coauthors(const unsigned int *arr, int m, int n, unsigned int* coAuthorCount)
{

  __shared__ unsigned int smem[2];
  unsigned int *bSum = &smem[0];
  unsigned int *findTarget = &smem[1];
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = m;

  // thread 0 in block sets smem bSum to 0
  if(threadIdx.x == 0)
  {
    *bSum = 0;
  }

  __syncthreads();

  for(int i = 0; i<m ; i++)
  {
    for(int j = 0; j<n; j++)
    {
      // thread 0 in block sets findTarget for all block
      // find target will be the same for all blocks
      if(threadIdx.x == 0)
      {
        *findTarget = arr[j*m+i];
        // printf("THREAD %d set findTarget to %d!\n", index, *findTarget);
      }

      // sync due to divergence
      __syncthreads();

      // since num threads are guaranteed to be >= m
      // the thread that has an index equal to current 
      // author being explored is dropped 

        // all threads traverse down in order to 
        // look if target is in their "lane"
        // if target is present block bSum is incremented
        // sync called due to possible divergence
      for (int thrd = index; thrd < m*n; thrd += stride)
      {
          if(arr[thrd] == *findTarget && index != i)
          {
            atomicAdd(bSum, 1);
          }
          __syncthreads();
      }
    }
  }

  // aggregate bSums to coAuthorCount
  if(threadIdx.x == 0)
  {
    atomicAdd(coAuthorCount, *bSum);
  }

}

__global__ void v_bucket(const unsigned int *arr, unsigned int *gbucket, int arrSize, int sbucketSize)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Dynamically allocated shared memory array 
  extern __shared__ unsigned int sbuckets[];

  // This loop forces blocks to prep their own shared memory
  // Shared memory is not shared accross blocks, it's shared across
  // threads. As a result indexing is controlled by threads
  // Interleaved access used here for thread coalescing. Stride is 
  // number of threads in a block because only 1 block accesses the 
  // shared memory so it's equivelent to blockDim.x * gridDim.x 
  // where gridDim.x is 1. 
  for (int i = threadIdx.x; i < sbucketSize; i += blockDim.x)
  {
    sbuckets[i] = 0;
  }

  // synchronize idle threads that skipped loop with threads that
  // went through loop. 
  __syncthreads();


  int stride = blockDim.x * gridDim.x;

  //if index exceeds size of sbucket (block with extra threads)
  //condition for loop will fail and those threads will be idle
  //stride prevents interleaved threads from one block overlapping
  //with threads from another block. Kernel is generally always 
  //launched with blocks and threads that exceed the size of the 
  //array being bucketed but if less the += stride fixes it. 
  for(int i = index; i<arrSize; i+=stride)
  {
    atomicAdd(&(sbuckets[arr[i]]), 1);
  }

  // synchronize idle threads that skipped loop with threads that
  // went through loop. 
  __syncthreads();

  // each block goes through own shared memory, and aggregates
  // value in index i of own shared memory into index i of gbucket
  // gbucket and shared memory bucket sizes are identical. Access
  // of shared memory is by block so indexing of loop is similair 
  // to setting shared memory to zero. Middle loop access involved 
  // more blocks accessing input array so indexing was different.
  // If blockDim.x*threadIdx.x > sbucketSize then some threads will
  // be idle. Hence, there should be a need dfor __syncthreads but
  // since this is the last operation in the kernel we don't need 
  // to syncthreads. Atomic add still used because multiple threads in blocks 
  // will access the same index in gbucket. This could potentially be
  // avoided if each block accessed seperate parts of their own shared
  // memory at different offsets from each other (thus eliminating the
  // need for atomicAdd because no two blocks would access the same indexes)
  // however, that's too hard and the current implementation is fast enough

  for(int i = threadIdx.x; i < sbucketSize; i += blockDim.x)
  {
    atomicAdd(&(gbucket[i]), sbuckets[i]);
  }

}

// Kernel function to add the elements of two arrays
__global__ void v_accumulate(unsigned int *arr, unsigned int *acc, int m, int n)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = m;
  if(index < m)
  {
    for (int i = index; i < m*n; i += stride)
    {
        acc[index] += (arr[i] >= 1)? 1 : 0;
    }
  }
}

// Kernel function to add the elements of two arrays
__global__ void v_set(unsigned int *arr, float val, int m)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < m)
    arr[index] = val;
}

// Kernel function to add the elements of two arrays
__global__ void v_set(int *arr, float val, int m)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < m)
    arr[index] = val;
}

// Kernel function to add the elements of two arrays
__global__ void v_set(float *arr, float val, int m)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < m)
    arr[index] = val;
}

map<unsigned int, unsigned int> coAuthorCountHistogram(unsigned int *coAuthorCounts, int m, int n, unsigned int maxCoAuthorCount)
{
  unsigned int threadCount = 1024;
  unsigned int blockCount = (maxCoAuthorCount+(threadCount-1))/threadCount;

  unsigned int* gbuckets = NULL; 
  cudaMallocManaged(&gbuckets, (maxCoAuthorCount)*sizeof(unsigned int));

  v_set KERNEL_ARG2(blockCount,threadCount)(gbuckets, 0, maxCoAuthorCount);
  cudaDeviceSynchronize();

  threadCount = 1024;
  blockCount = (m+(threadCount-1))/threadCount;

  v_bucket KERNEL_ARG3(blockCount,threadCount, maxCoAuthorCount*sizeof(unsigned int)) (coAuthorCounts, gbuckets, m, maxCoAuthorCount);
  cudaDeviceSynchronize();

  map<unsigned int, unsigned int> histogram;

  for(int i = 0; i<maxCoAuthorCount; i++)
  {
    if(gbuckets[i] != 0)
    {
      histogram[i] = gbuckets[i];
    }
  }
  
  cudaFree(gbuckets);

  return histogram;
}

map<int, set< int>> loadcoAuthors()
{
  ifstream infile("coauthors.txt");
  map<int, set<int>> graph;
  vector<int> node1arr, node2arr;
  int node1, node2; 
  while(infile >> node1 >> node2)
  {
      node1arr.push_back(node1);
      node2arr.push_back(node2);
  }
  for(int i = 0; i < node1arr.size(); i++)
  {
      graph[node1arr[i]].insert(node2arr[i]);
      graph[node2arr[i]].insert(node1arr[i]);
  }
  return graph;
}

unsigned int* flattenGraph(map<int, set<int>> graph, int& m, int& n)
{
  m = graph.size()+1;
  unsigned int* arr;

  n = 0;
  for(auto node : graph)
  {
    if(node.second.size() > n)
    {
      n = node.second.size();
    }
  }

  cudaMallocManaged(&arr, ((m)*n)*sizeof(unsigned int));

  unsigned int threadCount = 1024;
  unsigned int blockCount = ((m*n)+(threadCount-1))/threadCount;
  v_set KERNEL_ARG2(blockCount,threadCount)(arr, 0, m*n);
  cudaDeviceSynchronize();

  for(int i = 0; i<m; i++)
  {
    if(graph.find(i) != graph.end())
    {
      vector<unsigned int> coAuthors(graph[i].begin(), graph[i].end());
      for(int j = 0; j<coAuthors.size(); j++)
      {
        arr[j*m+i] = coAuthors[j];
      }
    }
  }
  return arr;
}

set<pair<int,int>, pairCompareSecondDescending> rankAuthorByCoAuthorCount(unsigned int* arr, unsigned int **coAuthorSums, int m, int n)
{
  set<pair<int,int>, pairCompareSecondDescending> rankings;
  cudaMallocManaged(coAuthorSums, (m)*sizeof(unsigned int));

  unsigned int threadCount = 1024;
  unsigned int blockCount = (m+(threadCount-1))/threadCount;

  v_set KERNEL_ARG2(blockCount,threadCount)(*coAuthorSums, 0, m);
  cudaDeviceSynchronize();

  v_accumulate KERNEL_ARG2(blockCount,threadCount)(arr, *coAuthorSums, m, n);
  cudaDeviceSynchronize();

  for(int i = 0; i<m; i++)
  {
    rankings.insert(make_pair(i, (*coAuthorSums)[i]));
  }

  return rankings;
}

void printRankings(set<pair<int,int>, pairCompareSecondDescending> rankings)
{
  for(auto it = rankings.begin(); it != rankings.end(); it++)
  {
    printf("Author[%d] coauthorCount = %d\n", it->first, it->second);
  }
}

unsigned int getAuthorThatShareCoAuthorCount(const unsigned int *arr, int m, int n)
{
  unsigned int* count ;
  cudaMallocManaged(&count, sizeof(unsigned int));
  
  unsigned int threadCount = 1024;
  unsigned int blockCount = (m+(threadCount-1))/threadCount;
  // unsigned int blockCount = 512;
  
  printf("Thread Count per block %d\n", threadCount);
  printf("blockCount %d\n", blockCount);
  printf("Total Threads %d\n", blockCount*threadCount);

  m_find_coauthors KERNEL_ARG2(blockCount, threadCount)(arr, m, n, count);
  cudaDeviceSynchronize();

  unsigned int returnable = *count;

  cudaFree(count);

  return returnable;
}

std::fstream& GotoLine(std::fstream& file, unsigned int num){
    file.seekg(std::ios::beg);
    for(int i=0; i < num - 1; ++i){
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }
    return file;
}

void printDesignPoint(float* designSpaceTensor, unsigned int dpIndex)
{
  unsigned int peCount = 37;
  unsigned int funcCount = 38;
  for(int funcTypeIdx = 0; funcTypeIdx<funcCount; funcTypeIdx++)
  {
    for(int peIndex = 0; peIndex<peCount; peIndex++)
    {
      printf("%d ", (int)(designSpaceTensor[dpIndex*(peCount*funcCount) + funcTypeIdx*peCount + peIndex]));
    }
    printf("\n");
  }
}

float* loadDSTensor(unsigned int& designPointsCount, unsigned int& peCount, unsigned int& funcCount)
{
  unsigned int fileInfoArray[1047][3] = {{0, 0, 0},{0, 1, 0},{0, 2, 0},{0, 3, 0},{0, 4, 0},{0, 5, 0},{0, 6, 0},{0, 7, 0},{0, 8, 0},{0, 9, 0},{0, 10, 0},{0, 11, 0},{0, 12, 0},{0, 13, 0},{0, 14, 0},{0, 15, 0},{0, 16, 0},{0, 17, 0},{0, 18, 0},{0, 19, 0},{0, 20, 0},{0, 21, 0},{0, 22, 0},{0, 23, 0},{0, 24, 0},{0, 25, 0},{0, 26, 0},{0, 27, 0},{0, 28, 0},{0, 29, 0},{0, 30, 0},{0, 31, 0},{0, 32, 0},{0, 33, 0},{0, 34, 0},{0, 35, 0},{0, 36, 0},{0, 37, 0},{0, 38, 0},{0, 39, 0},{1, 0, 0},{1, 0, 1},{1, 1, 0},{1, 2, 0},{1, 2, 1},{1, 3, 0},{1, 4, 0},{1, 4, 1},{1, 5, 0},{1, 5, 1},{1, 5, 2},{1, 5, 3},{1, 6, 0},{1, 6, 1},{1, 7, 0},{1, 8, 0},{1, 9, 0},{1, 9, 1},{1, 10, 0},{1, 10, 1},{1, 11, 0},{1, 12, 0},{1, 13, 0},{1, 13, 1},{1, 14, 0},{1, 14, 1},{1, 14, 2},{1, 14, 3},{1, 15, 0},{1, 16, 0},{1, 17, 0},{1, 18, 0},{1, 19, 0},{1, 19, 1},{1, 20, 0},{1, 21, 0},{1, 22, 0},{1, 23, 0},{1, 24, 0},{1, 25, 0},{1, 25, 1},{1, 26, 0},{1, 27, 0},{1, 27, 1},{1, 28, 0},{1, 28, 1},{1, 29, 0},{1, 29, 1},{1, 29, 2},{1, 29, 3},{1, 30, 0},{1, 31, 0},{1, 32, 0},{1, 32, 1},{1, 32, 2},{1, 32, 3},{1, 32, 4},{1, 32, 5},{1, 32, 6},{1, 32, 7},{1, 33, 0},{1, 34, 0},{1, 35, 0},{1, 35, 1},{1, 35, 2},{1, 35, 3},{1, 36, 0},{1, 37, 0},{1, 37, 1},{1, 38, 0},{1, 38, 1},{1, 39, 0},{1, 39, 1},{2, 0, 0},{2, 1, 0},{2, 1, 1},{2, 1, 2},{2, 1, 3},{2, 1, 4},{2, 1, 5},{2, 1, 6},{2, 1, 7},{2, 1, 8},{2, 1, 9},{2, 1, 10},{2, 1, 11},{2, 1, 12},{2, 1, 13},{2, 1, 14},{2, 1, 15},{2, 2, 0},{2, 3, 0},{2, 3, 1},{2, 3, 2},{2, 3, 3},{2, 4, 0},{2, 4, 1},{2, 4, 2},{2, 4, 3},{2, 4, 4},{2, 4, 5},{2, 4, 6},{2, 4, 7},{2, 5, 0},{2, 5, 1},{2, 6, 0},{2, 6, 1},{2, 6, 2},{2, 6, 3},{2, 6, 4},{2, 6, 5},{2, 6, 6},{2, 6, 7},{2, 7, 0},{2, 7, 1},{2, 7, 2},{2, 7, 3},{2, 7, 4},{2, 7, 5},{2, 7, 6},{2, 7, 7},{2, 7, 8},{2, 7, 9},{2, 7, 10},{2, 7, 11},{2, 7, 12},{2, 7, 13},{2, 7, 14},{2, 7, 15},{2, 8, 0},{2, 9, 0},{2, 9, 1},{2, 10, 0},{2, 10, 1},{2, 10, 2},{2, 10, 3},{2, 10, 4},{2, 10, 5},{2, 10, 6},{2, 10, 7},{2, 11, 0},{2, 11, 1},{2, 11, 2},{2, 11, 3},{2, 11, 4},{2, 11, 5},{2, 11, 6},{2, 11, 7},{2, 12, 0},{2, 12, 1},{2, 13, 0},{2, 13, 1},{2, 13, 2},{2, 13, 3},{2, 13, 4},{2, 13, 5},{2, 13, 6},{2, 13, 7},{2, 14, 0},{2, 14, 1},{2, 15, 0},{2, 16, 0},{2, 16, 1},{2, 16, 2},{2, 16, 3},{2, 16, 4},{2, 16, 5},{2, 16, 6},{2, 16, 7},{2, 16, 8},{2, 16, 9},{2, 16, 10},{2, 16, 11},{2, 16, 12},{2, 16, 13},{2, 16, 14},{2, 16, 15},{2, 16, 16},{2, 16, 17},{2, 16, 18},{2, 16, 19},{2, 16, 20},{2, 16, 21},{2, 16, 22},{2, 16, 23},{2, 16, 24},{2, 16, 25},{2, 16, 26},{2, 16, 27},{2, 16, 28},{2, 16, 29},{2, 16, 30},{2, 16, 31},{2, 16, 32},{2, 16, 33},{2, 16, 34},{2, 16, 35},{2, 16, 36},{2, 16, 37},{2, 16, 38},{2, 16, 39},{2, 16, 40},{2, 16, 41},{2, 16, 42},{2, 16, 43},{2, 16, 44},{2, 16, 45},{2, 16, 46},{2, 16, 47},{2, 16, 48},{2, 16, 49},{2, 16, 50},{2, 16, 51},{2, 16, 52},{2, 16, 53},{2, 16, 54},{2, 16, 55},{2, 16, 56},{2, 16, 57},{2, 16, 58},{2, 16, 59},{2, 16, 60},{2, 16, 61},{2, 16, 62},{2, 16, 63},{2, 17, 0},{2, 17, 1},{2, 18, 0},{2, 18, 1},{2, 19, 0},{2, 20, 0},{2, 20, 1},{2, 20, 2},{2, 20, 3},{2, 21, 0},{2, 21, 1},{2, 21, 2},{2, 21, 3},{2, 22, 0},{2, 22, 1},{2, 22, 2},{2, 22, 3},{2, 22, 4},{2, 22, 5},{2, 22, 6},{2, 22, 7},{2, 23, 0},{2, 23, 1},{2, 24, 0},{2, 24, 1},{2, 25, 0},{2, 25, 1},{2, 25, 2},{2, 25, 3},{2, 25, 4},{2, 25, 5},{2, 25, 6},{2, 25, 7},{2, 25, 8},{2, 25, 9},{2, 25, 10},{2, 25, 11},{2, 25, 12},{2, 25, 13},{2, 25, 14},{2, 25, 15},{2, 26, 0},{2, 27, 0},{2, 27, 1},{2, 28, 0},{2, 28, 1},{2, 28, 2},{2, 28, 3},{2, 28, 4},{2, 28, 5},{2, 28, 6},{2, 28, 7},{2, 29, 0},{2, 29, 1},{2, 29, 2},{2, 29, 3},{2, 30, 0},{2, 31, 0},{2, 32, 0},{2, 33, 0},{2, 33, 1},{2, 34, 0},{2, 35, 0},{2, 35, 1},{2, 35, 2},{2, 35, 3},{2, 36, 0},{2, 37, 0},{2, 38, 0},{2, 38, 1},{2, 38, 2},{2, 38, 3},{2, 38, 4},{2, 38, 5},{2, 38, 6},{2, 38, 7},{2, 39, 0},{2, 39, 1},{2, 39, 2},{2, 39, 3},{2, 39, 4},{2, 39, 5},{2, 39, 6},{2, 39, 7},{3, 0, 0},{3, 0, 1},{3, 1, 0},{3, 1, 1},{3, 1, 2},{3, 1, 3},{3, 1, 4},{3, 1, 5},{3, 1, 6},{3, 1, 7},{3, 1, 8},{3, 1, 9},{3, 1, 10},{3, 1, 11},{3, 1, 12},{3, 1, 13},{3, 1, 14},{3, 1, 15},{3, 2, 0},{3, 2, 1},{3, 3, 0},{3, 3, 1},{3, 3, 2},{3, 3, 3},{3, 4, 0},{3, 4, 1},{3, 4, 2},{3, 4, 3},{3, 4, 4},{3, 4, 5},{3, 4, 6},{3, 4, 7},{3, 4, 8},{3, 4, 9},{3, 4, 10},{3, 4, 11},{3, 4, 12},{3, 4, 13},{3, 4, 14},{3, 4, 15},{3, 5, 0},{3, 5, 1},{3, 5, 2},{3, 5, 3},{3, 6, 0},{3, 6, 1},{3, 6, 2},{3, 6, 3},{3, 6, 4},{3, 6, 5},{3, 6, 6},{3, 6, 7},{3, 6, 8},{3, 6, 9},{3, 6, 10},{3, 6, 11},{3, 6, 12},{3, 6, 13},{3, 6, 14},{3, 6, 15},{3, 7, 0},{3, 7, 1},{3, 7, 1},{3, 7, 2},{3, 7, 3},{3, 7, 4},{3, 7, 5},{3, 7, 6},{3, 7, 7},{3, 7, 8},{3, 7, 9},{3, 7, 10},{3, 7, 11},{3, 7, 13},{3, 7, 14},{3, 7, 15},{3, 7, 16},{3, 7, 17},{3, 7, 18},{3, 7, 19},{3, 7, 20},{3, 7, 21},{3, 7, 22},{3, 7, 23},{3, 7, 24},{3, 7, 25},{3, 7, 26},{3, 7, 27},{3, 7, 28},{3, 7, 29},{3, 7, 30},{3, 7, 31},{3, 8, 0},{3, 9, 0},{3, 9, 1},{3, 9, 2},{3, 9, 3},{3, 10, 0},{3, 10, 1},{3, 10, 2},{3, 10, 3},{3, 10, 4},{3, 10, 5},{3, 10, 6},{3, 10, 7},{3, 10, 8},{3, 10, 9},{3, 10, 10},{3, 10, 11},{3, 10, 12},{3, 10, 13},{3, 10, 14},{3, 10, 15},{3, 11, 0},{3, 11, 1},{3, 11, 2},{3, 11, 3},{3, 11, 4},{3, 11, 5},{3, 11, 6},{3, 11, 7},{3, 11, 8},{3, 11, 9},{3, 11, 10},{3, 11, 11},{3, 11, 12},{3, 11, 13},{3, 11, 14},{3, 11, 15},{3, 12, 0},{3, 12, 1},{3, 12, 2},{3, 12, 3},{3, 13, 0},{3, 13, 1},{3, 13, 2},{3, 13, 3},{3, 13, 4},{3, 13, 5},{3, 13, 6},{3, 13, 7},{3, 13, 8},{3, 13, 9},{3, 13, 10},{3, 13, 11},{3, 13, 12},{3, 13, 13},{3, 13, 14},{3, 13, 15},{3, 14, 0},{3, 14, 1},{3, 14, 2},{3, 14, 3},{3, 15, 0},{3, 16, 0},{3, 16, 1},{3, 16, 2},{3, 16, 3},{3, 16, 4},{3, 16, 5},{3, 16, 6},{3, 16, 7},{3, 16, 8},{3, 16, 9},{3, 16, 10},{3, 16, 11},{3, 16, 12},{3, 16, 13},{3, 16, 14},{3, 16, 15},{3, 16, 16},{3, 16, 17},{3, 16, 18},{3, 16, 19},{3, 16, 20},{3, 16, 21},{3, 16, 22},{3, 16, 23},{3, 16, 24},{3, 16, 25},{3, 16, 26},{3, 16, 27},{3, 16, 28},{3, 16, 29},{3, 16, 30},{3, 16, 31},{3, 16, 32},{3, 16, 33},{3, 16, 34},{3, 16, 35},{3, 16, 36},{3, 16, 37},{3, 16, 38},{3, 16, 39},{3, 16, 40},{3, 16, 41},{3, 16, 42},{3, 16, 43},{3, 16, 44},{3, 16, 45},{3, 16, 46},{3, 16, 47},{3, 16, 48},{3, 16, 49},{3, 16, 50},{3, 16, 51},{3, 16, 52},{3, 16, 53},{3, 16, 54},{3, 16, 55},{3, 16, 56},{3, 16, 57},{3, 16, 58},{3, 16, 59},{3, 16, 60},{3, 16, 61},{3, 16, 62},{3, 16, 63},{3, 16, 64},{3, 16, 65},{3, 16, 66},{3, 16, 67},{3, 16, 68},{3, 16, 69},{3, 16, 70},{3, 16, 71},{3, 16, 72},{3, 16, 73},{3, 16, 74},{3, 16, 75},{3, 16, 76},{3, 16, 77},{3, 16, 78},{3, 16, 79},{3, 16, 80},{3, 16, 81},{3, 16, 82},{3, 16, 83},{3, 16, 84},{3, 16, 85},{3, 16, 86},{3, 16, 87},{3, 16, 88},{3, 16, 89},{3, 16, 90},{3, 16, 91},{3, 16, 92},{3, 16, 93},{3, 16, 94},{3, 16, 95},{3, 16, 96},{3, 16, 97},{3, 16, 98},{3, 16, 99},{3, 17, 0},{3, 17, 1},{3, 17, 2},{3, 17, 3},{3, 18, 0},{3, 18, 1},{3, 18, 2},{3, 18, 3},{3, 19, 0},{3, 19, 1},{3, 20, 0},{3, 20, 1},{3, 20, 2},{3, 20, 3},{3, 20, 4},{3, 20, 5},{3, 20, 6},{3, 20, 7},{3, 21, 0},{3, 21, 1},{3, 21, 2},{3, 21, 3},{3, 21, 4},{3, 21, 5},{3, 21, 6},{3, 21, 7},{3, 22, 0},{3, 22, 1},{3, 22, 2},{3, 22, 3},{3, 22, 4},{3, 22, 5},{3, 22, 6},{3, 22, 7},{3, 23, 0},{3, 23, 1},{3, 24, 0},{3, 24, 1},{3, 24, 2},{3, 24, 3},{3, 25, 0},{3, 25, 1},{3, 25, 2},{3, 25, 3},{3, 25, 4},{3, 25, 5},{3, 25, 6},{3, 25, 7},{3, 25, 8},{3, 25, 9},{3, 25, 10},{3, 25, 11},{3, 25, 12},{3, 25, 13},{3, 25, 14},{3, 25, 15},{3, 25, 16},{3, 25, 17},{3, 25, 18},{3, 25, 19},{3, 25, 20},{3, 25, 21},{3, 25, 22},{3, 25, 23},{3, 25, 24},{3, 25, 25},{3, 25, 26},{3, 25, 27},{3, 25, 28},{3, 25, 29},{3, 25, 30},{3, 25, 31},{3, 26, 0},{3, 26, 1},{3, 26, 2},{3, 26, 3},{3, 27, 0},{3, 27, 1},{3, 27, 2},{3, 27, 3},{3, 28, 0},{3, 28, 1},{3, 28, 2},{3, 28, 3},{3, 28, 4},{3, 28, 5},{3, 28, 6},{3, 28, 7},{3, 29, 0},{3, 29, 1},{3, 29, 2},{3, 29, 3},{3, 29, 4},{3, 29, 5},{3, 29, 6},{3, 29, 7},{3, 30, 0},{3, 30, 1},{3, 31, 0},{3, 31, 1},{3, 32, 0},{3, 33, 0},{3, 33, 1},{3, 33, 2},{3, 33, 3},{3, 34, 0},{3, 35, 0},{3, 35, 1},{3, 35, 2},{3, 35, 3},{3, 35, 4},{3, 35, 5},{3, 35, 6},{3, 35, 7},{3, 36, 0},{3, 37, 0},{3, 37, 1},{3, 38, 0},{3, 38, 1},{3, 38, 2},{3, 38, 3},{3, 39, 0},{3, 39, 1},{3, 39, 2},{3, 39, 3},{3, 39, 4},{3, 39, 5},{3, 39, 6},{3, 39, 7},{4, 0, 0},{4, 0, 1},{4, 1, 0},{4, 1, 1},{4, 1, 2},{4, 1, 3},{4, 1, 4},{4, 1, 5},{4, 1, 6},{4, 1, 7},{4, 1, 8},{4, 1, 9},{4, 1, 10},{4, 1, 11},{4, 1, 12},{4, 1, 13},{4, 1, 14},{4, 1, 15},{4, 2, 0},{4, 2, 1},{4, 3, 0},{4, 3, 1},{4, 4, 0},{4, 4, 1},{4, 4, 2},{4, 4, 3},{4, 5, 0},{4, 5, 1},{4, 5, 2},{4, 5, 3},{4, 5, 4},{4, 5, 5},{4, 5, 6},{4, 5, 7},{4, 6, 0},{4, 6, 1},{4, 6, 2},{4, 6, 3},{4, 7, 0},{4, 7, 1},{4, 7, 2},{4, 7, 3},{4, 7, 4},{4, 7, 5},{4, 7, 6},{4, 7, 7},{4, 7, 8},{4, 7, 9},{4, 7, 10},{4, 7, 11},{4, 7, 12},{4, 7, 13},{4, 7, 14},{4, 7, 15},{4, 8, 0},{4, 9, 0},{4, 9, 1},{4, 9, 2},{4, 9, 3},{4, 10, 0},{4, 10, 1},{4, 10, 2},{4, 10, 3},{4, 11, 0},{4, 11, 1},{4, 11, 2},{4, 11, 3},{4, 11, 4},{4, 11, 5},{4, 11, 6},{4, 11, 7},{4, 12, 0},{4, 12, 1},{4, 13, 0},{4, 13, 1},{4, 13, 2},{4, 13, 3},{4, 14, 0},{4, 14, 1},{4, 14, 2},{4, 14, 3},{4, 14, 4},{4, 14, 5},{4, 14, 6},{4, 14, 7},{4, 15, 0},{4, 16, 0},{4, 16, 1},{4, 16, 2},{4, 16, 3},{4, 16, 4},{4, 16, 5},{4, 16, 6},{4, 16, 7},{4, 16, 8},{4, 16, 9},{4, 16, 10},{4, 16, 11},{4, 16, 12},{4, 16, 13},{4, 16, 14},{4, 16, 15},{4, 16, 16},{4, 16, 17},{4, 16, 18},{4, 16, 19},{4, 16, 20},{4, 16, 21},{4, 16, 22},{4, 16, 23},{4, 16, 24},{4, 16, 25},{4, 16, 26},{4, 16, 27},{4, 16, 28},{4, 16, 29},{4, 16, 30},{4, 16, 31},{4, 16, 32},{4, 16, 33},{4, 16, 34},{4, 16, 35},{4, 16, 36},{4, 16, 37},{4, 16, 38},{4, 16, 39},{4, 16, 40},{4, 16, 41},{4, 16, 42},{4, 16, 43},{4, 16, 44},{4, 16, 45},{4, 16, 46},{4, 16, 47},{4, 16, 48},{4, 16, 49},{4, 16, 50},{4, 16, 51},{4, 16, 52},{4, 16, 53},{4, 16, 54},{4, 16, 55},{4, 16, 56},{4, 16, 57},{4, 16, 58},{4, 16, 59},{4, 16, 60},{4, 16, 61},{4, 16, 62},{4, 16, 63},{4, 17, 0},{4, 17, 1},{4, 18, 0},{4, 18, 1},{4, 19, 0},{4, 19, 1},{4, 20, 0},{4, 20, 1},{4, 20, 2},{4, 20, 3},{4, 21, 0},{4, 21, 1},{4, 21, 2},{4, 21, 3},{4, 22, 0},{4, 22, 1},{4, 22, 2},{4, 22, 3},{4, 22, 4},{4, 22, 5},{4, 22, 6},{4, 22, 7},{4, 23, 0},{4, 23, 1},{4, 24, 0},{4, 24, 1},{4, 25, 0},{4, 25, 1},{4, 25, 2},{4, 25, 3},{4, 25, 4},{4, 25, 5},{4, 25, 6},{4, 25, 7},{4, 26, 0},{4, 27, 0},{4, 28, 0},{4, 28, 1},{4, 28, 2},{4, 28, 3},{4, 29, 0},{4, 29, 1},{4, 29, 2},{4, 29, 3},{4, 29, 4},{4, 29, 5},{4, 29, 6},{4, 29, 7},{4, 29, 8},{4, 29, 9},{4, 29, 10},{4, 29, 11},{4, 29, 12},{4, 29, 13},{4, 29, 14},{4, 29, 15},{4, 30, 0},{4, 31, 0},{4, 32, 0},{4, 32, 1},{4, 32, 2},{4, 32, 3},{4, 32, 4},{4, 32, 5},{4, 32, 6},{4, 32, 7},{4, 33, 0},{4, 33, 1},{4, 34, 0},{4, 35, 0},{4, 35, 1},{4, 35, 2},{4, 35, 3},{4, 35, 4},{4, 35, 5},{4, 35, 6},{4, 35, 7},{4, 35, 8},{4, 35, 9},{4, 35, 10},{4, 35, 11},{4, 35, 12},{4, 35, 13},{4, 35, 14},{4, 35, 15},{4, 36, 0},{4, 37, 0},{4, 37, 1},{4, 38, 0},{4, 38, 1},{4, 39, 0},{4, 39, 1},{5, 0, 0},{5, 0, 1},{5, 1, 0},{5, 2, 0},{5, 2, 1},{5, 3, 0},{5, 4, 0},{5, 5, 0},{5, 5, 1},{5, 5, 2},{5, 5, 3},{5, 5, 4},{5, 5, 5},{5, 5, 6},{5, 5, 7},{5, 6, 0},{5, 7, 0},{5, 8, 0},{5, 9, 0},{5, 9, 1},{5, 10, 0},{5, 11, 0},{5, 12, 0},{5, 13, 0},{5, 14, 0},{5, 14, 1},{5, 14, 2},{5, 14, 3},{5, 15, 0},{5, 16, 0},{5, 17, 0},{5, 18, 0},{5, 19, 0},{5, 19, 1},{5, 19, 2},{5, 19, 3},{5, 20, 0},{5, 21, 0},{5, 22, 0},{5, 23, 0},{5, 23, 1},{5, 24, 0},{5, 25, 0},{5, 26, 0},{5, 27, 0},{5, 28, 0},{5, 29, 0},{5, 29, 1},{5, 29, 2},{5, 29, 3},{5, 30, 0},{5, 31, 0},{5, 32, 0},{5, 32, 1},{5, 32, 2},{5, 32, 3},{5, 32, 4},{5, 32, 5},{5, 32, 6},{5, 32, 7},{5, 33, 0},{5, 34, 0},{5, 35, 0},{5, 35, 1},{5, 35, 2},{5, 35, 3},{5, 36, 0},{5, 37, 0},{5, 37, 1},{5, 37, 2},{5, 37, 3},{5, 38, 0},{5, 39, 0}};

  vector<vector<vector<unsigned int>>> designPoints;

  for(int i = 0; i<1047; i++)
  {
    string designPointFileString = "data/arch";
    designPointFileString += std::to_string(fileInfoArray[i][0]);
    designPointFileString += "_app";
    designPointFileString += std::to_string(fileInfoArray[i][1]);
    designPointFileString += "_bind";
    designPointFileString += std::to_string(fileInfoArray[i][2]);
    designPointFileString += ".txt";
    std::fstream designPointFile(designPointFileString);
    GotoLine(designPointFile, 6);
    string line;
    vector<vector<unsigned int>> designPoint;
    while(std::getline(designPointFile, line))
    {
      std::istringstream ss(line);
      unsigned int inputSize;
      vector<unsigned int> functionInputs;
      while(ss >> inputSize)
      {
        functionInputs.push_back(inputSize);
      }
      designPoint.push_back(functionInputs);
    }
    designPoints.push_back(designPoint);
  }

  vector<vector<unsigned int>> archVector;
  vector<unsigned int> allocatedPEs;
  for(int i = 0; i<6; i++)
  {
    string archFileString = "data/arch";
    archFileString += std::to_string(i);
    archFileString += ".txt";
    std::fstream archFile(archFileString);
    
    GotoLine(archFile, 6);
    string line;
    std::getline(archFile, line);
    unsigned int allocatedPe;
    std::istringstream ss(line);
    while(ss >> allocatedPe)
    {
      allocatedPEs.push_back(allocatedPe);
    }
    archVector.push_back(allocatedPEs);
    allocatedPEs.clear();
  }

  designPointsCount = 1047;
  peCount = 37;
  funcCount = 38;

  float *designSpaceTensor;
  unsigned int designSpaceSize = designPointsCount*peCount*funcCount;

  cudaMallocManaged(&designSpaceTensor, designSpaceSize*sizeof(float));
  
  unsigned int threadCount = 1024;
  unsigned int blockCount = (designSpaceSize+(threadCount-1))/threadCount;

  v_set KERNEL_ARG2(blockCount,threadCount)(designSpaceTensor, 0, designSpaceSize);
  cudaDeviceSynchronize();

  for(auto dpIndex = 0; dpIndex < designPoints.size(); dpIndex++)
  {
    for(auto funcTypeIndex = 0; funcTypeIndex < designPoints[dpIndex].size(); funcTypeIndex++)
    {
      for(auto peIndex = 0; peIndex < designPoints[dpIndex][funcTypeIndex].size(); peIndex++)
      {
        unsigned int peInputSize = designPoints[dpIndex][funcTypeIndex][peIndex];
        unsigned int tensorPeIndex = archVector[fileInfoArray[dpIndex][0]][peIndex];
        designSpaceTensor[dpIndex*(peCount*funcCount) + funcTypeIndex*peCount + tensorPeIndex] = peInputSize;
      } 
    }
  }

  return designSpaceTensor;
}

float* loadPerfTable(unsigned int& coefficientCount, unsigned int peCount, unsigned int funcCount)
{
  float c0[1406] = {0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 25, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 25, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 45, 45, 50, 50, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 835, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 75, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 45, 45, 50, 50, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 50, 50, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 25, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 105, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 25, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 45, 45, 50, 50, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12.5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 155, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12.5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 140, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 31.5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 345, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 25, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 225, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 200, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 105, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 50, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 25, 25, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 105, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 570, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 50, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 45, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 120, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 50, -1, -1, -1, -1, -1, -1, 45 };
  float c1[1406] = {0.5, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, 20, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0.5, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0.5, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 40, -1, -1, -1, -1, -1, -1, 50, 50, 55, 55, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 314, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 840, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 20, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 20, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 39, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 80, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 40, -1, -1, -1, -1, -1, -1, 50, 50, 55, 55, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 112, -1, -1, -1, -1, -1, -1, -1, -1, 55, 55, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 22, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 110, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 22, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 40, -1, -1, -1, -1, -1, -1, 50, 50, 55, 55, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 28, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13.75, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 570, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 160, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3.5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 20, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 28, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13.75, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 16.5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 145, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 35, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 355, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 26, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0.5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 671.5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 230, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 20.5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 205, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5.5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 106.25, -1, -1, -1, -1, -1, -1, -1, -1, 224, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 55, -1, -1, -1, -1, -1, -1, -1, 0.5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 30, 30, -1, -1, -1, -1, 31, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 110, -1, -1, -1, -1, 57.5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 575, -1, -1, -1, 112, -1, -1, -1, -1, -1, -1, -1, -1, -1, 55, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 50, -1, -1, 31.5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 125, -1, 80, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 55, -1, -1, -1, -1, -1, -1, 50};

  float* perfTable;
  coefficientCount = 2;
  unsigned int perfTableSize = peCount*funcCount;
  cudaMallocManaged(&perfTable, perfTableSize*sizeof(float));

  unsigned int threadCount = 1024;
  unsigned int blockCount = (perfTableSize+(threadCount-1))/threadCount;

  v_set KERNEL_ARG2(blockCount,threadCount)(perfTable, 0, perfTableSize);
  cudaDeviceSynchronize();

  for(int i = 0; i<peCount*funcCount; i++)
  {
    perfTable[i] = c0[i];
  }

  for(int i = 0; i<peCount*funcCount; i++)
  {
    perfTable[peCount*funcCount + i] = c1[i];
  }

  // for(int i = 0; i<2; i++)
  // {
  //   printf("c[%d]\n", i);
  //   for(int j = 0; j<38; j++)
  //   {
  //     for(int k = 0; k<37; k++)
  //     {
    
  //       printf("%.02f ", perfTable[i*peCount*funcCount + j*peCount + k]);
  //     }
  //     printf("\n");
  //   }
  // }

  return perfTable;
}


// Kernel function to add the elements of two arrays
__global__ void t_mult(float *designSpaceTensor, float *perfTable, float* latencyTensor, int designPointsCount,int peCount,int funcCount)
{
  int designPointSize = peCount*funcCount;
  int thrdDesignPointIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int perfTableC1Offset = designPointSize;
  int globalIndex = blockIdx.y*designPointSize + thrdDesignPointIndex;
  int stride = peCount;
  if(thrdDesignPointIndex < peCount)
  {
    for (int dpIdx = globalIndex, perfIdx = thrdDesignPointIndex; perfIdx < designPointSize; dpIdx += stride, perfIdx += stride)
    {
        latencyTensor[dpIdx] = perfTable[perfIdx] + designSpaceTensor[dpIdx]*perfTable[perfIdx + perfTableC1Offset];
        // if(threadIdx.x == 0)
        // {
        //   printf("G %d, dpIdx %d\n",globalIndex, dpIdx);
        // }
        // __syncthreads();
    }
  }
}

float* createAndSetCudaManagedMemory(unsigned int size)
{
  float* mem; 
  cudaMallocManaged(&mem, size*sizeof(float));

  unsigned int threadCount = 1024;
  unsigned int blockCount = (size+(threadCount-1))/threadCount;

  v_set KERNEL_ARG2(blockCount,threadCount)(mem, 0, size);
  cudaDeviceSynchronize();

  return mem;
}

int main(void)
{
  
  unsigned int designPointsCount;
  unsigned int peCount;
  unsigned int funcCount;
  unsigned int coefficientCount;

  float* designSpaceTensor = loadDSTensor(designPointsCount, peCount, funcCount);
  float* perfTable = loadPerfTable(coefficientCount, peCount, funcCount);

  unsigned int designSpaceSize = designPointsCount*peCount*funcCount;

  float* latencyTensor = createAndSetCudaManagedMemory(designSpaceSize);
  
  dim3 dimGrid;
  unsigned int threadCount = 32;
  dimGrid.x = (peCount+(threadCount-1))/threadCount;
  dimGrid.y = designPointsCount;

  t_mult KERNEL_ARG2(dimGrid,threadCount)(designSpaceTensor,perfTable,latencyTensor,designPointsCount,peCount,funcCount);
  cudaDeviceSynchronize();


  // auto graph = loadcoAuthors();
  // int m,n;

  // gettimeofday(&t1, 0);

  // unsigned int* arr = flattenGraph(graph,m,n);
  // unsigned int* coAuthorSums = NULL;
  // auto rankings = rankAuthorByCoAuthorCount(arr, &coAuthorSums, m, n);
  // unsigned int maxCoAuthorCount = rankings.begin()->second + 1; // 0 coauthors 
  // auto histogram = coAuthorCountHistogram(coAuthorSums, m, n, maxCoAuthorCount);

  // gettimeofday(&t2, 0);

  // int count = 0; 
  // printf("TOP 10 AUTHORS!\n");
  // for(auto author = rankings.begin(); count < 10; author++)
  // {
  //     printf("AUTHOR[%d] CoAuthorCount %d\n", author->first, author->second);
  //     count++;
  // }
  // printf("COAUTHOR HISTOGRAM!\n");
  // for(auto bin : histogram)
  // {
  //   printf("bin[%d] = %d\n", bin.first, bin.second);
  // }


  // printf("Authors that share a coAuthor Count %d\n", getAuthorThatShareCoAuthorCount(arr,m, n));

  // // Free memory
  // cudaFree(arr);
  // cudaFree(coAuthorSums);

  // double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

  // printf("Time to Evaluate:  %3.1f ms \n", time);

  return 0;
}
