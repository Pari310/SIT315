#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

#define N 400 // matrix size

int main(int argc, char* argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // check if we have enough processes
    if (size < 2) {
        cerr << "Error: Need at least 2 processes for this program." << endl;
        MPI_Finalize();
        return -1;
    }

    // allocate memory for matrices
    int A[N][N] = {0};
    int B[N][N] = {0};
    int C[N][N] = {0};

    // initialize matrices A and B on rank 0 process
    if (rank == 0) {
        srand(time(nullptr)); // seed the random number generator
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = rand() % 10; // fill A with random integer values between 0 and 99
                B[i][j] = rand() % 10; // fill B with random integer values between 0 and 99
            }
        }
    }

    // broadcast matrices A and B to all processes
    MPI_Bcast(A, N*N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N*N, MPI_INT, 0, MPI_COMM_WORLD);

    // start timer
    clock_t start_time = clock();

    // compute matrix multiplication
    int chunk_size = N / size;
    int start_row = rank * chunk_size;
    int end_row = (rank + 1) * chunk_size;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // gather results to rank 0 process
    MPI_Gather(C + start_row, chunk_size*N, MPI_INT, C, chunk_size*N, MPI_INT, 0, MPI_COMM_WORLD);

    // stop timer and calculate elapsed time
    clock_t end_time = clock();
    double elapsed_time = (static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC) * 1000;;

    // print results on rank 0 process
    if (rank == 0) {
        cout << "Matrix multiplication completed in " << elapsed_time << " millis." << endl;
    }

    MPI_Finalize();
    return 0;
}
