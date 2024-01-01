#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#define  Max(a,b) ((a)>(b)?(a):(b))
#define  Min(a,b) ((a)<(b)?(a):(b))

#define  N   (2*2*2*2*2*2+2)

double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
int rank, size, *i_s, *i_f, buf, cub, bufN, buf0;
int *tag;
int reco1, reco2;

double eps, s;
//double A [N][N][N];
double ***F, ***S;
double **cub_send1, *cub_send2, **cub_rec1, *cub_rec2;
double t1, t2;
MPI_Request *req;
MPI_Status *stat;

void relax();
void init();
void verify(); 

int main(int an, char **as)
{
    MPI_Init(&an, &as);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0)
        t1 = MPI_Wtime();

    req = (MPI_Request *) malloc(2 * size * sizeof(MPI_Request));
    stat = (MPI_Status *) malloc(2 * size * sizeof(MPI_Status));
    tag = (int *) malloc(2 * size * sizeof(int));

    for (int root = 0; root < size; ++root) {
        tag[2 * root] = root;
        tag[2 * root + 1] = root;
    }

    i_s = (int *) malloc(size * sizeof(int));
    i_f = (int *) malloc(size * sizeof(int));

    cub = N / size;

    for (int root = 0; root < size; ++root) {
        i_s[root] = cub * root;
        if (root == size - 1) {
            i_f[root] = N - 1;
        } else {
            i_f[root] = i_s[root] + cub - 1;
        }
    }

    buf = i_f[rank] - i_s[rank] + 1;
    bufN = i_f[size - 1] - i_s[size - 1] + 1;
    buf0 = i_f[0] - i_s[0] + 1;
    reco1 = buf0 * buf * N;
    reco2 = bufN * buf * N;

    F = (double ***) malloc(N * sizeof(double **));
    for (i = 0; i < N; ++i) {
        F[i] = (double **) malloc(buf * sizeof(double *));
        for (j = 0; j < buf; ++j) 
            F[i][j] = (double *) malloc(N * sizeof(double));
    }

    S = (double ***) malloc(buf * sizeof(double **));
    for  (i = 0; i < buf; ++i) {
        S[i] = (double **) malloc(N * sizeof(double **));
        for (j = 0; j < N; ++j)
            S[i][j] = (double *) malloc(N * sizeof(double));
    }

    cub_send1 = (double **) malloc((size - 1) * sizeof(double *));
    for (int root = 0; root < size - 1; ++root) {
        cub_send1[root] = (double *) malloc(reco1 * sizeof(double));
    }

    cub_send2 = (double *) malloc(reco2 * sizeof(double));

    cub_rec1 = (double **) malloc((size - 1) * sizeof(double *));
    for (int root = 0; root < size - 1; ++root) {
        cub_rec1[root] = (double *) malloc(reco1 * sizeof(double));
    }
    
    cub_rec2 = (double *) malloc(reco2 * sizeof(double));

	int it;

	init();

	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
        if (rank == 0)
            //printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}

	verify();

    for (int root = 0; root < size - 1; ++root) {
        free(cub_send1[root]);
    }

    free(cub_send1);
    free(cub_send2);

    for (int root = 0; root < size - 1; ++root) {
        free(cub_rec1[root]);
    }

    free(cub_rec1);
    free(cub_rec2);

    for (i = 0; i < N; ++i) {
        for (j = 0; j < buf; ++j) 
            free(F[i][j]);

        free(F[i]);
    }

    free(F);

    for (i = 0; i < buf; ++i) {
        for (j = 0; j < N; ++j) 
            free(S[i][j]);

        free(S[i]);
    }

    free(S);

    free(req);
    free(stat);

    free(i_s);
    free(i_f);
    free(tag);

    MPI_Finalize();

    if(rank == 0) {
        t2 = MPI_Wtime();
        printf("%.3lf\n", t2 - t1);
    }

	return 0;
}


void init()
{ 
	for(i=0; i<=N-1; i++)
	for(j=i_s[rank]; j<=i_f[rank]; j++)
	for(k=0; k<=N-1; k++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
		F[i][j - i_s[rank]][k]= 0.;
		else F[i][j - i_s[rank]][k]= ( 4. + i + j + k) ;
	}

	for(i=i_s[rank]; i<=i_f[rank]; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
		S[i - i_s[rank]][j][k]= 0.;
		else S[i - i_s[rank]][j][k]= ( 4. + i + j + k) ;
	}

} 

void relax()
{
	for(i=1; i<=N-2; i++)
	for(j=Max(i_s[rank], 1); j<=Min(i_f[rank], N-2); j++)
	for(k=1; k<=N-2; k++)
	{
		F[i][j - i_s[rank]][k] = (F[i-1][j - i_s[rank]][k]+F[i+1][j - i_s[rank]][k])/2.;
	}

    for (int root = 0; root < size - 1; ++root) {
        for (i = i_s[root]; i <= i_f[root]; ++i) {
            for (j = i_s[rank]; j <= i_f[rank]; ++j) {
                for (k = 0; k < N; ++k) {
                    cub_send1[root][(i - i_s[root]) * buf * N + (j - i_s[rank]) * N + k] = F[i][j - i_s[rank]][k];
                }
            }
        }
    }

    for (i = i_s[size - 1]; i <= i_f[size - 1]; ++i) {
        for (j = i_s[rank]; j <= i_f[rank]; ++j) {
            for (k = 0; k < N; ++k) {
                cub_send2[(i - i_s[size - 1]) * buf * N + (j - i_s[rank]) * N + k] = F[i][j - i_s[rank]][k];
            }
        }
    }

    for (int root = 0; root < size - 1; ++root) {
        MPI_Isend(&cub_send1[root][0], reco1, MPI_DOUBLE, root, tag[2 * rank], MPI_COMM_WORLD, &req[2 * root]);
    }

    MPI_Isend(&cub_send2[0], reco2, MPI_DOUBLE, size - 1, tag[2 * rank], MPI_COMM_WORLD, &req[2 * (size - 1)]);

    for (int root = 0; root < size - 1; ++root) {
        MPI_Irecv(&cub_rec1[root][0], reco1, MPI_DOUBLE, root, tag[2 * root + 1], MPI_COMM_WORLD, &req[2 * root + 1]);
    }

    MPI_Irecv(&cub_rec2[0], reco2, MPI_DOUBLE, size - 1, tag[2 * (size - 1) + 1], MPI_COMM_WORLD, &req[2 * (size - 1) + 1]);

    MPI_Waitall(2 * size, req, stat);

    for (int root = 0; root < size - 1; ++root) {
        for (i = i_s[rank]; i <= i_f[rank]; ++i) {
            for (j = i_s[root]; j <= i_f[root]; ++j) {
                for (k = 0; k < N; ++k) {
                    S[i - i_s[rank]][j][k] = cub_rec1[root][(i - i_s[rank]) * buf0 * N + (j - i_s[root]) * N + k];
                }
            }
        }
    }

    for (i = i_s[rank]; i <= i_f[rank]; ++i) {
        for (j = i_s[size - 1]; j <= i_f[size - 1]; ++j) {
            for (k = 0; k < N; ++k) {
                S[i - i_s[rank]][j][k] = cub_rec2[(i - i_s[rank]) * bufN * N + (j - i_s[size - 1]) * N + k];
            }
        }
    }


	for(i=Max(i_s[rank], 1); i<=Min(i_f[rank], N-2); i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
		S[i - i_s[rank]][j][k] =(S[i - i_s[rank]][j-1][k]+S[i - i_s[rank]][j+1][k])/2.;
	}

    double local_eps = eps;

	for(i=Max(i_s[rank], 1); i<=Min(i_f[rank], N-2); i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
		double e;
		e=S[i - i_s[rank]][j][k];
		S[i - i_s[rank]][j][k] = (S[i - i_s[rank]][j][k-1]+S[i - i_s[rank]][j][k+1])/2.;
		local_eps=Max(local_eps,fabs(e-S[i - i_s[rank]][j][k]));
	}

    for (int root = 0; root < size - 1; ++root) {
        for (i = i_s[rank]; i <= i_f[rank]; ++i) {
            for (j = i_s[root]; j <= i_f[root]; ++j) {
                for (k = 0; k < N; ++k) {
                    cub_send1[root][(i - i_s[rank]) * buf0 * N + (j - i_s[root]) * N + k] = S[i - i_s[rank]][j][k];
                }
            }
        }
    }

    for (i = i_s[rank]; i <= i_f[rank]; ++i) {
        for (j = i_s[size - 1]; j <= i_f[size - 1]; ++j) {
            for (k = 0; k < N; ++k) {
                cub_send2[(i - i_s[rank]) * bufN * N + (j - i_s[size - 1]) * N + k] = S[i - i_s[rank]][j][k];
            }
        }
    }

    for (int root = 0; root < size - 1; ++root) {
        MPI_Isend(&cub_send1[root][0], reco1, MPI_DOUBLE, root, tag[2 * rank], MPI_COMM_WORLD, &req[2 * root]);
    }

    MPI_Isend(&cub_send2[0], reco2, MPI_DOUBLE, size - 1, tag[2 * rank], MPI_COMM_WORLD, &req[2 * (size - 1)]);

    for (int root = 0; root < size - 1; ++root) {
        MPI_Irecv(&cub_rec1[root][0], reco1, MPI_DOUBLE, root, tag[2 * root + 1], MPI_COMM_WORLD, &req[2 * root + 1]);
    }

    MPI_Irecv(&cub_rec2[0], reco2, MPI_DOUBLE, size - 1, tag[2 * (size - 1) + 1], MPI_COMM_WORLD, &req[2 * (size - 1) + 1]);

    MPI_Waitall(2 * size, &req[0], &stat[0]);

    for (int root = 0; root < size - 1; ++root) {
        for (i = i_s[root]; i <= i_f[root]; ++i) {
            for (j = i_s[rank]; j <= i_f[rank]; ++j) {
                for (k = 0; k < N; ++k) {
                    F[i][j - i_s[rank]][k] = cub_rec1[root][(i - i_s[root]) * buf * N + (j - i_s[rank]) * N + k];
                }
            }
        }
    }

    for (i = i_s[size - 1]; i <= i_f[size - 1]; ++i) {
        for (j = i_s[rank]; j <= i_f[rank]; ++j) {
            for (k = 0; k < N; ++k) {
                F[i][j - i_s[rank]][k] = cub_rec2[(i - i_s[size - 1]) * buf * N + (j - i_s[rank]) * N + k];
            }
        }
    }

    MPI_Allreduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

}

void verify()
{
	double ss;

	ss=0.;
	for(i=i_s[rank]; i<=i_f[rank]; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		ss=ss+S[i - i_s[rank]][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
	}

    MPI_Allreduce(&ss, &s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	if(rank == 0)
        printf("  S = %f\n",s);
}
