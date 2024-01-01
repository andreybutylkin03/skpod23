#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#define  Max(a,b) ((a)>(b)?(a):(b))
#define  Min(a,b) ((a)<(b)?(a):(b))

#define  N   (2*2*2*2*2*2*2*2+2)

double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k, z;
double t1, t2;
double eps, s;
double A [N][N][N];
int cub, size, root, *i_s, *i_f;

void relax();
void init();
void verify(); 

int main(int an, char **as)
{
	int it;

    omp_set_num_threads((int)strtol(as[1], NULL, 10));

    size = (int)strtol(as[1], NULL, 10);

    cub = N / size;

    i_s = (int *) malloc(size * sizeof(int));
    i_f = (int *) malloc(size * sizeof(int));

    for (root = 0; root < size; ++root) {
        i_s[root] = cub * root;
        if (root == size - 1) {
            i_f[root] = N - 1;
        } else {
            i_f[root] = i_s[root] + cub - 1;
        }
    }

	t1 = omp_get_wtime();	

	init();

	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		//printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}

	verify();

	t2 = omp_get_wtime();
	printf("%lf\n", t2-t1);

    free(i_s);
    free(i_f);

	return 0;
}


void init()
{ 
#pragma omp parallel shared(A, i_s, i_f, size) private(i, j, k, z)
    {
#pragma omp single
        {
            for (z = 0; z < size; ++z) {
#pragma omp task
                for(i=i_s[z]; i<=i_f[z]; i++)
                for(j=0; j<=N-1; j++)
                for(k=0; k<=N-1; k++)
                {
                    if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
                    A[i][j][k]= 0.;
                    else A[i][j][k]= ( 4. + i + j + k) ;
                }
            }
        }
    }
} 

void relax()
{
#pragma omp parallel shared(A, eps, i_s, i_f, size) private(i, j, k, z) 
    {
#pragma omp single
        {
            for(i=1; i<=N-2; i++) {
                for (z = 0; z < size; ++z){
#pragma omp task 
                    for(j=Max(i_s[z], 1); j<=Min(i_f[z], N-2); j++)
                    for(k=1; k<=N-2; k++)
                    {
                        A[i][j][k] = (A[i-1][j][k]+A[i+1][j][k])/2.;
                    }
                }
#pragma omp taskwait
            }
        }

#pragma omp single
        {
            for (z = 0; z < size; ++z){
#pragma omp task 
                for(i=Max(i_s[z], 1); i<=Min(i_f[z], N-2); i++)
                for(j=1; j<=N-2; j++)
                for(k=1; k<=N-2; k++)
                {
                    A[i][j][k] =(A[i][j-1][k]+A[i][j+1][k])/2.;
                } 
            }
        }
        
#pragma omp single
        {
            for (z = 0; z < size; ++z) {
                double ee;
#pragma omp task private(ee) 
                {
	                for(i=Max(i_s[z], 1); i<=Min(i_f[z], N-2); i++)
                    for(j=1; j<=N-2; j++)
                    for(k=1; k<=N-2; k++)
                    {
                        double vr=A[i][j][k];
                        A[i][j][k] = (A[i][j][k-1]+A[i][j][k+1])/2.;
                        ee=Max(ee,fabs(vr-A[i][j][k]));
                    }
#pragma omp critical
                    eps = Max(ee, eps);
                }
            }
        }
    }
}


void verify()
{
#pragma omp parallel shared(A, i_s, i_f, size) private(i, j, k, z)
    {
        double ss = 0;

#pragma omp single
        {
            for(z = 0; z < size; ++z) {
#pragma omp task private(ss)
                {
                    for(i=i_s[z]; i<=i_f[z]; i++)
                    for(j=0; j<=N-1; j++)
                    for(k=0; k<=N-1; k++)
                    {
                        ss=ss+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
                    }

#pragma omp atomic
                    s += ss;
                }
            }
        }
    }

    //printf("  S = %lf\n",s);
}
