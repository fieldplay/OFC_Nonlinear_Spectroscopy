#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

typedef double complex cmplx;

typedef struct parameters{
    double freq_unit;
    int N_freq;
    int N_comb;
    int N_res;
    double* frequency;
    double* field_freq1;
    double* field_freq2;
    double gamma_comb;
    double delta_freq;
    int N_terms;
    int* index;
    double* modulations;
} parameters;

typedef struct molecule{
    int nDIM;
    double* energies;
    double* gamma;
    cmplx* mu;
    cmplx* polarization_mnv;
} molecule;

//====================================================================================================================//
//                                                                                                                    //
//                                        AUXILIARY FUNCTIONS FOR MATRIX OPERATIONS                                   //
//   ---------------------------------------------------------------------------------------------------------------  //
//    Given a matrix or vector and their dimensionality, these routines perform the operations of printing, adding,   //
//        scaling, copiesing to another compatible data structure, finding trace, or computing the maximum element.   //
//                                                                                                                    //
//====================================================================================================================//


void print_complex_mat(const cmplx *A, const int nDIM)
//----------------------------------------------------//
// 	          PRINTS A COMPLEX MATRIX                 //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e + %3.3eJ  ", creal(A[i * nDIM + j]), cimag(A[i * nDIM + j]));
		}
	    printf("\n");
	}
	printf("\n\n");
}

void print_complex_vec(const cmplx *A, const int vecDIM)
//----------------------------------------------------//
// 	          PRINTS A COMPLEX VECTOR                 //
//----------------------------------------------------//
{
	int i;
	for(i=0; i<vecDIM; i++)
	{
		printf("%3.3e + %3.3eJ  ", creal(A[i]), cimag(A[i]));
	}
	printf("\n");
}

void print_double_mat(const double *A, const int nDIM)
//----------------------------------------------------//
// 	            PRINTS A REAL MATRIX                  //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e  ", A[i * nDIM + j]);
		}
	    printf("\n");
	}
	printf("\n\n");
}

void print_double_vec(const double *A, const int vecDIM)
//----------------------------------------------------//
// 	          PRINTS A REAL VECTOR                    //
//----------------------------------------------------//
{
	int i;
	for(i=0; i<vecDIM; i++)
	{
		printf("%3.3e  ", A[i]);
	}
	printf("\n");
}


void copy_mat(const cmplx *A, cmplx *B, const int nDIM)
//----------------------------------------------------//
// 	        COPIES MATRIX A ----> MATRIX B            //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            B[i * nDIM + j] = A[i * nDIM + j];
        }
    }
}

void add_mat(const cmplx *A, cmplx *B, const int nDIM1, const int nDIM2)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM1; i++)
    {
        for(j=0; j<nDIM2; j++)
        {
            B[i * nDIM2 + j] += A[i * nDIM2 + j];
        }
    }
}


void add_vec(const double *A, double *B, const int nDIM)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        B[i] += A[i];
    }
}

void scale_mat(cmplx *A, const double factor, const int nDIM1, const int nDIM2)
//----------------------------------------------------//
// 	     SCALES A BY factor ----> MATRIX B = A + B    //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM1; i++)
    {
        for(int j=0; j<nDIM2; j++)
        {
            A[i * nDIM2 + j] *= factor;
        }
    }
}


void scale_vec(double *A, const double factor, const int nDIM)
//----------------------------------------------------//
// 	     SCALES A BY factor ----> VECTOR B = A + B    //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        A[i] *= factor;
    }
}


cmplx complex_trace(const cmplx *A, const int nDIM)
//----------------------------------------------------//
// 	                 RETURNS TRACE[A]                 //
//----------------------------------------------------//
{
    cmplx trace = 0.0 + I * 0.0;
    for(int i=0; i<nDIM; i++)
    {
        trace += A[i*nDIM + i];
    }
    printf("Trace = %3.3e + %3.3eJ  \n", creal(trace), cimag(trace));

    return trace;
}


double complex_abs(cmplx z)
//----------------------------------------------------//
// 	            RETURNS ABSOLUTE VALUE OF Z           //
//----------------------------------------------------//
{

    return sqrt((creal(z)*creal(z) + cimag(z)*cimag(z)));
}


double complex_max_element(const cmplx *A, const int nDIM)
//----------------------------------------------------//
// 	   RETURNS ELEMENT WITH MAX ABSOLUTE VALUE        //
//----------------------------------------------------//
{
    double max_el = A[0];
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            if(complex_abs(A[i * nDIM + j]) > max_el)
            {
                max_el = complex_abs(A[i * nDIM + j]);
            }
        }
    }
    return max_el;
}

double vec_max(const double *const A, const int nDIM)
//----------------------------------------------------//
// 	   RETURNS ELEMENT WITH MAX ABSOLUTE VALUE        //
//----------------------------------------------------//
{
    double max_el = A[0];
    for(int i=0; i<nDIM; i++)
    {
        if(A[i] > max_el)
        {
            max_el = A[i];
        }
    }
    return max_el;
}

double vec_sum(const double *const A, const int nDIM)
//----------------------------------------------------//
// 	            RETURNS SUM OF VECTOR ELEMENTS        //
//----------------------------------------------------//
{
    double sum = 0.0;
    for(int i=0; i<nDIM; i++)
    {
        sum += A[i];
    }
    return sum;
}


double vec_diff_norm(const double *const A, const double *const B, const int nDIM)
//----------------------------------------------------//
// 	   RETURNS L-1 NORM OF VECTOR DIFFERENCE          //
//----------------------------------------------------//
{
    int nfrac = (int)(nDIM/2.2);
    double norm = 0.0;
    double norm_long_wavelength = 0.0;
    for(int i=0; i<nfrac; i++)
    {
        norm_long_wavelength += fabs(A[i]-B[i]);
    }

    for(int i=0; i<nDIM; i++)
    {
        norm += fabs(A[i]-B[i]);
    }
    printf("norm = %g, norm1 = %g \n", norm, norm_long_wavelength);
    return norm;
}


//====================================================================================================================//
//                                                                                                                    //
//                                              INTEGRALS OF SPECTROSCOPIC TERMS                                      //
//   ---------------------------------------------------------------------------------------------------------------  //
//    Given a set of modulations (\omega_M1; \omega_M2; \omega_M3) and permuted indices (m, n, v) calculate the       //
//    non-linear OFC spectroscopic integrals for each spectroscopic term occurring in the susceptibility function     //
//    \chi_3 using analytical forms developed via solving the Cauchy integral with frequencies in the upper z-plane   //
//====================================================================================================================//


//====================================================================================================================//
//                                                                                                                    //
//                                                  INTEGRAL OF TYPE A-1                                              //
//   ---------------------------------------------------------------------------------------------------------------  //
//      I1 = 1/(ABC) + 1/(ABD) + 1/(BCE) - 1/(ADE*)                                                                   //
//      where:                                                                                                        //
//      A -> {\omega} + \omega_M_i + m_i(\Delta \omega) + \Omega_b + i\tau                                            //
//      B -> \omega_M_k + m_k(\Delta \omega) + \Omega_a + i\tau                                                       //
//      C -> \omega_M_k + \omega_M_j + (m_k + m_j)(\Delta \omega) + \Omega_b + 2i\tau                                 //
//      D -> {\omega} + \omega_M_i - \omega_M_j + (m_i - m_j)(\Delta \omega) + \Omega_a + 2i\tau                      //
//      E -> -{\omega} + \omega_M_k + \omega_M_j - \omega_M_i + (m_k + m_j - m_i)(\Delta \omega) + 3i\tau             //
//                                                                                                                    //
//====================================================================================================================//

void pol3_int1(molecule* mol, parameters* params, const cmplx wg_c, const cmplx wg_b, const cmplx wg_a, const int sign)
{
    double delta_freq = params->delta_freq;
    int N_terms = params->N_terms;
    double gamma_comb = params->gamma_comb;

    double omega_M_k = params->modulations[0];
    double omega_M_j = params->modulations[1];
    double omega_M_i = params->modulations[2];

    int m_k_0 = ceil((- omega_M_k - crealf(wg_a))/delta_freq);
    int m_j_0 = ceil((- omega_M_k - omega_M_j - crealf(wg_b))/delta_freq) - m_k_0;

    #pragma omp parallel for
    for(int out_i = 0; out_i < params->N_freq; out_i++)
        {
            const double omega = params->frequency[out_i];
            int m_i_0 = m_k_0 + m_j_0 - ceil((omega - omega_M_k - omega_M_j + omega_M_i)/delta_freq);
            cmplx result = 0. + 0. * I;

            for(int m_i = m_i_0 - N_terms; m_i < m_i_0 + N_terms; m_i++)
            {
                const cmplx term_A = omega + omega_M_i + m_i * delta_freq + wg_b + gamma_comb * I;
                for(int m_j = m_j_0 - N_terms; m_j < m_j_0 + N_terms; m_j++)
                {
                    const cmplx term_D = omega + omega_M_i - omega_M_j + (m_i - m_j) * delta_freq + wg_a + 2 * I * gamma_comb;
                    for(int m_k = m_k_0 - N_terms; m_k < m_k_0 + N_terms; m_k++)
                    {
                        const cmplx term_B = omega_M_k + m_k * delta_freq + wg_a + gamma_comb * I;
                        const cmplx term_C = omega_M_k + omega_M_j + (m_k + m_j) * delta_freq + wg_b + 2 * I * gamma_comb;
                        const cmplx term_E = -omega + (omega_M_k + omega_M_j - omega_M_i) + (m_k + m_j - m_i) * delta_freq + 3 * I * gamma_comb;
                        const cmplx term_E_star = omega - (omega_M_k + omega_M_j - omega_M_i) - (m_k + m_j - m_i) * delta_freq + 3 * I * gamma_comb;
                        result += (1./(term_A * term_D * term_E_star)) + (1./(term_B * term_C * term_E));
                    }

                }
            }

            mol->polarization_mnv[out_i] += sign*result/(omega + wg_c);
        }
}

//====================================================================================================================//
//                                                                                                                    //
//                                                  INTEGRAL OF TYPE A-2                                              //
//   ---------------------------------------------------------------------------------------------------------------  //
//                                                                                                                    //
//====================================================================================================================//
void pol3_int2(molecule* mol, parameters* params, const cmplx wg_3, const cmplx wg_2, const cmplx wg_1, const int sign)
{

}

//====================================================================================================================//
//                                                                                                                    //
//                                                  INTEGRAL OF TYPE A-3                                              //
//   ---------------------------------------------------------------------------------------------------------------  //
//                                                                                                                    //
//====================================================================================================================//
void pol3_int3(molecule* mol, parameters* params)
{

}

void CalculateResponse(molecule* mol, parameters* params)
{
    int m, n, v, l, nDIM;

    nDIM = mol->nDIM;
    l = 0;
    m = params->index[0];
    n = params->index[1];
    v = params->index[2];

    cmplx wg_nl = mol->energies[n] - mol->energies[l] + I * mol->gamma[n * nDIM + l];
    cmplx wg_vl = mol->energies[v] - mol->energies[l] + I * mol->gamma[v * nDIM + l];
    cmplx wg_ml = mol->energies[m] - mol->energies[l] + I * mol->gamma[m * nDIM + l];
    cmplx wg_nv = mol->energies[n] - mol->energies[v] + I * mol->gamma[n * nDIM + v];
    cmplx wg_mv = mol->energies[m] - mol->energies[v] + I * mol->gamma[m * nDIM + v];
    cmplx wg_vm = mol->energies[v] - mol->energies[m] + I * mol->gamma[v * nDIM + m];
    cmplx wg_vn = mol->energies[v] - mol->energies[n] + I * mol->gamma[v * nDIM + n];
    cmplx wg_mn = mol->energies[m] - mol->energies[n] + I * mol->gamma[m * nDIM + n];
    cmplx wg_nm = mol->energies[n] - mol->energies[m] + I * mol->gamma[n * nDIM + m];

    pol3_int1(mol, params, -conj(wg_vl), -conj(wg_nl), -conj(wg_ml), -1);
    pol3_int1(mol, params, -conj(wg_nv), -conj(wg_mv), wg_vl, 1);
    pol3_int1(mol, params, -conj(wg_nv), wg_vm, -conj(wg_ml), 1);
    pol3_int1(mol, params, -conj(wg_mn), wg_nl, wg_vl, -1);
    pol3_int1(mol, params, wg_vn, -conj(wg_nl), -conj(wg_ml), 1);
    pol3_int1(mol, params, wg_nm, -conj(wg_mv), wg_vl, -1);
    pol3_int1(mol, params, wg_nm, wg_mv, -conj(wg_ml), -1);
    pol3_int1(mol, params, wg_ml, wg_nl, wg_vl, 1);

}