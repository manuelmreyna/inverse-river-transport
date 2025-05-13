# This is a python translation of the program written by Hollenbeck, K. J. (1998)
#
# DEHOOG - numerical inverse Laplace transform
#
# Description: 
#   algorithm: de Hoog et al's quotient difference method with accelerated 
#   convergence for the continued fraction expansion
#
#   [de Hoog, F. R., Knight, J. H., and Stokes, A. N. (1982). An improved 
#   method for numerical inversion of Laplace transforms. S.I.A.M. J. Sci. 
#   and Stat. Comput., 3, 357-366.]
#
#   Modification: The time vector is split in segments of equal magnitude
#   which are inverted individually. This gives a better overall accuracy.   
#
#   Details: de Hoog et al's algorithm f4 with modifications (T->2*T and 
#    introduction of tol). Corrected error in formulation of z.
#
#   Copyright: Karl Hollenbeck
#             Department of Hydrodynamics and Water Resources
#             Technical University of Denmark, DK-2800 Lyngby
#             email: karl@isv16.isva.dtu.dk
#
#   22 Nov 1996, MATLAB 5 version 27 Jun 1997 updated 1 Oct 1998
#   IF YOU PUBLISH WORK BENEFITING FROM THIS M-FILE, PLEASE CITE IT AS:
#       Hollenbeck, K. J. (1998) INVLAP.M: A matlab function for numerical 
#       inversion of Laplace transforms by the de Hoog algorithm, 
#       http://www.isva.dtu.dk/staff/karl/invlap.htm 
#
#   Renamed by P. Renard to be integrated in the hytool toolbox.
#
# example: 
#   identity function in Laplace space:
#   F = lambda s: 1/(s**2)        
#   invlap(F, np.array([1,2,3]))                # gives [1,2,3]
#
# See also: stefhest
#

import numpy as np 

def invlap(F, t,args, alpha=0, tol=1e-10,M = 200):
    """
    F      = laplace-space function (string refering to an m-file),must
             have form F(s, args), where s is the Laplace parameter 
             and return column vector as result
    t      = column vector of times for which real-space function values
             are sought
    args   = other parameters of F
    alpha  = largest pole of F (default zero)
    tol    = numerical tolerance of approaching pole (default 1e-10)
    f      = vector of real-space values f(t)   
    """
    f = []
    # Split up t vector in pieces of the same order of magnitude, invert one piece
    # at a time. Simultaneous inversion for times covering several orders of
    # magnitudes gives inaccurate results for the small times.
    allt = t
    logallt = np.log10(allt)
    iminlogallt = int(np.floor(np.min(logallt)))
    imaxlogallt = int(np.ceil(np.max(logallt)))

    for ilogt in range(iminlogallt, imaxlogallt + 1):
        t = allt[(logallt >= ilogt) & (logallt < (ilogt + 1))]
        if t.size > 0:  # Maybe no elements in that magnitude
            T = np.max(t) * 2
            gamma = alpha - np.log(tol) / (2 * T)
            # NOTE: The correction alpha -> alpha-log(tol)/(2*T) is not in de Hoog's
            #   paper, but in Mathematica's Mathsource (NLapInv.m) implementation of
            #   inverse transforms
            nt = len(t)
            run = np.arange(2 * M + 1)  # so there are 2M+1 terms in Fourier series expansion

            # Find F argument, call F with it, get 'a' coefficients in power series
            s = gamma + 1j * np.pi * run / T
            a = F(s, *args)
            a = a.flatten()
            a[0] = a[0] / 2  # zero term is halved

            # Build up e and q tables. Superscript is now row index, subscript column
            # CAREFUL: paper uses null index, so all indices are shifted by 1 here
            e = np.zeros((2 * M + 1, M + 1), dtype=complex)
            q = np.zeros((2 * M, M + 1), dtype=complex)  # Column 0 (here: 1) does not exist
            e[:, 0] = np.zeros(2 * M + 1)
            q[:, 1] = a[1:2 * M + 1] / a[:2 * M]
            for r in range(2, M + 2):  # Step through columns (called r...)
                e[:2 * (M - r+1), r-1] = q[1:2 * (M - r+1) + 1, r-1] - q[:2 * (M - r+1), r-1] + e[1:2 * (M - r+1) + 1, r - 2]
                if r < M:  # One column fewer for q
                    rq = r + 1
                    q[:2 * (M - rq) + 2, rq-1] = q[1:2 * (M - rq) + 3, rq - 2] * e[1:2 * (M - rq) + 3, rq - 2] / e[:2 * (M - rq) + 2, rq - 2]

            # Build up d vector (index shift: 1)
            d = np.zeros(2 * M + 1, dtype=complex)
            d[0] = a[0]
#             d[1::2] = -q[:M, 1:M + 1].diagonal()
#             d[2::2] = -e[:M, 1:M + 1].diagonal()
            d[1::2] = -q[0, 1:M + 1].T
            d[2::2] = -e[0, 1:M + 1].T

            # Build up A and B vectors (index shift: 2)co
            # - now make into matrices, one row for each time
            A = np.zeros((2 * M + 2, nt), dtype=complex)
            B = np.zeros((2 * M + 2, nt), dtype=complex)
            A[1, :] = d[0] * np.ones(nt)
            B[:2, :] = np.ones((2, nt))
            z = np.exp(1j * np.pi * t / T)  # Row vector

            dz = np.outer(d, z)
            
            for n in range(2, 2 * M + 2):
                A[n, :] = A[n - 1, :] + dz[n - 1, :] * A[n - 2, :]
                B[n, :] = B[n - 1, :] + dz[n - 1, :] * B[n - 2, :]


            # Double acceleration
            h2M = 0.5 * (1 + (d[2 * M - 1] - d[2 * M]) * z)
            R2Mz = -h2M * (1 - np.sqrt(1 + d[2 * M] * z / (h2M**2)))
            A[2 * M + 1, :] = A[2 * M, :] + R2Mz * A[2 * M - 1, :]
            B[2 * M + 1, :] = B[2 * M, :] + R2Mz * B[2 * M - 1, :]

            # Inversion, vectorized for times, make result a column vector
            fpiece = (1 / T * np.exp(gamma * t) * np.real(A[2 * M + 1, :] / B[2 * M + 1, :])).flatten()
            f.extend(fpiece)

    return np.array(f)