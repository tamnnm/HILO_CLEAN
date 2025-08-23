# Import modules
import numpy as np
import os
import xarray as xr


# * Used for all resolution
# -------------------- Calculation of vorticity -------------------- #
def vort_cal(Uwind, Vwind, Rx, Ry, R_time, name=""):
    # Uwind and Vwind are 2D arrays of wind speed in the x and y direction
    # respectively. The function returns a 2D array of vorticity.
    # The vorticity is calculated as the curl of the wind field.
    # The curl is calculated as the derivative of the y-component of the wind
    # with respect to x minus the derivative of the x-component of the wind
    # with respect to y.
    # The derivatives are calculated using the numpy gradient function.
    # The vorticity is returned in units of s^-1.
    """_summary_
    Equation: Dv/Dx - Du/Dy
    Args:
        Uwind (DataArray, dim = (Nx,Ny)): Vertical wind
        Vwind (DataArray, dim = (Nx,Ny)): Horizontal wind
        Rx (DataArray, dim = Nx): Longitude
        Ry (DataArray, dim = Ny): Latitude
    Returns:
        Vort (DataArray): Vorticity
    """

    if os.path.exists(name):
        Vort_Darray = xr.open_dataarray(name)
    else:
        # * Constant
        deg2rad = np.pi/180
        R_e = 6378161  # unit:meters
        Factor = deg2rad*R_e

        Nx = len(Rx)
        Ny = len(Ry)
        N_time = len(R_time)
        Vort = np.zeros((N_time, Nx, Ny))
        for t in range(N_time):
            V = Vwind[t]
            U = Uwind[t]
            print(V.shape, U.shape)
            vort = np.zeros((Nx, Ny))
            for i in range(1, Nx-1):
                for j in range(1, Ny-1):
                    # Approximation of Dx,Dy
                    Dx = (Rx[i+1]-Rx[i-1])*Factor*np.cos(Ry[j]*deg2rad)
                    Dy = (Ry[j+1]-Ry[j-1])*Factor
                    vort[i, j] = (V[i+1, j]-V[i-1, j])/Dx - \
                        (U[i+1, j]-U[i, j-1])/Dy
                    print(i, j)
            Vort[t] = vort
        Vort_Darray = xr.DataArray(Vort, coords=[R_time, Rx, Ry], dims=[
            'time', 'lon', 'lat'], name="vort")
        if name != "":
            Vort_Darray.to_netcdf(name)
        else:
            Vort_Darray.to_netcdf("vort.1881.nc")
    return Vort_Darray

# * Only used for resolution <= 0.35 degree
# ----- Solve for LU decomposition of partial derivative matrix ---- #


def LU_SPLINE(IN, OUT, A, B, L, W, IVJ, IPU, P3, NX, NY, MMX, MMY):
    NX1 = NX-1
    NX2 = NX-2
    NX3 = NX-3
    R = np.zeros(NX)
    # @param: R: Intermediate array for LU decomposition
    if IVJ != 0:
        for j in range(0, NY, IPU):
            R1 = P3*W[0]**2*(IN[j, 1]-IN[j, 0])
            R2 = P3*W[1]**2*(IN[j, 2]-IN[j, 1])
            R[0] = R1+R2
            R[0] = R[0]-W[0]*OUT[j, 0]
            R1 = R2
            for i in range(2, NX2):
                R2 = P3*W[i]**2*(IN[j, i+1]-IN[j, i])
                R[i-1] = R1+R2
                R1 = R2
            R2 = P3*W[NX2]**2*(IN[j, NX1]-IN[j, NX2])
            # The NX2th element & will not change
            R[NX3] = R1+R2
            R[NX3] = R[NX3]-W[NX2]*OUT[j, NX1]
            # Normalization
            R[0] = R[0]/B[0]
            for i in range(1, NX2):
                R[i] = (R[i]-A[i-1]*R[i-1])/B[i]
            # Backward substitution
            # ? The last element (NX2) is not included
            for i in reversed(range(NX3)):
                R[i] = R[i]-L[i]*R[i+1]

            # Move the values to the output
            for i in range(1, NX1):
                OUT[j, i] = R[i-1]
    else:
        for j in range(0, NY, IPU):
            R1 = P3*W[0]**2*(IN[1, j]-IN[0, j])
            R2 = P3*W[1]**2*(IN[2, j]-IN[1, j])
            R[0] = R1+R2
            R[0] = R[0]-W[0]*OUT[0, j]
            R1 = R2
            for i in range(2, NX2):
                R2 = P3*W[i]**2*(IN[i+1, j]-IN[i, j])
                R[i-1] = R1+R2
                R1 = R2
            R2 = P3*W[NX2]**2*(IN[NX1, j]-IN[NX2, j])
            # The NX2th element & will not change
            R[NX3] = R1+R2
            R[NX3] = R[NX3]-W[NX2]*OUT[NX1, j]
            # Normalization
            R[0] = R[0]/B[0]
            for i in range(1, NX2):
                R[i] = (R[i]-A[i-1]*R[i-1])/B[i]
            # Backward substitution
            # ? The last element (NX2) is not included
            # ? Last loop is R[NX3-1] is NX3th element
            for i in reversed(range(NX3)):
                R[i] = R[i]-L[i]*R[i+1]

            # Move the values to the output
            for i in range(1, NX1):
                OUT[i, j] = R[i-1]
    return OUT, R
# ------- Calculation of coefficient matrix for spline interpolation for each point ------- #


def COEFF_SPLINE(DATA, X, Y, NX, NY, PP, NB=4, EPS=1e-10):
    """_summary_
        - Bicubic spline interpolation use 16 points to decide a surface of 4 center points (the a_ij) -> Used to decide any point within the 4 center points.
        - Original version: a Long in FORTRAN
    Args:
        DATA(No_X,No_Y): 2D array of data
        X(No_X),Y(No_Y): 1D array of coordinate
        NX,NY: Number of points to be processed
        ! Default: No_X=NX, No_Y=NY
        PP: Parameter of the birational spline interpolation
            PP > -1
            PP = 0: Bilinear interpolation
            PP -> inf: Bi-cubic interpolation (main method)
        eps: Tolerance of the method
        NB: Number of boundary points (default = 4)
    Returns:
        Coef(NX,NY, 4, 4): 2D array of interpolated data
    """

    # Output array
    A = np.zeros((NX, NY, NB, NB))

    # Initial value of bound points
    # ? Do we actually need this?
    # ? Lúc sau đưa vào trong RATPE để tính LU decomposition thì thực chất nó là cái physical length of output
    # @param: MMX, MMY: Dimension of the output
    # Dimension of P,Q,R (interpolation sub-efficients)
    MMX, MMY = NX, NY
    # Dimension of the input
    No_X, No_Y = NX, NY

    # Quick-access parameters
    NX1 = NX-1
    NY1 = NY-1
    NX2 = NX-2
    NY2 = NY-2
    NX3 = NX-3
    NY3 = NY-3
    # Modulus of the number of points
    II, II1, II2, II3, JJ, JJ1, JJ2, JJ3 = NX % 4, NX1 % 4, NX2 % 4, NX3 % 4, NY % 4, NY1 % 4, NY2 % 4, NY3 % 4
    # Weight-scaled parameter for interpolation
    P1 = PP+1
    P2 = PP+2
    P3 = PP+3
    WK = 1/P1

    # * First check on the number of points
    if NX < 2 or NY < 2:
        print("Error: Not enough data points")
        return

    # * Initialize arrays and parameter
    # Intermediate arrays for interpolation
    P, Q, R = (np.zeros((NX, NY)) for i in range(3))
    # Intermediate diagonal arrays for LU decomposition
    AX, BX, LX = (np.zeros((NX)) for i in range(3))
    # @param AX: Lower diagonal + Upper diagonal
    #! This needed to be clarified but this is my guess. Or else, it is quite confuding
    #! B_k = B_k - C_k * L_k (L_k=A_k/B_k)
    # @param BX: Main diagonal
    # @param LX: Lower diagonal of L decomposition

    #! AX[0] = a2, BX[0] = b1, LX[0] = c2
    # Example matrix
    # [b1, c1, 0, 0]
    # [a2, b2, c2, 0]
    # [0, a3, b3, c3]
    # [0, 0, a4, b4]
    AY, BY, LY = (np.zeros((NY)) for i in range(3))
    # Sub-coefficients of the interpolation (a_ij)
    B, C, D, E = (np.zeros((NB, NB)) for i in range(4))
    # Weight of difference depending on grid
    WX, WY = np.zeros((NX1)), np.zeros((NY1))

    # * Weight production
    for i in range(II1):
        WX[i] = 1/(X[i+1]-X[i])
    for j in range(JJ1):
        WY[j] = 1/(Y[j+1]-Y[j])

    # ? This is the same as loop from (II1+1,NX1,1)
    # ? But we do this to focus on matrix of 4x4

    for i in range(II1, NX1, 4):
        for m in range(4):
            WX[i+m] = 1/(X[i+m+1]-X[i+m])
    for j in range(JJ1, NY1, 4):
        for m in range(4):
            WY[j+m] = 1/(Y[j+m+1]-Y[j+m])

    # Initialisation of sub-coefficients

    B[:, 0] = np.array([P2*WK, -WK, -WK, WK])
    B[:, 2] = np.array([-WK, P2*WK, WK, -WK])
    E = B

    # Check eps function
    def check_eps(a, eps=EPS):
        if type(a) is list:
            status = all(abs(i) < eps for i in a)
        else:
            status = abs(a) < eps
        if status == False:
            raise ValueError("Error: Singular matrix")
        else:
            return

    # * Partial derivatives at the boundaries
    # For x-direction (First and last row)
    WP1 = WX[0]
    WP2 = WX[-1]
    for j in range(JJ):
        P[0, j] = (DATA[1, j]-DATA[0, j])*WP1
        P[-1, j] = (DATA[-1, j]-DATA[-2, j])*WP2
    for j in range(JJ, NY, 4):
        for m in range(4):
            P[0, j+m] = (DATA[1, j+m]-DATA[0, j+m])*WP1
            P[-1, j+m] = (DATA[-1, j+m]-DATA[-2, j+m])*WP2
    # For y-direction (First and last column)
    WQ1 = WY[0]
    WQ2 = WY[-1]
    for i in range(II):
        Q[i, 0] = (DATA[i, 1]-DATA[i, 0])*WQ1
        Q[i, -1] = (DATA[i, -1]-DATA[i, -2])*WQ2
    for i in range(II, NX, 4):
        for m in range(4):
            Q[i+m, 0] = (DATA[i+m, 1]-DATA[i+m, 0])*WQ1
            Q[i+m, -1] = (DATA[i+m, -1]-DATA[i+m, -2])*WQ2

    # for xy direction
    # ! Only on 4 vertices
    R[0, 0] = 1/2*((P[0, 1]-P[0, 0])*WQ1+(Q[1, 0]-Q[0, 0])*WP1)
    R[0, -1] = 1/2*((P[0, -1]-P[0, -2])*WQ2+(Q[1, -1]-Q[0, -1])*WP1)
    R[-1, 0] = 1/2*((P[-1, 1]-P[-1, 0])*WQ1+(Q[-1, 0]-Q[-2, 0])*WP2)
    R[-1, -1] = 1/2*((P[-1, -1]-P[-1, -2])*WQ2+(Q[-1, -1]-Q[-2, -1])*WP2)

    # * LU decomposition (partial derivatives for the rest of the grid)
    # @param IVJ: 0 = iteration row-wise, 1 = iteration column-wise
    # @param IPU: Step for iteration
    # For x-direction
    # TODO: Coeff matrix for x
    if NX != 2:
        for i in range(II3):
            AX[i] = WX[i+1]
            BX[i] = P2 * (WX[i]+WX[i+1])
            check_eps(BX[i])
        for i in range(II3, NX3, 4):
            for m in range(4):
                AX[i+m] = WX[i+m+1]
                BX[i+m] = P2 * (WX[i+m]+WX[i+m+1])
                check_eps(BX[i+m])
        BX[NX3] = P2*(WX[NX2]+WX[NX3])

        # LU decomposition
        for k in range(1, NX2):
            LX[k-1] = AX[k-1]/BX[k-1]
            BX[k] = BX[k]-LX[k-1]*AX[k-1]
        P, RX = LU_SPLINE(IN=DATA, OUT=P, A=AX, B=BX, L=LX, W=WX,
                          IVJ=0, IPU=1, P3=P3, NX=No_X, NY=No_Y, MMX=MMX, MMY=MMY)

    # TODO: Coef matrix for y
    if NY != 2:
        for j in range(JJ3):
            AY[j] = WY[j+1]
            BY[j] = P2 * (WY[j]+WY[j+1])
            check_eps(BY[j])
        for j in range(JJ3, NY3, 4):
            for m in range(4):
                AY[j+m] = WY[j+m+1]
                BY[j+m] = P2 * (WY[j+m]+WY[j+m+1])
                check_eps(BY[j+m])
        BY[NY2] = P2*(WY[NY2]+WY[NY1])
        for k in range(1, NY2):
            LY[k] = AY[k-1]/BY[k-1]
            BY[k] = BY[k]-LY[k]*AY[k-1]

        Q, RY = LU_SPLINE(IN=DATA, OUT=Q, A=AY, B=BY, L=LY, W=WY, IVJ=0, IPU=1, P3=P3,
                          NX=No_X, NY=No_Y, MMX=MMX, MMY=MMY)

    # TODO: Coef matrix for xy
    if NX != 2 and NY != 2:
        R, RXY = LU_SPLINE(IN=Q, OUT=R, A=AX, B=BX, L=LX, IVJ=0, W=WX,
                           IPU=NY1, P3=P3, NX=MMX, NY=MMY, MMX=MMX, MMY=MMY)
        R, RXY = LU_SPLINE(IN=P, OUT=R, A=AY, B=BY, L=LY, IVJ=1, W=WY,
                           IPU=1, P3=P3, NX=MMX, NY=MMY, MMX=MMX, MMY=MMY)

    # * Calculation of the sub-coefficients
    for i in range(NX1):
        # Weight-related matrix (x-direction)
        WA = (X[i+1]-X[i])/P3
        B[0, 1] = B[0, 0]*WA
        B[1, 1] = B[1, 0]*WA
        B[2, 1] = -B[0, 1]
        B[3, 1] = -B[1, 1]
        B[0, 3] = B[3, 1]
        B[1, 3] = B[1, 1]*P2
        B[2, 3] = B[1, 1]
        B[3, 3] = -B[1, 3]
        for j in range(NY1):
            C[0, 0] = DATA[i, j]
            C[0, 1] = Q[i, j]
            C[1, 0] = P[i, j]
            C[1, 1] = R[i, j]
            C[0, 2] = DATA[i, j+1]
            C[0, 3] = Q[i, j+1]
            C[1, 2] = P[i, j+1]
            C[1, 3] = R[i, j+1]
            C[2, 0] = DATA[i+1, j]
            C[2, 1] = Q[i+1, j]
            C[3, 0] = P[i+1, j]
            C[3, 1] = R[i+1, j]
            C[2, 2] = DATA[i+1, j+1]
            C[2, 3] = Q[i+1, j+1]
            C[3, 2] = P[i+1, j+1]
            C[3, 3] = R[i+1, j+1]
            for m in range(NB):
                for n in range(NB):
                    sum = 0
                    for o in range(NB):
                        sum = sum+B[m, o]*C[o, n]
                    D[m, n] = sum

            # Weight-related matrix (y-direction)
            WB = (Y[j+1]-Y[j])/P3
            E[0, 1] = E[0, 0]*WB
            E[1, 1] = E[1, 0]*WB
            E[2, 1] = -E[0, 1]
            E[3, 1] = -E[1, 1]
            E[0, 3] = E[3, 1]
            E[1, 3] = E[1, 1]*P2
            E[2, 3] = E[1, 1]
            E[3, 3] = -E[1, 3]
            for m in range(NB):
                for n in range(NB):
                    sum = 0
                    for o in range(NB):
                        sum = sum+D[m, o]*E[n, o]
                    A[i, j, m, n] = sum
    return


def VAL_SPLINE(COEF, X, Y, NX, NY, XX, YY, PP):
    """
    Args:
        COEF(NX,NY,4,4): Coefficient matrix
        X(No_X),Y(No_Y): Coordinate
        NX,NY: Number of points to be processed
        ! Default: No_X=NX, No_Y=NY
        XX,YY: Coordinate of the point to be interpolated
        PP: Parameter of the birational spline interpolation
            PP > -1
            PP = 0: Bilinear interpolation
            PP -> inf: Bi-cubic interpolation (main method)
    Returns:
        VAL: Interpolated value
    """

    # * Quick-access parameters
    i, j = 0, 0
    NX1 = NX-1
    NY1 = NY-1
    # -------------------- Find the index of XX in X ------------------- #
    l = 0 if XX < X[0] else NX1

    while True:
        if XX < X[i]:
            l = i
            i = 1
        elif XX <= X[i+1]:
            break
        else:
            if l > i+1:
                # Middle point
                k = (i+NX1)//2
                if XX < X[k]:
                    l = k
                else:
                    i = k
        if l <= i + 1:
            break

    # -------------------- Find the index of YY in Y ------------------- #
    l = 0 if YY < Y[0] else NY1

    while True:
        if YY < Y[j]:
            l = j
            j = 1
        elif YY <= Y[j+1]:
            break
        else:
            if l > j+1:
                # Middle point
                k = (j+NY1)//2
                if YY < Y[k]:
                    l = k
                else:
                    j = k
        if l <= j + 1:
            break

    # ------------------ Coefficient for interpolation ----------------- #
    # @param H: Coefficient matrix for interpolation
    H = np.zeros((4, 4))
    HH = np.zeros((4, 4))
    xxx = (XX-X[i])/(X[i+1]-X[i])
    yyy = (YY-Y[j])/(Y[j+1]-Y[j])
    W1 = 1-xxx
    W2 = 1-yyy
    for l in range(4):
        H[0, l] = W1
        H[1, l] = xxx
        H[2, l] = (W1**3)/(PP*xxx+1)
        H[3, l] = (xxx**3)/(PP*W1+1)
        HH[l, 0] = W2
        HH[l, 1] = yyy
        HH[l, 2] = (W2**3)/(PP*yyy+1)
        HH[l, 3] = (yyy**3)/(PP*W2+1)

    # ------------------ Interpolation of the point ------------------ #
    VAL = 0
    for l in range(4):
        for m in range(4):
            VAL = VAL+COEF[i, j, l, m]*H[l, m]*HH[l, m]
    return VAL
