from numpy import *




def square(a):
    '''
      Task: This function tests if a matrix is square. It returns True 
            if a represents a square matrix.
      Parameters: a is a numpy array.
      Example: square(array([[1, 2], [3, 4]])) must return True.
      Test: This function is is tested in tests/test_square.py
      
    '''

    return shape(a)[0]== shape(a)[1]
    raise Exception("Function not implemented")




'''
  This method is taken from the slides Chapter 2
'''
def swap(a, i, j):
    if len(shape(a)) == 1:
        a[i],a[j] = a[j],a[i] # unpacking
    else:
        a[[i, j], :] = a[[j, i], :]

'''
  This method is a modified version from the slides Chapter 2
'''
def gauss_elimination(a, b, verbose=False):
    n, m = shape(a)
    n2  = shape(b)[0]
    assert(n==n2)
    for k in range(n-1):
        for i in range(k+1, n):
            assert(a[k,k] != 0)
            if (a[i,k] != 0):
                lmbda = a[i,k]/a[k,k]
                a[i, k:n] = a[i, k:n] - lmbda*a[k, k:n]
                b[i] = b[i] - lmbda*b[k]
            if verbose:
                print(a, b)

'''
  This method is a modified version from the slides Chapter 2
'''
def gauss_elimination_pivot(a, b, verbose=False):
    n, m = shape(a)
    n2  = shape(b)[0]
    assert(n==n2)
    s = zeros(n)
    for i in range(n):
        s[i] = max(abs(a[i, :]))
    for k in range(n-1):
        # New in pivot version
        p = argmax(abs(a[k:,k])/s[k:]) + k
        swap(a, p, k)
        swap(b, p, k)
        swap(s, p, k)
        for i in range(k+1, n):
            assert(a[k,k] != 0)
            if (a[i,k] != 0):
                lmbda = a[i,k]/a[k,k]
                a[i, k:n] = a[i, k:n] - lmbda*a[k, k:n]
                b[i] = b[i] - lmbda*b[k]
            if verbose:
                print(a, b)

'''
  This method is a modified version from the slides Chapter 2
'''
def gauss_substitution(a, b):
    n, m = shape(a)
    n2 = shape(b)[0]
    assert(n==n2)
    x = zeros(n)
    for i in range(n-1, -1, -1): # decreasing index
        x[i] = (b[i] - dot(a[i,i+1:], x[i+1:]))/a[i,i]
    return x
'''
  This method is from the slides Chapter 2
'''
def gauss_pivot(a, b):
    gauss_elimination_pivot(a, b)
    return gauss_substitution(a, b)
'''
  This method is from the slides Chapter 2
'''
def gauss(a, b):
    gauss_elimination(a, b)
    return gauss_substitution(a, b)


def fit_poly_2(points):
    '''
       This function finds a polynomial P of degree 2 that passes 
       through the 3 points contained in list 'points'. It returns a numpy
       array containing the coefficient of a polynomial P: array([a0, a1, a2]),
       where P(x) = a0 + a1*x + a2*x**2. Every (x, y) in 'points' must 
       verify y = a0 + a1*x + a2*x**2.
      Parameters: points is a Python list of 3 pairs representing 2D points.
      Example: fit_poly_2([(0, -1), (1, -2), (2, -9)]) must return array([-1, 2, 3])
      Test: This function is is tested by the following functions in tests/test_fit_poly.py:
            - test_fit_poly_2 tests a basic fit
            - test_fit_poly_raises tests that the function raises an 
              AssertionError when the polynomial cannot be fit (for 
              instance, 3 points are aligned).
    '''

    p1, p2, p3 = points

    x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
    x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
    assert (abs(x1 * y2 - x2 * y1) > 1e-12)

    a = array([
        [1, p1[0], p1[0] ** 2],
        [1, p2[0], p2[0] ** 2],
        [1, p3[0], p3[0] ** 2]
    ])
    b = array([
        [p1[1]],
        [p2[1]],
        [p3[1]]
    ])

    return gauss_pivot(a,b)
    raise Exception("Function not implemented")


def fit_poly(points):
    '''
      This function is a generalization of the previous one. It 
      finds a polynomial P of degree n that passes 
      through the n+1 points contained in list 'points'. It 
      returns a numpy array containing the coefficient of a 
      polynomial P: array([a0, a1, ..., an]), where P(x) = a0 + 
      a1*x + a2*x**2 + ... + an*x**n. Every (x, y) in 'points' 
      must verify y = P(x).
      Parameters: points is a Python list of pairs representing 2D points.
      Examples: fit_poly([(0, -1), (1, -2), (2, -9)]) must return 
                array([-1, 2, -3]) (as in the previous function) fit_poly([(0, 2), 
                (1, 6), (2, 24), (3, 62)]) must return array([2, -1, 4, 1])
      Test: This function is is tested by the following functions in tests/test_fit_poly.py:
            - test_fit_poly tests a basic fit
            - test_fit_poly_n tests the fit on a random polynomial of degre <= 6.
    '''
    points_set=set(points)

    if(len(points_set)!= len(points)):
        raise AssertionError

    a = []
    b = []
    element = []
    for i in points:
        for n in range(len(points)):
            element.append(i[0] ** n)
        a.append(element.copy())
        b.append([i[1]])
        element.clear()

    a = array(a)
    b = array(b)
    return gauss_pivot(a, b)
    raise Exception("Function not implemented")




def tridiag_solver_n(n):
    '''
      This function returns the solution of the following tridiagonal equations:
            4x[1] - x[2] = 9
            -x[i-1] + 4x[i] - x[i+1] = 5, i=2,....n-1
            -x[n-1] + 4x[n] = 5
            The system of equations is the same
            as in problem 2.2.9 in the Textbook, except that here n is a 
            parameter of the function. All correct answers will be accepted,
            but you are strongly encouraged to exploit the tridiagonal nature
            of the system.
      Parameters: n is an integer representing the dimension of the system.
      Examples: tridiag_solver_n(2) must return array([41/15, 29/15])
      Test: This function is is tested by the function in tests/test_tridiag_solver.py.
    '''

    a=identity(n)*4
    for i in range(shape(a)[0]-1):
        a[i][i+1]=-1
        a[i+1][i]=-1
    b=ones(n)*5
    b[0]=9
    return gauss_pivot(a,b)
    raise Exception("Function not implemented")



def gauss_multiple(a, b):
    '''
      This function returns the solution of the system written as
            AX=B, where A is an n x n square matrix, and X and B are n x m matrices.
            It is equivalent to solving m systems of the form Ax=b, where
            x and b are column vectors. You have to extend the implementation
            of Gauss elimination presented in the course to work with m constant
            vectors instead of only 1. This is problem 2.1.14 in the textbook.
            It is up to you to decide if your function will modify a and b (the
            tests should work in both cases).
      Parameters: a is a numpy array representing a square matrix. b is a numpy
            array representing a matrix with as many lines as in a.
      Test: This function is is tested by the function test_gauss_multiple in tests/test_gauss_multiple.py.
    '''

    solution = []
    current_b=[]
    for i in b.T:
        current_b = []
        for j in i:
            current_b.append(j)
        solution.append(gauss_pivot(a.copy(),current_b))
    solution = transpose(solution)
    return solution
    raise Exception("Function not implemented")


def gauss_multiple_pivot(a, b):
    '''
      This function returns the same result as the previous one,
            except that it uses scaled row pivoting.
      Parameters: a is a numpy array representing a square matrix. b is a numpy
            array representing a matrix with as many lines as in a.
      Test: This function is is tested by the function 
            test_gauss_multiple_pivot in tests/test_gauss_multiple.py.
    '''

    solution = []
    current_b = []
    for i in b.T:
        current_b = []
        for j in i:
            current_b.append(j)
        solution.append(gauss_pivot(a.copy(), current_b))
    solution = transpose(solution)
    return solution
    raise Exception("Function not implemented")


def matrix_invert(m):
    '''
      This function returns the inverse of the square matrix a passed 
            as a paramter. 
      Parameters: a is a numpy array representing a non-singular square matrix.
      Test: This function is is tested by function test_inverse in tests/test_inverse.py
      Hint: Remember that the inverse of A is the solution of n linear systems of n 
            equations.
    '''

    n=shape(m)[0]
    b=identity(n)
    return gauss_multiple_pivot(m , b)



