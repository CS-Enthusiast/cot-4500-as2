# src/main/assignment_2.py

import math
import numpy as np

# ----------------------------
# Question 1: Neville's Method
# ----------------------------
def neville(x, y, x0):
    """
    Use Neville's method to compute the interpolation at x0.
    :param x: list of x data points
    :param y: list of corresponding y values
    :param x0: value at which to interpolate
    :return: interpolated value f(x0)
    """
    n = len(x)
    # Create an n x n table
    Q = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Initialize the first column with y values.
    for i in range(n):
        Q[i][0] = y[i]
    
    # Build the table column by column.
    for j in range(1, n):
        for i in range(n - j):
            Q[i][j] = ((x0 - x[i+j]) * Q[i][j-1] + (x[i] - x0) * Q[i+1][j-1]) / (x[i] - x[i+j])
    
    return Q[0][n-1]

def question1():
    # Given data for Question 1
    x_vals = [3.6, 3.8, 3.9]
    y_vals = [1.675, 1.436, 1.318]
    x0 = 3.7
    result = neville(x_vals, y_vals, x0)
    print("Question 1: Neville's Method")
    print("Interpolated value f(3.7) =", result)
    print("-" * 50)

# ----------------------------
# Question 2 & 3: Newton's Forward Difference Method
# ----------------------------
def forward_difference_table(x, y):
    """
    Compute the forward difference table.
    :param x: list of x values (assumed equally spaced)
    :param y: list of corresponding y values
    :return: a list of lists representing the forward differences
    """
    n = len(x)
    table = [y.copy()]
    for j in range(1, n):
        diff = []
        for i in range(n - j):
            diff.append(table[j-1][i+1] - table[j-1][i])
        table.append(diff)
    return table

def newton_forward_poly(x_eval, table, x0, degree, h):
    """
    Evaluate Newton's forward difference polynomial at x_eval.
    :param x_eval: point to evaluate
    :param table: forward difference table
    :param x0: starting x value
    :param degree: degree of polynomial (1, 2, or 3)
    :param h: spacing between x values
    :return: approximate f(x_eval)
    """
    p = (x_eval - x0) / h
    result = table[0][0]
    term = 1
    for j in range(1, degree + 1):
        term *= (p - (j - 1))
        result += (term / math.factorial(j)) * table[j][0]
    return result

def question2_and_3():
    # Data for Newton's Forward Difference Method
    x_data = [7.2, 7.4, 7.5, 7.6]
    y_data = [23.5492, 25.3913, 26.8224, 27.4589]
    
    fd_table = forward_difference_table(x_data, y_data)
    h = x_data[1] - x_data[0]  # spacing
    
    print("Question 2: Newton's Forward Difference Table")
    for i, col in enumerate(fd_table):
        print(f"Δ^{i}: {col}")
    
    x_eval = 7.3
    print("\nQuestion 3: Newton's Forward Polynomial Approximations for f(7.3)")
    for deg in [1, 2, 3]:
        approx = newton_forward_poly(x_eval, fd_table, x_data[0], deg, h)
        print(f"Degree {deg} approximation at x = {x_eval}: {approx}")
    print("-" * 50)

# ----------------------------
# Question 4: Hermite Interpolation Using Divided Differences
# ----------------------------
def hermite_divided_difference(x, y, dy):
    """
    Build the Hermite divided difference table.
    :param x: list of x values
    :param y: list of f(x) values
    :param dy: list of f'(x) values
    :return: tuple (z, Q) where z is the extended x list and Q is the divided difference table
    """
    n = len(x)
    m = 2 * n  # each x value appears twice
    z = []
    Q = [[0.0 for _ in range(m)] for _ in range(m)]
    
    for i in range(n):
        z.append(x[i])
        z.append(x[i])
        Q[2*i][0] = y[i]
        Q[2*i+1][0] = y[i]
        # For the repeated node, first divided difference is the derivative
        Q[2*i+1][1] = dy[i]
        if i != 0:
            Q[2*i][1] = (Q[2*i][0] - Q[2*i-1][0]) / (z[2*i] - z[2*i-1])
    
    # Fill in the remaining divided differences
    for i in range(2, m):
        for j in range(2, i+1):
            Q[i][j] = (Q[i][j-1] - Q[i-1][j-1]) / (z[i] - z[i-j])
    
    return z, Q

def question4():
    # Data for Hermite interpolation
    x_vals = [3.6, 3.8, 3.9]
    y_vals = [1.675, 1.436, 1.318]
    dy_vals = [-1.195, -1.188, -1.182]
    
    z, hermite_table = hermite_divided_difference(x_vals, y_vals, dy_vals)
    print("Question 4: Hermite Divided Difference Table")
    for i in range(len(z)):
        row = [f"{hermite_table[i][j]:.5f}" for j in range(i+1)]
        print(f"z[{i}] = {z[i]:.3f} : " + ", ".join(row))
    print("-" * 50)

# ----------------------------
# Question 5: Cubic Spline Interpolation
# ----------------------------
def cubic_spline_system(x, y):
    """
    Build the matrix system for natural cubic spline interpolation for interior nodes.
    :param x: list of x values
    :param y: list of corresponding f(x) values
    :return: tuple (A, b) where A is the coefficient matrix and b is the right-hand side vector
    """
    n = len(x)
    h = [x[i+1] - x[i] for i in range(n-1)]
    A = np.zeros((n-2, n-2))
    b = np.zeros(n-2)
    
    # Form equations for interior nodes (i = 1, 2, ..., n-2)
    for i in range(1, n-1):
        A[i-1, i-1] = (h[i-1] + h[i]) / 3.0  # diagonal element
        
        if i - 1 > 0:
            A[i-1, i-2] = h[i-1] / 6.0  # lower diagonal
        if i < n - 2:
            A[i-1, i] = h[i] / 6.0      # upper diagonal
        
        b[i-1] = (y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1]
    
    return A, b

def question5():
    # Data for cubic spline interpolation
    x_points = [2, 5, 8, 10]
    y_points = [3, 5, 7, 9]
    
    A, b = cubic_spline_system(x_points, y_points)
    print("Question 5: Cubic Spline Interpolation")
    print("Matrix A:")
    print(A)
    print("\nVector b:")
    print(b)
    
    # Solve for interior second derivatives (M values) at nodes x_points[1] and x_points[2]
    M_interior = np.linalg.solve(A, b)
    print("\nVector M (second derivatives at interior nodes):")
    print(M_interior)
    print("-" * 50)

# ----------------------------
# Main execution: run all questions
# ----------------------------
def main():
    question1()
    question2_and_3()
    question4()
    question5()

if __name__ == "__main__":
    main()
# src/main/assignment_2.py

import math
import numpy as np

# ----------------------------
# Question 1: Neville's Method
# ----------------------------
def neville(x, y, x0):
    """
    Use Neville's method to compute the interpolation at x0.
    :param x: list of x data points
    :param y: list of corresponding y values
    :param x0: value at which to interpolate
    :return: interpolated value f(x0)
    """
    n = len(x)
    # Create an n x n table
    Q = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Initialize the first column with y values.
    for i in range(n):
        Q[i][0] = y[i]
    
    # Build the table column by column.
    for j in range(1, n):
        for i in range(n - j):
            Q[i][j] = ((x0 - x[i+j]) * Q[i][j-1] + (x[i] - x0) * Q[i+1][j-1]) / (x[i] - x[i+j])
    
    return Q[0][n-1]

def question1():
    # Given data for Question 1
    x_vals = [3.6, 3.8, 3.9]
    y_vals = [1.675, 1.436, 1.318]
    x0 = 3.7
    result = neville(x_vals, y_vals, x0)
    print("Question 1: Neville's Method")
    print("Interpolated value f(3.7) =", result)
    print("-" * 50)

# ----------------------------
# Question 2 & 3: Newton's Forward Difference Method
# ----------------------------
def forward_difference_table(x, y):
    """
    Compute the forward difference table.
    :param x: list of x values (assumed equally spaced)
    :param y: list of corresponding y values
    :return: a list of lists representing the forward differences
    """
    n = len(x)
    table = [y.copy()]
    for j in range(1, n):
        diff = []
        for i in range(n - j):
            diff.append(table[j-1][i+1] - table[j-1][i])
        table.append(diff)
    return table

def newton_forward_poly(x_eval, table, x0, degree, h):
    """
    Evaluate Newton's forward difference polynomial at x_eval.
    :param x_eval: point to evaluate
    :param table: forward difference table
    :param x0: starting x value
    :param degree: degree of polynomial (1, 2, or 3)
    :param h: spacing between x values
    :return: approximate f(x_eval)
    """
    p = (x_eval - x0) / h
    result = table[0][0]
    term = 1
    for j in range(1, degree + 1):
        term *= (p - (j - 1))
        result += (term / math.factorial(j)) * table[j][0]
    return result

def question2_and_3():
    # Data for Newton's Forward Difference Method
    x_data = [7.2, 7.4, 7.5, 7.6]
    y_data = [23.5492, 25.3913, 26.8224, 27.4589]
    
    fd_table = forward_difference_table(x_data, y_data)
    h = x_data[1] - x_data[0]  # spacing
    
    print("Question 2: Newton's Forward Difference Table")
    for i, col in enumerate(fd_table):
        print(f"Δ^{i}: {col}")
    
    x_eval = 7.3
    print("\nQuestion 3: Newton's Forward Polynomial Approximations for f(7.3)")
    for deg in [1, 2, 3]:
        approx = newton_forward_poly(x_eval, fd_table, x_data[0], deg, h)
        print(f"Degree {deg} approximation at x = {x_eval}: {approx}")
    print("-" * 50)

# ----------------------------
# Question 4: Hermite Interpolation Using Divided Differences
# ----------------------------
def hermite_divided_difference(x, y, dy):
    """
    Build the Hermite divided difference table.
    :param x: list of x values
    :param y: list of f(x) values
    :param dy: list of f'(x) values
    :return: tuple (z, Q) where z is the extended x list and Q is the divided difference table
    """
    n = len(x)
    m = 2 * n  # each x value appears twice
    z = []
    Q = [[0.0 for _ in range(m)] for _ in range(m)]
    
    for i in range(n):
        z.append(x[i])
        z.append(x[i])
        Q[2*i][0] = y[i]
        Q[2*i+1][0] = y[i]
        # For the repeated node, first divided difference is the derivative
        Q[2*i+1][1] = dy[i]
        if i != 0:
            Q[2*i][1] = (Q[2*i][0] - Q[2*i-1][0]) / (z[2*i] - z[2*i-1])
    
    # Fill in the remaining divided differences
    for i in range(2, m):
        for j in range(2, i+1):
            Q[i][j] = (Q[i][j-1] - Q[i-1][j-1]) / (z[i] - z[i-j])
    
    return z, Q

def question4():
    # Data for Hermite interpolation
    x_vals = [3.6, 3.8, 3.9]
    y_vals = [1.675, 1.436, 1.318]
    dy_vals = [-1.195, -1.188, -1.182]
    
    z, hermite_table = hermite_divided_difference(x_vals, y_vals, dy_vals)
    print("Question 4: Hermite Divided Difference Table")
    for i in range(len(z)):
        row = [f"{hermite_table[i][j]:.5f}" for j in range(i+1)]
        print(f"z[{i}] = {z[i]:.3f} : " + ", ".join(row))
    print("-" * 50)

# ----------------------------
# Question 5: Cubic Spline Interpolation
# ----------------------------
def cubic_spline_system(x, y):
    """
    Build the matrix system for natural cubic spline interpolation for interior nodes.
    :param x: list of x values
    :param y: list of corresponding f(x) values
    :return: tuple (A, b) where A is the coefficient matrix and b is the right-hand side vector
    """
    n = len(x)
    h = [x[i+1] - x[i] for i in range(n-1)]
    A = np.zeros((n-2, n-2))
    b = np.zeros(n-2)
    
    # Form equations for interior nodes (i = 1, 2, ..., n-2)
    for i in range(1, n-1):
        A[i-1, i-1] = (h[i-1] + h[i]) / 3.0  # diagonal element
        
        if i - 1 > 0:
            A[i-1, i-2] = h[i-1] / 6.0  # lower diagonal
        if i < n - 2:
            A[i-1, i] = h[i] / 6.0      # upper diagonal
        
        b[i-1] = (y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1]
    
    return A, b

def question5():
    # Data for cubic spline interpolation
    x_points = [2, 5, 8, 10]
    y_points = [3, 5, 7, 9]
    
    A, b = cubic_spline_system(x_points, y_points)
    print("Question 5: Cubic Spline Interpolation")
    print("Matrix A:")
    print(A)
    print("\nVector b:")
    print(b)
    
    # Solve for interior second derivatives (M values) at nodes x_points[1] and x_points[2]
    M_interior = np.linalg.solve(A, b)
    print("\nVector M (second derivatives at interior nodes):")
    print(M_interior)
    print("-" * 50)

# ----------------------------
# Main execution: run all questions
# ----------------------------
def main():
    question1()
    question2_and_3()
    question4()
    question5()

if __name__ == "__main__":
    main()
