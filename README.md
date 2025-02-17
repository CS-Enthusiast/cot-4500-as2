# Programming Assignment 2 - Interpolation Methods

This repository contains the solution for Programming Assignment 2 in COT 4500. In this assignment, we implement several interpolation methods in Python, including:

- **Neville's Method**
- **Newton's Forward Difference Method**
- **Hermite Interpolation using Divided Differences**
- **Cubic Spline Interpolation**

These methods are used to approximate functions and solve interpolation problems as covered in our course.

## Repository Structure

The repository is organized as follows:

cot-4500-as2/
├── src/
│   ├── main/
│   │   ├── __init__.py
│   │   └── assignment_2.py       # Main assignment implementation
│   └── test/
│       ├── __init__.py
│       └── test_assignment_2.py  # Unit tests for the assignment
├── requirements.txt              # Required Python libraries (NumPy)
└── README.md                     # This file

## Requirements

- **Python 3**
- **NumPy**

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## How to Compile and Run

### Running the Main Program
From the repository root, execute:

```bash
python3 src/main/assignment_2.py
```

This command will run the main assignment script and display outputs for:

- Neville's Method: Interpolated value for f(3.7) using a 2nd degree polynomial.
- Newton's Forward Difference Method: The forward difference table and polynomial approximations (degrees 1, 2, and 3) for f(7.3).
- Hermite Interpolation: The divided difference table for Hermite interpolation with both function and derivative values.
- Cubic Spline Interpolation: The coefficient matrix (A), vector (b), and the computed second derivatives for the interior nodes.

### Running the Unit Tests
To run the tests and verify the implementation, execute:

```bash
python3 -m unittest discover -s src/test
```

This command will run all tests defined in test_assignment_2.py and confirm that all functions are working as expected.



