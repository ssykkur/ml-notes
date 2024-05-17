import numpy as np



def swap_rows(M, row_index_1, row_index_2):
    M = M.copy()
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M


#retrieve the index of the first non-zero value in a specified column
def get_non_zero_column(M, column, starting_row):
    column_array = M[starting_row: ,column]
    print(column_array)
    for i, val in enumerate(column_array):

        print(i, val)
        if not np.isclose(val, 0, atol=1e-5):
            index = i + starting_row
            return index
    return -1


#retrieve the first non-zero value index in a specified row 
def get_non_zero_row(M, row, augmented = False):

    M = M.copy()

    if augmented == True:
        M = M[:, :-1]
    
    row_array = M[row]
    for i, val in enumerate(row_array):
        if not np.isclose(val, 0, atol=1e-5):
            return i
    return -1

def augmented_matrix(A, B):
    augmented_M = np.hstack((A, B))
    return augmented_M


def row_echelon_form(A, B):

    det_A = np.linalg.det(A)

    if np.isclose(det_A, 0) == True:
        return 'Singular system'

    A = A.copy()
    B = B.copy()
    A = A.astype('float64')
    B = B.astype('float64')
    num_rows = len(A)

    M = augmented_matrix(A, B)

    for row in range(num_rows):
        pivot_candidate = M[row, row]
        
        if np.isclose(pivot_candidate, 0):
            non_zero_value = get_non_zero_column(M, row, row)
            swap_rows(M, row, non_zero_value)
            pivot = M[row, row]
        else:
            pivot = pivot_candidate
    
        M[row] = M[row] * pivot**-1

        for j in range(row + 1, num_rows):
            value_below_pivot = M[j, row]
            M[j] = M[j] - value_below_pivot * M[row]

    return M


def back_substitution(M):
    M = M.copy()
    num_rows = M.shape[0]

    for row in reversed(range(num_rows)):
        substituition_row = M[row]
        index = get_non_zero_row(M, row, augmented=True)
        for j in reversed(range(row)):
            row_to_reduce = M[j]
            value = row_to_reduce[index]

            row_to_reduce = row_to_reduce - substituition_row * value

            M[j,:] = row_to_reduce

    solution = M[:, -1]

    return M

def gaussian_elimination(A, B):

    row_echelon_form_M = row_echelon_form(A, B)

    if not isinstance(row_echelon_form_M, str):
        solution = back_substitution(row_echelon_form_M)

    return solution