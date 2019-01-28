import numpy as np

def cholesky_factorization(matrix):
    dimension = np.shape(matrix)[0]
    l = np.array(matrix, copy=True)
    for i in range(dimension-1):
        for j in range(i+1,dimension):
            l[i,j] = 0

    for i in range(dimension):
        for j in range(i+1,dimension):
            l[j:, j]=l[j:,j]-l[j:,i]*l[j,i]/l[i,i]
        l[i:,i]=l[i:,i]/np.sqrt(l[i,i])
    return l


def main():
    matrix = np.array([[9,3,-6,12], [3,26,-7,-11], [-6,-7,9,7], [12,-11,7,65]])
    l=cholesky_factorization(matrix)

    print('Choleskyfactor:\n',l)
    print('Test:\n', np.matmul(l,l.transpose()))
    print('Compare:\n', matrix)


if __name__ == '__main__': main()