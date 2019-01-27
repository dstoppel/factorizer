import numpy as np

def gaussElemination(matrix):

    assert len(np.shape(matrix)) == 2
    assert np.shape(matrix)[0] == np.shape(matrix)[1]

    dimension = np.shape(matrix)[0]

    u = np.array(matrix,copy=True)
    l = np.identity(dimension)

    for k in range(dimension-1):
        for i in range(k+1,dimension):
            l[i,k] = u[i,k]/u[k,k]
            u[i,k:dimension] = u[i,k:dimension]-l[i,k]*u[k,k:dimension]

    return l,u


def solve_ly(b, l):
    dimension = np.shape(b)[0]

    y = np.zeros(dimension)
    y[0] = b[0]/l[0,0]

    for i in range(1,dimension):
        y[i]=1/l[i,i]*(b[i]-np.sum(l[i,0:i]*y[0:i]))
    return y


def solve_ux(u,y):
    dimension = np.shape(y)[0]
    x = np.zeros(dimension)
    x[-1] = y[-1]/u[-1,-1]

    for i in range(dimension-2,-1,-1):
        x[i] = 1/u[i,i]*(y[i]-np.sum(u[i,(i+1):]*x[(i+1):]))
    return x


def lu_solver(matrix,b):
    l, u = gaussElemination(matrix)
    y = solve_ly(b, l)
    x = solve_ux(u, y)
    return x


def main():

    matrix = np.array([[2,1,1,0],[4,3,3,1],[8,7,9,5],[6,7,9,8]])
    b=np.array([1,1,0,0])

    l,u =gaussElemination(matrix)
    print('Matrix:')
    print(matrix)
    print('lower: \n', l)
    print('upper:\n', u)

    y = solve_ly(b,l)
    x =solve_ux(u,y)
    print('y: \n', y)
    print('x: \n',x)
    print('Test:\nMatrix*x: ')
    print(np.matmul(matrix,x))
    print('b: \n',b)


if __name__ == '__main__': main()