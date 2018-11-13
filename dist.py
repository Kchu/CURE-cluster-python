import numpy
 
#输入的A,B为numpy.matrix格式
def dist(A, B):
    BT = B.transpose()
    vecProd = A * BT
    SqA =  A.getA()**2
    sumSqA = numpy.matrix(numpy.sum(SqA, axis=1))
    sumSqAEx = numpy.tile(sumSqA.transpose(), (1, vecProd.shape[1]))    
    SqB = B.getA()**2
    sumSqB = numpy.sum(SqB, axis=1)
    sumSqBEx = numpy.tile(sumSqB, (vecProd.shape[0], 1))    
    SqED = sumSqBEx + sumSqAEx - 2*vecProd   
    ED = (SqED.getA())**0.5
    return numpy.matrix(ED)