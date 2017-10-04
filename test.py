import numpy as np
import LinearRegression as lm
def checkAppendIntercept():
    np.random.seed(1)
    X = np.random.rand(10,5)
    calc_X = lm.appendIntercept(X)
    real_X = np.array([[  1.00000000e+00,   4.17022005e-01,   7.20324493e-01,
          1.14374817e-04,   3.02332573e-01,   1.46755891e-01],
       [  1.00000000e+00,   9.23385948e-02,   1.86260211e-01,
          3.45560727e-01,   3.96767474e-01,   5.38816734e-01],
       [  1.00000000e+00,   4.19194514e-01,   6.85219500e-01,
          2.04452250e-01,   8.78117436e-01,   2.73875932e-02],
       [  1.00000000e+00,   6.70467510e-01,   4.17304802e-01,
          5.58689828e-01,   1.40386939e-01,   1.98101489e-01],
       [  1.00000000e+00,   8.00744569e-01,   9.68261576e-01,
          3.13424178e-01,   6.92322616e-01,   8.76389152e-01],
       [  1.00000000e+00,   8.94606664e-01,   8.50442114e-02,
          3.90547832e-02,   1.69830420e-01,   8.78142503e-01],
       [  1.00000000e+00,   9.83468338e-02,   4.21107625e-01,
          9.57889530e-01,   5.33165285e-01,   6.91877114e-01],
       [  1.00000000e+00,   3.15515631e-01,   6.86500928e-01,
          8.34625672e-01,   1.82882773e-02,   7.50144315e-01],
       [  1.00000000e+00,   9.88861089e-01,   7.48165654e-01,
          2.80443992e-01,   7.89279328e-01,   1.03226007e-01],
       [  1.00000000e+00,   4.47893526e-01,   9.08595503e-01,
          2.93614148e-01,   2.87775339e-01,   1.30028572e-01]])
    if np.all(np.isclose(real_X,calc_X)):
        print("PASSED : appendIntercept Function")
    else:
        print("FAILED : appendIntercept Function")

def checkCostFunc():
    np.random.seed(2)
    m = 10
    y = np.random.rand(m,)
    y_predicted = np.random.rand(m,)
    calculatedCost = lm.costFunc(m,y,y_predicted)
    realCost = 0.075000505675425072
    if calculatedCost==realCost:
        print("PASSED : CostFunc Function")
    else:
        print("FAILED : CostFunc Function")

def checkCalcGradients():
    np.random.seed(3)
    m = 10
    x = np.random.rand(m,20)
    y = np.random.rand(m,)
    y_p = np.random.rand(m,)
    calcGrad  = lm.calcGradients(x,y,y_p,m)
    realGrad  = np.array([-0.05425541, -0.04381124, -0.05959325, -0.03675508, -0.01118115,
       -0.05390415, -0.09321702, -0.01038522, -0.00185729, -0.04773877,
       -0.03408592,  0.00746619,  0.00090633, -0.01870412, -0.00821488,
       -0.01664091, -0.11836125, -0.03610672, -0.08967235, -0.02161973])
    if np.all(np.isclose(calcGrad,realGrad)):
        print("PASSED : calcGradients Function")
    else:
        print("FAILED : calcGradients Function")

def checkMakeGradientUpdate():
    np.random.seed(4)
    theta = np.random.rand(20,)
    grads = np.random.rand(20,)
    calcUpdate = lm.makeGradientUpdate(theta,grads)
    realUpdate = np.array([ 0.96702984,  0.54723225,  0.97268436,  0.71481599,  0.69772882,
        0.2160895 ,  0.97627445,  0.00623026,  0.25298236,  0.43479153,
        0.77938292,  0.19768507,  0.86299324,  0.98340068,  0.16384224,
        0.59733394,  0.0089861 ,  0.38657128,  0.04416006,  0.95665297])

    if calcUpdate is not None and np.all(np.isclose(calcUpdate,realUpdate)):
        print("PASSED : makeGradientUpdate Function")
    else:
        print("FAILED : makeGradientUpdate Function")

def checkPredict():
    np.random.seed(6)
    theta  = np.random.rand(7,)
    X = np.random.rand(10,7)
    calcX = lm.predict(X,theta)
    realX = np.array([ 1.7089558 ,  2.20884418,  2.18216447,  1.80692415,  2.12231727,
        1.41312956,  1.82242337,  2.11752865,  1.70792641,  0.8332109 ])
    if np.all(np.isclose(calcX,realX)):
        print("PASSED : predict Function")
    else:
        print("FAILED : predict Function")

def checkTrain():
    np.random.seed(5)
    theta  = np.random.rand(5,)
    X  = np.random.rand(10,5)
    y = np.random.rand(10,)
    model  = {}
    calcModel  = lm.train(theta,X,y,model)
    calcModel['J']=calcModel['J'][:50]
    realModel = {}
    realModel['J']=[0.23849093475226227, 0.23849093474760394, 0.23849093474294566,
                    0.23849093473828731, 0.23849093473362895, 0.23849093472897059,
                    0.23849093472431226, 0.23849093471965394, 0.23849093471499558,
                    0.23849093471033728, 0.23849093470567886, 0.23849093470102059,
                    0.23849093469636218, 0.23849093469170385, 0.23849093468704555,
                    0.23849093468238722, 0.23849093467772886, 0.2384909346730705,
                    0.23849093466841217, 0.23849093466375382, 0.23849093465909549,
                    0.23849093465443719, 0.23849093464977877, 0.23849093464512042,
                    0.23849093464046217, 0.23849093463580379, 0.2384909346311454,
                    0.2384909346264871, 0.23849093462182877, 0.23849093461717041,
                    0.23849093461251208, 0.23849093460785373, 0.23849093460319537,
                    0.2384909345985371, 0.23849093459387868, 0.23849093458922033,
                    0.23849093458456197, 0.23849093457990361, 0.23849093457524534,
                    0.23849093457058701, 0.23849093456592868, 0.23849093456127032,
                    0.23849093455661202, 0.23849093455195361, 0.23849093454729525,
                    0.23849093454263687, 0.23849093453797865, 0.23849093453332024,
                    0.23849093452866194, 0.23849093452400352]

    realModel['theta']=[0.22199316135627545, 0.87073228953402304, 0.20671913831457267,
                        0.91861088834692606, 0.48841117787717347]
    
    if realModel== calcModel:
        print("PASSED : test Function")
    else:
        print("FAILED : test Function")

if __name__ == "__main__":
    checkAppendIntercept()
    checkCostFunc()
    checkCalcGradients()
    checkMakeGradientUpdate()
    checkPredict()
    checkTrain()
