#1. import useful python packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.pyplot import plot 
from matplotlib.pyplot import imshow

#import os
#2. python system 
import warnings
warnings.filterwarnings(action='ignore')


#3. rpy2 
import rpy2
import rpy2.robjects as ro
#ro.r('library(devtools)') ## to use source_url 
#ro.r('library(tidyverse)')

############################## rpy2 functions ##############################
def p2r(A):
    from rpy2.robjects.vectors import FloatVector 
    from rpy2.robjects.vectors import StrVector as s2r_temp

    def a2r_temp(a):
        if type(a) in {float,int,bool}: a=[a]
        a=list(a)
        rtn=FloatVector(a)
        return rtn

    def m2r_temp(A):
        Acopy=A.T.copy()
        nrow=Acopy.shape[0]
        Acopy.shape=(np.prod(Acopy.shape),1)
        rtn=ro.r.matrix(a2r_temp(m2a(Acopy)),ncol=nrow)
        del(Acopy)
        ro.globalenv['A']=rtn
        return rtn

    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    def pd2r_temp(A):
        with localconverter(ro.default_converter + pandas2ri.converter):
            rtn = ro.conversion.py2rpy(A)
        return rtn
    
    if type(A)==type(initpd('0',n=2,p=2)):
        rtn=pd2r_temp(A) 
    elif type(A)==type(init('0',(2,2))):
        rtn=m2r_temp(A)
    elif type(A)==str: #  elif type(pd.DataFrame(np.matrix(A)).iloc[0,0])==str: 와 순서바꾸면 안됨
        rtn=s2r_temp(A)        
    elif type(pd.DataFrame(np.matrix(A)).iloc[0,0])==str:
        rtn=s2r_temp(pd.DataFrame(np.matrix(A)).T.iloc[:,0])
    else:
        rtn=a2r_temp(A)
    return rtn 

def push(py,rname=None):
    import inspect
    def retrieve_name(var):
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]
    if rname==None: rname = retrieve_name(py)
    ro.globalenv[rname]=p2r(py)

def r2p(A):
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    def r2a_temp(a):
        return list(a)
    
    def r2m_temp(A):
        return np.matrix(A)
    
    def r2pd_temp(A):
        with localconverter(ro.default_converter + pandas2ri.converter):
            rtn = ro.conversion.rpy2py(A)
        return rtn        
    
    ro.globalenv['temp']=A
    if ro.r('is.null(dim(temp))')[0]==False: ## in the cases of matrix or dataframe
        if ro.r('is.data.frame(temp)')[0]: 
            rtn=r2pd_temp(A)
        elif ro.r('is.matrix(temp)')[0]:
            rtn=r2m_temp(A)
        else:
            print('I don\`t know which type of this data in R.')
    else:
        rtn=r2a_temp(A)
    ro.r('rm("temp")')
    return rtn

def pull(r):
    return r2p(ro.globalenv[r])



############################## python functions ##############################
def cc(start=0,end=1,samplingFreq=1):
    rtn=m2a(cbind(np.arange(start,end,1/samplingFreq),end).T)
    if start==int(start) and end==int(end) and samplingFreq==1: rtn=np.array(list(map(int,rtn)))
    return rtn

def co(start=0,end=1,samplingFreq=1):
    rtn=m2a(np.asmatrix(np.arange(start,end,1/samplingFreq)).T)
    if start==int(start) and end==int(end) and samplingFreq==1: rtn=np.array(list(map(int,rtn)))
    return rtn

def oc(start=0,end=1,samplingFreq=1):
    rtn=m2a(np.asmatrix(np.arange(start,end,1/samplingFreq)).T+1/samplingFreq)
    if start==int(start) and end==int(end) and samplingFreq==1: rtn=np.array(list(map(int,rtn)))
    return rtn 

def oo(start=0,end=1,samplingFreq=1):
    rtn=np.asmatrix(np.arange(start,end,1/samplingFreq)).T
    rtn=m2a(rtn[1:len(rtn)])
    if start==int(start) and end==int(end) and samplingFreq==1: rtn=np.array(list(map(int,rtn)))
    return rtn 

### 입력은 n×p matrix , 혹은 n×p DataFrame 으로 바꿀 수 있는 어떤자료 / 출력 n×p np.matrix 임. 이때 첫번째 row가 0임. 
def lagg(inputMatrix,lag):
    inputdf=pd.DataFrame(inputMatrix)
    rtn=np.asmatrix(inputdf.shift(lag))
    rtn[range(lag),:]=0
    return rtn

def cbind(*Mat):
    lenofMat=len(Mat)
    if lenofMat==1: 
        print("You must enter two or more input objects.")
        rtn=Mat[0]
    elif lenofMat==2: 
        rtn=cbindtemp(Mat[0],Mat[1])
    else: 
        rtn=cbindtemp(Mat[0],Mat[1])
        for i in co(2,lenofMat):
            rtn=cbindtemp(rtn,Mat[i])
    return rtn 

def rbind(*Mat):
    lenofMat=len(Mat)
    if lenofMat==1: 
        print("You must enter two or more input objects.")
        rtn=Mat[0]
    elif lenofMat==2: 
        rtn=rbindtemp(Mat[0],Mat[1])
    else: 
        rtn=rbindtemp(Mat[0],Mat[1])
        for i in co(2,lenofMat):
            rtn=rbindtemp(rtn,Mat[i])
    return rtn

def cbindtemp(A,B):
    typ=['matrix','matrix']
    
    if isinstance(A, pd.core.series.Series): 
        A=a2c(A)
    if isinstance(B, pd.core.series.Series): 
        B=a2c(B)
    A=np.asmatrix(A)
    B=np.asmatrix(B)

    # row-vector에 대한 처리 
    if A.shape[0]==1: typ[0]='rowvec'
    if B.shape[0]==1: typ[1]='rowvec'

    # col-vector에 대한 처리 
    if A.shape[1]==1: typ[0]='colvec'
    if B.shape[1]==1: typ[1]='colvec'    
    
    # 스칼라에 대한 처리 
    if A.shape==(1,1): typ[0]='scala'
    if B.shape==(1,1): typ[1]='scala'
        
    if typ==['scala','scala']:  A=np.array(A); B=np.array(B)
    if typ==['scala','rowvec']: A=np.array(A); 
    if typ==['scala','colvec']: A=np.full(B.shape,A[0,0]); 
    if typ==['scala','matrix']: A=np.full((B.shape[0],1),A[0,0]); 
    	
    if typ==['rowvec','scala']: B=np.array(B)
    #if typ==['rowvec','rowvec']:
    if typ==['rowvec','colvec']: A=A.T
    if typ==['rowvec','matrix']: A=A.T
        
    if typ==['colvec','scala']:  B=np.full(A.shape,B[0,0])
    if typ==['colvec','rowvec']: B=B.T
    #if typ==['colvec','colvec']: 
    #if typ==['colvec','matrix']: 
    
    if typ==['matrix','scala']:  B=np.full((A.shape[0],1),B[0,0])
    if typ==['matrix','rowvec']: B=B.T
    #if typ==['matrix','colvec']: 
    #if typ==['matrix','matrix']:
    
    return np.hstack([A,B])
    
def rbindtemp(A,B):
    typ=['matrix','matrix']
    
    A=np.asmatrix(A)
    B=np.asmatrix(B)

    # row-vector에 대한 처리 
    if A.shape[0]==1: typ[0]='rowvec'
    if B.shape[0]==1: typ[1]='rowvec'

    # col-vector에 대한 처리 
    if A.shape[1]==1: typ[0]='colvec'
    if B.shape[1]==1: typ[1]='colvec'    
    
    # 스칼라에 대한 처리 
    if A.shape==(1,1): typ[0]='scala'
    if B.shape==(1,1): typ[1]='scala'
        
    if typ==['scala','scala']:  A=np.array(A); B=np.array(B)
    if typ==['scala','rowvec']: A=np.full(B.shape,A[0,0]); 
    if typ==['scala','colvec']: A=np.array(A);
    if typ==['scala','matrix']: A=np.full((1,B.shape[1]),A[0,0]); 
    	
    if typ==['rowvec','scala']: B=np.full((1,A.shape[1]),B[0,0]); 
    #if typ==['rowvec','rowvec']:
    if typ==['rowvec','colvec']: B=B.T
    #if typ==['rowvec','matrix']: 
        
    #if typ==['colvec','scala']:  
    if typ==['colvec','rowvec']: A=A.T
    #if typ==['colvec','colvec']: 
    if typ==['colvec','matrix']: A=A.T
    
    if typ==['matrix','scala']:  B=np.full((1,A.shape[1]),B[0,0])
    #if typ==['matrix','rowvec']: 
    if typ==['matrix','colvec']: B=B.T
    #if typ==['matrix','matrix']:
    
    return np.vstack([A,B])

def ifo(A):
    print("type of data  : ",type(A))
    try: len(A)
    except TypeError as e: print("len of data   : ",e)
    else: print("len of data   : ",len(A))
        
    try: A.shape
    except AttributeError as e : print("shape of data : ",e)
    else: print("shape of data : ",A.shape)

### 배열의 차원을 출력해주는 함수 
def dim(A):
    if type(A) == str: rtn=0
    else:
        try: A.shape
        except AttributeError: 
            try: len(A)
            except TypeError: rtn=0
            else: rtn=1
        else: rtn=len(list(A.shape))
    return rtn

### (축소) n×1 이거나 1×n matrix 혹은 pd를 길이가 n인 np.array로 변환 
def m2a(A):
    if dim(A)==2: 
        if A.shape[0]==1: rtn=np.array(A)[0,:]
        elif A.shape[1]==1: rtn=np.array(A.T)[0,:]
        else: 
            print("We can't convert this type of data since the shape of input is \""+str(A.shape)+"\". So we will not do any conversion.")
            rtn=A
    else :
        print("The dimension of input matrix should be 2. But the dimension of your input is \""+str(dim(A))+"\". So we will not do any conversion.")
        rtn=A
    return rtn

## (확장) list, np.array, pd.series와 같은 1차원배열을 n×1 matrix, 즉 n×1 col-vector 로 변환 
def a2c(a):
    if dim(a)==1: 
        rtn=np.matrix(a).T
    else : 
        print("The dimension of input matrix should be 1. But the dimension of your input is \""+str(dim(a))+"\". So we will not do any conversion.")
        rtn=a
    return rtn

## (확장) a2c와 똑같은 함수: 1차원배열 -> n×1 col-vector 
def a2m(a):
    if dim(a)==1: 
        rtn=np.matrix(a).T
    else : 
        print("The dimension of input matrix should be 1. But the dimension of your input is \""+str(dim(a))+"\". So we will not do any conversion.")
        rtn=a
    return rtn

## (축소) 1차원배열 -> 0차원배열 
def a2s(a): 
    if dim(a)==1: 
        if len(a)==1: rtn=a[0]
        else: 
            print("We can't convert this type of data since the lenth of input is \""+str(len(a))+"\". So we will not do any conversion.")
            rtn=a
    else : 
        print("The dimension of input should be 1. But the dimension of your input is \""+str(dim(a))+"\". So we will not do any conversion.")
        rtn=a
    return rtn

## (확장) 0차원배열 -> 1차원배열 
def s2a(a): 
    if dim(a)==0: 
        rtn=np.array([a])
    else : 
        print("The dimension of input matrix should be 0. But the dimension of your input is \""+str(dim(a))+"\". So we will not do any conversion.")
        rtn=a
    return rtn

## (축소) 2차원배열 -> 0차원배열 
def m2s(A): 
    if dim(A)==2: 
        if (A.shape[0]==1 & A.shape[1]==1): rtn=np.matrix(A)[0,0]
        else: 
            print("We can't convert this type of data since the shape of input is \""+str(A.shape)+"\". So we will not do any conversion.")
            rtn=A
    else :
        print("The dimension of input matrix should be 2. But the dimension of your input is \""+str(dim(A))+"\". So we will not do any conversion.")
        rtn=A
    return rtn

## (확장) 0차원배열 -> 2차원배열 
def s2m(a): # a2c와 똑같은 함수임. 
    if dim(a)==0: 
        rtn=np.matrix(a)
    else : 
        print("The dimension of input matrix should be 0. But the dimension of your input is \""+str(dim(a))+"\". So we will not do any conversion.")
        rtn=a
    return rtn    

def sprod(*index): 
    nofMindex=len(index)
    indexListtype=list(range(0,nofMindex))
    for i in range(0,nofMindex):
        if type(index[i]) is str: indexListtype[i]=[index[i]]
        elif np.asmatrix(index[i]).shape==(1,1): indexListtype[i]=[str(np.asmatrix(index[i])[0,0])]
        else: indexListtype[i]=index[i]
        
    mindex=pd.MultiIndex.from_product(indexListtype)
    val=init("0",mindex.shape[0])
    data=pd.DataFrame(val,index=mindex).reset_index()
    mindextable=data.iloc[:,0:data.shape[1]-1]
    rtn=['']*mindextable.shape[0]
    for i in range(0,mindextable.shape[0]):
        for j in range(0,mindextable.shape[1]):
            rtn[i]=rtn[i]+str(mindextable.iloc[i,j])
        
    return rtn

def ids(pddata):
    push(pddata.columns,"vname")
    print(r2p(ro.r("str_c(str_c('(',str_c(1:length(vname)-1),') ',vname),collapse='\n')"))[0])
