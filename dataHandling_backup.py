import numpy as np
import pandas as pd

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

def info(A):
    print("type of data  : ",type(A))
    try: len(A)
    except TypeError as e: print("len of data   : ",e)
    else: print("len of data   : ",len(A))
        
    try: A.shape
    except AttributeError as e : print("shape of data : ",e)
    else: print("shape of data : ",A.shape)


### 배열의 차원을 출력해주는 함수 
def dim(A):
    try: A.shape
    except AttributeError: 
        try: len(A)
        except TypeError: rtn=0
        else: rtn=1
    else: rtn=len(list(A.shape))
    return rtn

def dim(A):
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

### 초기화 (1) 0 (2) 유니폼 (3) 단위행렬 (4) 정규분포
def init(typ,dim):
    if dim*0==0: # dim 이 1차원일 경우 
        if typ=="0": rtn=np.zeros(dim)
        elif typ=="u": rtn=np.random.random(dim)
        elif typ=="I": 
            print("In the case of vectors, you cannot define a identity matrix.")
            rtn=np.ones(dim)             
        elif typ=="n": rtn=np.random.normal(0,1,dim)
        
    else: # dim 이 2차원일 경우 
        if typ=="0": rtn=np.asmatrix(np.zeros(dim))
        elif typ=="u": rtn=np.asmatrix(np.random.random(dim))
        elif typ=="I": 
            if dim[0]==dim[1]: rtn=np.eye(dim[0])
            elif dim[0]>dim[1]: rtn=rbind(np.eye(dim[1]),np.zeros((dim[0]-dim[1],dim[1])))
            elif dim[0]<dim[1]: rtn=cbind(np.eye(dim[0]),np.zeros((dim[0],dim[1]-dim[0])))
        elif typ=="n": rtn=np.asmatrix(np.random.normal(0,1,dim))
    return rtn

### elementwise 연산 
# 입력: 스칼라, 컬럼벡터, 로우벡터, 매트릭스
# 출력: 스칼라, 컬럼벡터, 로우벡터, 매트릭스 
# 연산종류: scala2scala
# ** 주의사항 **
# - 데이터프레임형태의 입력은 받지 않도록 한다. 데이터프레임형태의 입려은 판다스에 내장된 .transform 메소드를 사용하는것이 더 나음. 
# - 왜냐하면 데이터프레임->매트릭스로의 변환이 자유롭지 않기 때문에 이로 인한 오류가 발생할 수 있음. 

def transform(X,fun,plt=False):
    Xmat=np.asmatrix(X)
    # identifying dim of input X
    if Xmat.shape[0]==1 and Xmat.shape[1]==1: Xtype='scala'
    if Xmat.shape[0]==1 and Xmat.shape[1]>1: Xtype='rowvec'
    if Xmat.shape[0]>1 and Xmat.shape[1]==1: Xtype='colvec'
    if Xmat.shape[0]>1 and Xmat.shape[1]>1: Xtype='matrix'
    
    Xpd=pd.DataFrame(Xmat)
    if Xtype=='scala':
        print('Scala is not appropriate as an input of this function. Thus no operation is performed, therefore the output value is the same as the input value. ')
        rtn=X
        if plt==True: disp=pd.concat([Xpd,Xpd],keys=['input','result'])
    if Xtype=='rowvec':
        rtn=eval('Xpd.iloc[0,:].transform('+fun+')')
        if plt==True: disp=pd.concat([Xpd,pd.DataFrame(rtn).T],keys=['input','result'],axis=0)
        rtn=np.asmatrix(rtn)
    if Xtype=='colvec':
        rtn=eval('Xpd.iloc[:,0].transform('+fun+')')
        if plt==True: disp=pd.concat([Xpd,rtn],keys=['input','result'],axis=1)
        rtn=np.asmatrix(rtn).T
    if Xtype=='matrix':
        rtn=eval('Xpd.iloc[:,:].transform('+fun+')')
        if plt==True: disp=pd.concat([Xpd,rtn],keys=['input','result'],axis=0)
        rtn=np.asmatrix(rtn)
    if plt==True:
        from IPython.display import display 
        display(disp)
    return rtn 

    
### 행별 or 열별 연산 
# 입력: 매트릭스
# 출력: 컬럼벡터, 로우벡터, 매트릭스 
# 연산종류: array2array, array2scala. 
# axis종류: column wise, row wise 
# ** 주의사항 
# - 데이터프레임형태의 입력은 받지 않도록 한다. 이 경우는 판다스에 내장된 .apply 메소드를 사용하는것이 더 나음. 
# - 특히 데이터프레임->매트릭스로의 변환이 자유롭지 않아 여기에서 에러가 발생할 수 있다. 
# - 입력의 형태는 매트릭스로 고정한다. 벡터나 스칼라 입력은 받지않는다. 이 경우 apply 를 쓸 이유 자체가 없음. 
def apply(X,fun,axis=0,plt=False):     # axis=0: column-wise / axis=1: row-wise. 
    Xmat=np.asmatrix(X)
    # identifying dim of input X
    if Xmat.shape[0]==1 and Xmat.shape[1]==1: Xtype='scala'
    if Xmat.shape[0]==1 and Xmat.shape[1]>1: Xtype='rowvec'
    if Xmat.shape[0]>1 and Xmat.shape[1]==1: Xtype='colvec'
    if Xmat.shape[0]>1 and Xmat.shape[1]>1: Xtype='matrix'
    
    # identifying range of fun. 
    # domain of fun is always vector. 
    test=[1,2]
    resultoftest=np.asmatrix(eval(fun+'(test)'))
    if resultoftest.shape[0]==1 and resultoftest.shape[1]==1: funtype='array2scala'
    if resultoftest.shape[0]==1 and resultoftest.shape[1]>1: funtype='array2array'
    #if resultoftest.shape[0]>1 and resultoftest.shape[1]==1: funtype='array2array'
    if resultoftest.shape[0]>1 and resultoftest.shape[1]>1: funtype='array2matrix'
    
    # axis=0: column-wise
    if axis==0 and Xtype=='matrix' and funtype=='array2scala':
        disp=pd.DataFrame(Xmat)
        rtn=eval('disp.apply('+fun+')')
        disp=disp.T
        disp['result']=rtn
        rtn=np.asmatrix(rtn)
        disp=disp.T
    elif axis==0 and Xtype=='matrix' and funtype=='array2array':
        rtn=eval('np.asmatrix(pd.DataFrame(Xmat).apply('+fun+'))')
        disp=pd.concat([pd.DataFrame(Xmat),pd.DataFrame(rtn)],keys=['input','result'])
    elif axis==0 and Xtype=='matrix' and funtype=='array2matrix':
        print('The "array2matrix" type operator is not supported. Since no operation is performed, the output value is the same as the input value.')
        rtn=Xmat
        disp=pd.concat([pd.DataFrame(Xmat),pd.DataFrame(rtn)],keys=['input','result'])
        
    # axis=1: row-wise
    elif axis==1 and Xtype=='matrix' and funtype=='array2scala':
        disp=pd.DataFrame(Xmat)
        disp['result']=eval('disp.apply('+fun+',axis=1)')
        rtn=np.asmatrix(disp['result']).T
    elif axis==1 and Xtype=='matrix' and funtype=='array2array':
        rtn=eval('np.asmatrix(pd.DataFrame(Xmat).T.apply('+fun+')).T')
        disp=pd.concat([pd.DataFrame(Xmat),pd.DataFrame(rtn)],keys=['input','result'])
    elif axis==1 and Xtype=='matrix' and funtype=='array2matrix':
        print('The "array2matrix" type operator is not supported. Since no operation is performed, the output value is the same as the input value.')
        rtn=Xmat
        disp=pd.concat([pd.DataFrame(Xmat),pd.DataFrame(rtn)],keys=['input','result'])
    elif Xtype=='scala' or Xtype=='rowvec' or Xtype=='colvec':
        print('Input of this function should be a matrix. Thus no operation is performed, therefore the output value is the same as the input value.')
        rtn=Xmat
        if Xtype=='colvec': 
            disp=pd.concat([pd.DataFrame(Xmat),pd.DataFrame(rtn)],keys=['input','result'],axis=1)
        else:
            disp=pd.concat([pd.DataFrame(Xmat),pd.DataFrame(rtn)],keys=['input','result'],axis=0)
        if Xtype=='scala': rtn=Xmat[0,0]
    
    if plt==True:
        if type(disp) is str : print(disp)
        else:
            from IPython.display import display 
            display(disp)
    
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

def initmpd(*index,p=1,iname=None,vname=None): # mpd is short for multi-indexed pandas dataframe. 
    nofMindex=len(index)
    indexListtype=list(range(0,nofMindex))
    for i in range(0,nofMindex):
        if type(index[i]) is str: indexListtype[i]=[index[i]]
        elif np.asmatrix(index[i]).shape==(1,1): indexListtype[i]=[str(np.asmatrix(index[i])[0,0])]
        else: indexListtype[i]=index[i]

    mindex=pd.MultiIndex.from_product(indexListtype)
    if iname==None: iname=sprod('index',cc(1,len(indexListtype)))
    if vname==None: vname=sprod('X',cc(1,p))
    n=mindex.shape[0]
    val=init('0',(n,p))
    rtn=pd.DataFrame(val,index=mindex).reset_index()
    rtn.columns=iname+vname
    return rtn
# example: initmpd(['a','b','c'],['(i)','(ii)'],p=4)

def initpd(typ,n,p=1,vname=None): 
    if vname==None: vname=sprod('X',cc(1,p))
    val=init(typ,(n,p))
    rtn=pd.DataFrame(val)
    rtn.columns=vname
    return rtn   


## Interaction between R and python 
import rpy2
import rpy2.robjects as ro
#import rpy2.robjects.packages as rpkg
#from rpy2.robjects.packages import importr as library
ro.r('library(devtools)')

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


