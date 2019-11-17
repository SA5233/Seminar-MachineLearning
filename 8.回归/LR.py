#!/usr/bin/env python
# coding: utf-8

# # 一，简单线性回归

# In[1]:


import numpy as np
import pandas as pd
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['simhei'] #显示中文
plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号

import os
os.chdir('C:\\Users\\SA\\Documents\\machine learning\\code\\Ch08')
# In[2]:


ex0 = pd.read_csv('ex0.txt',sep='\t',header=None)


# In[3]:


ex0.head()


# In[4]:


ex0.shape


# In[5]:


ex0.describe()


# # 获取特征矩阵和标签列

# In[6]:


def get_Mat(dataSet):
    xMat = np.mat(dataSet.iloc[:,:-1].values)
    yMat = np.mat(dataSet.iloc[:,-1].values).T
    return xMat,yMat


# In[7]:


xMat,yMat = get_Mat(ex0)


# In[8]:


xMat[:10]


# In[9]:


yMat[:10]


# # 数据可视化

# In[10]:


def plotShow(dataSet):
    xMat,yMat=get_Mat(dataSet)
    plt.scatter(xMat.A[:,1],yMat.A,c='b',s=5) #xMat[:,1]才是自变量
    plt.show()


# In[11]:


len(xMat.A[:,1])


# In[12]:


plotShow(ex0)


# # flatten 与 matrix.A 的用法

# In[13]:


a = [[1,3],[2,4],[3,5],["abc","def"]]  
a1 = [y for x in a for y in x]  


# ### flatten 对数组, 矩阵，不能对列表

# In[14]:


a = [[1,3],[2,4],[3,5]] 
a = np.array(a)
b = a.flatten()
a,b


# In[15]:


x = xMat[:10]
c = x.flatten()
c


# ### 对列表

# In[16]:


a = [[1,3],[2,4],[3,5],["abc","def"]]  
b = [y for x in a for y in x]  # 变量+外循环+内循环
b


# ### metrix.A 把矩阵转化为数组

# In[17]:


x = xMat[:10]
d = x.A
d


# ## $ \hat{w}=(X^TX)^{-1}X^Ty $

# In[18]:


def standRegres(dataSet):
    xMat,yMat =get_Mat(dataSet)
    xTx = xMat.T*xMat
    if np.linalg.det(xTx)==0:  #计算矩阵的行列式              
        print('矩阵为奇异矩阵，无法求逆')
        return
    ws=xTx.I*(xMat.T*yMat) #xTx.I 求逆
    return ws


# In[19]:


ws = standRegres(ex0)


# In[20]:


ws


# ## $ \hat{y}=X\cdot\hat{w} $

# In[21]:


def plotReg(dataSet):
    xMat,yMat=get_Mat(dataSet)
    plt.scatter(xMat.A[:,1],yMat.A,c='b',s=5)
    ws = standRegres(dataSet)
    yHat = xMat*ws
    plt.plot(xMat[:,1],yHat,c='r')
    plt.show()


# In[22]:


plotReg(ex0)


# In[23]:


xMat,yMat =get_Mat(ex0)
ws =standRegres(ex0)
yHat = xMat*ws
np.corrcoef(yHat.T,yMat.T) #皮尔逊积矩相关系数，保证两个都是行向量，输入一个矩阵


# In[24]:


v = np.vstack((yHat.T,yMat.T))  #把数组按照竖直方向堆叠，np.hstack()水平方向
np.corrcoef(v)


# # 二，局部加权线性回归
# ## $ SSE=(y-Xw)^TM(y-Xw) $
# ## $ M(i,i)=exp\{\frac{|x^i-x|^2}{-2k^2}\} $
# 

# In[25]:


#高斯核函数的图像
xMat,yMat = get_Mat(ex0)
x=0.5
xi = np.arange(0,1.0,0.01)
k1,k2,k3=0.5,0.1,0.01
m1 = np.exp((xi-x)**2/(-2*k1**2))
m2 = np.exp((xi-x)**2/(-2*k2**2))
m3 = np.exp((xi-x)**2/(-2*k3**2))
#创建画布
fig = plt.figure(figsize=(6,8),dpi=120)
#子画布1，原始数据集
fig1 = fig.add_subplot(411)
plt.scatter(xMat.A[:,1],yMat.A,c='b',s=5)
#子画布2，k=0.5
fig2 = fig.add_subplot(412)
plt.plot(xi,m1,color='r')
plt.legend(['k = 0.5'])
#子画布3，k=0.1
fig3 = fig.add_subplot(413)
plt.plot(xi,m2,color='g')
plt.legend(['k = 0.1'])
#子画布4，k=0.01
fig4 = fig.add_subplot(414)
plt.plot(xi,m3,color='orange')
plt.legend(['k = 0.01'])
plt.show()


# ## $ M(i,i)=exp\{\frac{|x^i-x|^2}{-2k^2}\} $
# ## $ \hat{w}=(X^TMX)^{-1}X^TMy $
# ## $ \hat{y}=X\cdot\hat{w} $

# In[26]:


def LWLR(testMat,xMat,yMat,k=1.0):
    n=testMat.shape[0]
    m=xMat.shape[0]
    weights =np.mat(np.eye(m)) #生成mxm的对角矩阵
    yHat = np.zeros(n) #生成n个0的数组
    for i in range(n):
        for j in range(m):
            diffMat = testMat[i]-xMat[j] # 测试集和训练集距离越近，权重越大
            weights[j,j]=np.exp(diffMat*diffMat.T/(-2*k**2))
        xTx = xMat.T*(weights*xMat)
        if np.linalg.det(xTx)==0:
            print('矩阵为奇异矩阵，不能求逆')
            return
        ws = xTx.I*(xMat.T*(weights*yMat))
        yHat[i]= testMat[i]*ws
    return ws,yHat


# In[27]:


np.argsort([2,1,3])


# In[28]:


xMat,yMat = get_Mat(ex0)
#将数据点排列（argsort()默认升序排列，返回索引）
srtInd = xMat[:,1].argsort(0) # 0 means 垂直排列（按行
#srtInd = np.argsort(xMat[:10,1],axis=0) 
xSort=xMat[srtInd][:,0]
xMat[srtInd].shape


# In[29]:


xSort


# In[30]:


#计算不同k取值下的y估计值yHat
ws1,yHat1 = LWLR(xMat,xMat,yMat,k=1.0)
ws2,yHat2 = LWLR(xMat,xMat,yMat,k=0.01)
ws3,yHat3 = LWLR(xMat,xMat,yMat,k=0.003)


# In[31]:


#创建画布
fig = plt.figure(figsize=(6,8),dpi=120)
#子图1绘制k=1.0的曲线
fig1=fig.add_subplot(311)  #将画布分割成3行1列，图像画在从左到右从上到下的第1块
plt.scatter(xMat[:,1].A,yMat.A,c='b',s=2)
plt.plot(xSort[:,1],yHat1[srtInd],linewidth=1,color='r')
#plt.plot(xMat[:,1],yHat1,c='r')
plt.title('局部加权回归曲线，k=1.0',size=10,color='r')
#子图2绘制k=0.01的曲线
fig2=fig.add_subplot(312)
plt.scatter(xMat[:,1].A,yMat.A,c='b',s=2)
plt.plot(xSort[:,1],yHat2[srtInd],linewidth=1,color='r')
plt.title('局部加权回归曲线，k=0.01',size=10,color='r')
#子图3绘制k=0.003的曲线
fig3=fig.add_subplot(313)
plt.scatter(xMat[:,1].A,yMat.A,c='b',s=2)
plt.plot(xSort[:,1],yHat3[srtInd],linewidth=1,color='r')
plt.title('局部加权回归曲线，k=0.003',size=10,color='r')
#调整子图的间距
plt.tight_layout(pad=1.2)
plt.show()


# In[32]:


fig = plt.figure(figsize=(6,3),dpi=100)
plt.scatter(xMat[:,1].A,yMat.A,c='b',s=2)
plt.plot(xMat[:,1],yHat2,linewidth=1,color='r')
plt.title('局部加权回归曲线，k=0.01',size=10,color='r')
plt.show()


# ### 动态演示

# In[33]:


#无排序动态图
import matplotlib.pyplot as plt

ax = []                    # 定义一个 x 轴的空列表用来接收动态的数据
ay = []                    # 定义一个 y 轴的空列表用来接收动态的数据
plt.ion()                  # 开启一个画图的窗口
for i in range(xMat.shape[0]):       # 遍历0-99的值
    ax.append(xMat[i,1])           # 添加 i 到 x 轴的数据中
    ay.append(yHat2[i])        # 添加 i 的平方到 y 轴的数据中
    plt.clf()              # 清除之前画的图
    plt.plot(ax,ay)        # 画出当前 ax 列表和 ay 列表中的值的图形
    plt.pause(0.1)         # 暂停一秒
    plt.ioff() 


# In[ ]:


#有排序动态图
import matplotlib.pyplot as plt

srtInd = xMat[:,1].argsort(0) # 0 means 垂直排列（按行
xsort = srtInd.A.flatten()
#xsort = []
#for i in range(xMat.shape[0]):
#    xsort.append(srtInd[i,0])
ax = []                    # 定义一个 x 轴的空列表用来接收动态的数据
ay = []                    # 定义一个 y 轴的空列表用来接收动态的数据
plt.ion()                  # 开启一个画图的窗口
for i in xsort:       # 遍历0-99的值
    ax.append(xMat[i,1])           # 添加 i 到 x 轴的数据中
    ay.append(yHat2[i])        # 添加 i 的平方到 y 轴的数据中
    plt.clf()              # 清除之前画的图
    plt.plot(ax,ay)        # 画出当前 ax 列表和 ay 列表中的值的图形
    plt.pause(0.1)         # 暂停一秒
    plt.ioff() 


# In[35]:


#四种模型相关系数比较
np.corrcoef(yHat.T,yMat.T)   #最小二乘法


# In[36]:


np.corrcoef(yHat1,yMat.T)    #k=1.0模型


# In[37]:


np.corrcoef(yHat2,yMat.T)    #k=0.01模型


# In[38]:


np.corrcoef(yHat3,yMat.T)    #k=0.003模型,过拟合


# # 三，预测鲍鱼的年龄

# In[39]:


abalone = pd.read_csv('abalone.txt',sep='\t',header=None)
abalone.columns=['性别','长度','直径','高度','整体重量','肉重量','内脏重量','壳重','年龄']


# In[40]:


abalone.head()


# In[41]:


abalone.shape


# In[42]:


abalone.info()


# In[43]:


abalone.describe()


# ## 数据可视化

# In[44]:


mpl.cm.rainbow(np.linspace(0, 1, 10))


# In[45]:


def dataPlot(dataSet):
    m,n=dataSet.shape
    fig = plt.figure(figsize=(8,20),dpi=100)
    colormap = mpl.cm.rainbow(np.linspace(0, 1, n))
    for i in range(n):
        fig_ = fig.add_subplot(n,1,i+1)
        plt.scatter(range(m),dataSet.iloc[:,i].values,s=2,c=colormap[i])  #第i列的数据描点画图
        plt.title(dataSet.columns[i])
        plt.tight_layout(pad=1.2)


# In[46]:


dataPlot(abalone)


# In[47]:


#剔除高度特征中≥0.4的异常值
aba = abalone.loc[abalone['高度']<0.4,:] 
dataPlot(aba)


# In[48]:


def randSplit(dataSet,rate):
    l = list(dataSet.index)
    random.seed(123)       #随机数种子，方便教学
    random.shuffle(l)
    dataSet.index = l
    m = dataSet.shape[0]
    n = int(m*rate)
    train = dataSet.loc[range(n),:]
    test = dataSet.loc[range(n,m),:]
    test.index=range(test.shape[0])
    dataSet.index =range(dataSet.shape[0])
    return train,test


# In[49]:


train,test = randSplit(aba,0.8)


# In[50]:


train.head()


# In[51]:


train.shape


# In[52]:


test.shape


# In[53]:


dataPlot(train)


# In[54]:


dataPlot(test)


# ## 计算误差平方和ESS

# In[55]:


def essCal(dataSet, regres):          # regress 求回归系数的函数
    xMat,yMat = get_Mat(dataSet)
    ws = regres(dataSet)
    yHat = xMat*ws
    ess = ((yMat.A.flatten() - yHat.A.flatten())**2).sum()
    return ess


# In[56]:


essCal(ex0, standRegres)


# ## 计算 $ R^2 $

# In[57]:


def rSquare(dataSet,regres):
    xMat,yMat=get_Mat(dataSet)
    ess = essCal(dataSet,regres)
    tss = ((yMat.A-yMat.mean())**2).sum()
    r2 = 1 - ess / tss
    return r2


# In[58]:


rSquare(ex0, standRegres)


# ## 构建加权线性模型

# In[59]:


def essPlot(train,test):
    X0,Y0 = get_Mat(train)
    X1,Y1 =get_Mat(test)
    train_ess = []
    test_ess = []
    for k in np.arange(0.2,10,0.5):
        ws1,yHat1 = LWLR(X0[:99],X0[:99],Y0[:99],k)  #数据量大计算缓慢，选取前一百个数据
        ess1 = ((Y0[:99].A.T - yHat1)**2).sum()
        train_ess.append(ess1)
        
        ws2,yHat2 = LWLR(X1[:99],X0[:99],Y0[:99],k)
        ess2 = ((Y1[:99].A.T - yHat2)**2).sum()
        test_ess.append(ess2)
        
    plt.plot(np.arange(0.2,10,0.5),train_ess,color='b')
    plt.plot(np.arange(0.2,10,0.5),test_ess,color='r')
    plt.xlabel('不同k取值')
    plt.ylabel('ESS')
    plt.legend(['train_ess','test_ess'])


# In[60]:


essPlot(train,test)


# In[61]:


train,test = randSplit(aba,0.8)
trainX,trainY = get_Mat(train)
testX,testY = get_Mat(test)
ws0,yHat0 = LWLR(testX,trainX,trainY,k=2)


# In[62]:


y=testY.A.flatten()
plt.scatter(y,yHat0,c='b',s=5);  # ; 代替plt.show()


# In[63]:


def LWLR_pre(dataSet):
    train,test = randSplit(dataSet,0.8)
    trainX,trainY = get_Mat(train)
    testX,testY = get_Mat(test)
    ws,yHat = LWLR(testX,trainX,trainY,k=2)
    ess = ((testY.A.T - yHat)**2).sum()
    tss = ((testY.A-testY.mean())**2).sum()
    r2 = 1 - ess / tss
    return ess,r2


# In[ ]:


LWLR_pre(aba)


# # 四，岭回归

# ## $ \hat{w} = (X^TX+\lambda I )^{-1}X^Ty $

# In[ ]:


np.eye(5)


# In[118]:


def ridgeRegres(dataSet, lam=0.2):
    xMat,yMat=get_Mat(dataSet)  # xMat 行数<列数
    xTx = xMat.T * xMat
    denom = xTx + np.eye(xMat.shape[1])*lam
    ws = denom.I * (xMat.T * yMat)
    return ws


# In[119]:


#回归系数比较
standRegres(aba)         #线性回归


# In[ ]:


ridgeRegres(aba)         #岭回归


# In[ ]:


#相关系数R2比较
rSquare(aba,standRegres) #线性回归


# In[ ]:


rSquare(aba,ridgeRegres) #岭回归


# In[ ]:


def ridgeTest(dataSet,k=30):  #取30个不同的λ值
    xMat,yMat=get_Mat(dataSet)
    m,n=xMat.shape
    wMat = np.zeros((k,n))
    #特征标准化
    yMean = yMat.mean(0)  # 0 means 按列计算
    xMeans = xMat.mean(0)
    xVar = xMat.var(0)
    yMat = yMat-yMean
    xMat = (xMat-xMeans)/xVar
    for i in range(k):
        xTx = xMat.T*xMat
        lam = np.exp(i-10)
        denom = xTx+np.eye(n)*lam    #lam增速快
        ws=denom.I*(xMat.T*yMat)
        wMat[i,:]=ws.T
    return wMat


# In[ ]:


k = np.arange(0,30,1)
lam = np.exp(k-10)
plt.plot(lam);


# In[ ]:


#回归系数矩阵
wMat = ridgeTest(aba,k=30)


# In[ ]:


wMat.shape


# In[ ]:


#绘制岭迹图
plt.plot(np.arange(-10,20,1),wMat)
plt.xlabel('log(λ)')
plt.ylabel('回归系数');


# ## np里面的三个log函数

# In[ ]:


e = np.exp(1)
np.log(e)


# In[ ]:


np.log2(2)


# In[ ]:


np.log10(10)


# # 六，lasso

# In[64]:


#lasso是在linear_model下
from sklearn.linear_model import Lasso


# In[65]:


las = Lasso(alpha = 0.05)   #alpha为惩罚系数，值越大惩罚力度越大
las.fit(aba.iloc[:, :-1], aba.iloc[:, -1])


# In[67]:


las.coef_


# In[68]:


def regularize(xMat,yMat):
    inxMat = xMat.copy()                   #数据拷贝
    inyMat = yMat.copy()
    yMean = yMat.mean(0)                   #行与行操作，求均值
    inyMat = inyMat - yMean                #数据减去均值
    xMeans = inxMat.mean(0)                #行与行操作，求均值
    xVar = inxMat.var(0)                   #行与行操作，求方差
    inxMat = (inxMat - xMeans) / xVar      #数据减去均值除以方差实现标准化
    return inxMat, inyMat


# In[69]:


def rssError(yMat, yHat):
    ess = ((yMat.A-yHat.A)**2).sum()
    return ess


# ## 向前逐步回归

# In[70]:


def stageWise(dataSet, eps = 0.01, numIt = 100):
    xMat0,yMat0 = get_Mat(dataSet)             
    xMat,yMat = regularize(xMat0, yMat0)            #数据标准化
    m, n = xMat.shape
    wsMat = np.zeros((numIt, n))                    #初始化numIt次迭代的回归系数矩阵
    ws = np.zeros((n, 1))                           #初始化回归系数矩阵
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):                          #迭代numIt次
        # print(ws.T)                               #打印当前回归系数矩阵
        lowestError = np.inf                        #正无穷
        for j in range(n):                          #遍历每个特征的回归系数
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign             #微调回归系数
                yHat = xMat * wsTest                #计算预测值
                ess = rssError(yMat, yHat)          #计算平方误差
                if ess < lowestError:               #如果误差更小，则更新当前的最佳回归系数
                    lowestError = ess
                    wsMax = wsTest
        ws = wsMax.copy()
        wsMat[i,:] = ws.T                           #记录numIt次迭代的回归系数矩阵
    return wsMat


# In[74]:


stageWise(aba, eps = 0.01, numIt = 200)


# In[78]:


wsMat= stageWise(aba, eps = 0.001, numIt = 5000)
wsMat


# In[75]:


def standRegres0(dataSet):
    xMat0,yMat0 =get_Mat(dataSet)
    xMat,yMat = regularize(xMat0, yMat0) #增加标准化这一步
    xTx = xMat.T*xMat
    if np.linalg.det(xTx)==0:
        print('矩阵为奇异矩阵，无法求逆')
        return
    ws=xTx.I*(xMat.T*yMat)
    yHat = xMat*ws
    return ws


# In[76]:


standRegres0(aba).T


# In[79]:


wsMat[-1]


# In[80]:


plt.plot(wsMat)
plt.xlabel('迭代次数')
plt.ylabel('回归系数');


# # 七，预测乐高玩具的价格

# In[81]:


from bs4 import BeautifulSoup


# In[82]:


infile = 'lego\lego8288.html'
HTML_DOC = open(infile,encoding = 'utf-8').read()


# In[83]:


soup = BeautifulSoup(HTML_DOC,'lxml')
i=1
currentRow = soup.find_all('table', r = f'{i}')
currentRow


# In[84]:


currentRow[0].find_all('a')[1].text


# In[85]:


title = currentRow[0].find_all('a')[1].text


# In[86]:


title


# In[87]:


lwrTitle = title.lower()


# In[88]:


lwrTitle


# In[92]:


lwrTitle.find('lego') #返回位置


# In[93]:


currentRow[0].find_all('td')[3].find_all('span')


# In[94]:


currentRow[0].find_all('td')[4]


# In[97]:


soldPrice = currentRow[0].find_all('td')[4].text
soldPrice


# In[96]:


soldPrice.replace('$','')


# In[98]:


def scrapePage(data, infile, yr, numPce, origPrc):
    HTML_DOC = open(infile,encoding = 'utf-8').read()
    soup = BeautifulSoup(HTML_DOC,'lxml')
    i=1
    #根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r = f'{i}')
    while(len(currentRow) != 0):
        currentRow = soup.find_all('table', r = f'{i}')
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        #查找是否有全新标签
        if (lwrTitle.find('new') > -1):
            newFlag = 1
        else:newFlag = 0
        #查找是否已经标志出售，我们只收集已出售的数据
        soldbutt = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldbutt) == 0:
            print(f"商品 #{i} 没有出售")
        else:
            #解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','')
            priceStr = priceStr.replace(',','')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            #去掉不完整的套装价格
            if  sellingPrice > origPrc * 0.5:
                data.append([yr, numPce, newFlag, origPrc,sellingPrice])
        i+=1
        currentRow = soup.find_all('table', r = f'{i}')


# In[99]:


def setDataCollect(data):
    scrapePage(data, 'lego/lego8288.html', 2006, 800, 49.99)    
    scrapePage(data, 'lego/lego10030.html', 2002, 3096, 269.99) 
    scrapePage(data, 'lego/lego10179.html', 2007, 5195, 499.99) 
    scrapePage(data, 'lego/lego10181.html', 2007, 3428, 199.99) 
    scrapePage(data, 'lego/lego10189.html', 2008, 5922, 299.99) 
    scrapePage(data, 'lego/lego10196.html', 2009, 3263, 249.99) 


# In[100]:


data=[]
setDataCollect(data)


# In[102]:


df = pd.DataFrame(data)


# In[103]:


df.head()


# In[104]:


df.columns = ['出品年份','部件数目','全新否','原价','二手售价']
df.head()


# In[105]:


df.info()


# In[106]:


df.describe()


# In[107]:


col_name = df.columns.tolist()


# In[108]:


col_name


# In[109]:


col_name.insert(0,'X0')
col_name


# In[110]:


df=df.reindex(columns=col_name)


# In[111]:


df["X0"]=1


# In[112]:


df.head()


# In[113]:


ws = standRegres(df)
ws


# In[114]:


xMat,yMat = get_Mat(df)
yHat = xMat*ws


# In[115]:


#画出真实值和预测值的散点图
plt.scatter(range(df.shape[0]),yMat.A)
plt.scatter(range(df.shape[0]),yHat.A);


# In[116]:


#简单线性回归
rSquare(df,standRegres)


# In[120]:


#岭回归
rSquare(df,ridgeRegres)


# In[121]:


from sklearn.tree import DecisionTreeRegressor as DTR
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
dtr = DTR().fit(X, y)
dtr.score(X, y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




