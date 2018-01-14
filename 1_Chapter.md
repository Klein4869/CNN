<script type="text/javascript"
   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
# 矩阵与行列式

##一、行列式

### 1.行列式的定义

### 2.行列式的性质
#####①行列式等于转置
#####②行列式的变换法则（交换行列，某一行列倍加到另一列）
#####③行列式中某一行列各元素均为两项之和，可以拆分为两个行列式之和
#####④行列式的展开（代数余子式）
#####⑤常用行列式：对角行列式，分块对角行列式
#####⑥克莱姆法则：解线性方程组，性质如下
######&emsp;&emsp;(1)如果方程组的系数行列式$$D≠0$$
######&emsp;&emsp;那么它有唯一解$$x_{i}=\frac{D_{i}}{D}$$
######&emsp;&emsp;(2)方程组无解或者有两个不同解，则系数行列式\\(D=0\\)
######&emsp;&emsp;(3)若齐次线性方程组的系数行列式\\(D\neq0\\)，则只有\\(0\\)解,若有非零解，系数行列式\\(D=0\\)

##二、矩阵
###1.矩阵的定义和记号
###2.矩阵的运算（加减，数乘，矩阵的乘法，转置，幂，多项式，矩阵的行列式）
#####①矩阵的乘法不满足交换律，可交换是两个方阵之间的性质，叫做可交换性
#####②转置重点，易错$$(AB)^T = B^T A^T$$
#####③方阵多项式的定义，m次多项式有m+1项
#####&emsp;&emsp;两个\\(A\\)的多项式，\\(\phi(A)\\)和\\(f(A)\\)，满足
$$\phi(A)f(A) = f(A)\phi(A)$$
#####&emsp;&emsp;由此得知，方阵的多项式可以像输的多项式一样分解因式
#####④方阵的幂满足下列条件：
#####&emsp;&emsp;(i)\\(A^kA^l = A^{k+l}\\)
#####&emsp;&emsp;(ii)\\({(A^k)}^l = A^{kl}\\)
###3.逆矩阵