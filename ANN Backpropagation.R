library(neuralnet)
library(forecast)
library(nnfor)

dataa=read.csv(file.choose())
data2=dataa$RT
data1=ts(data2,frequency=12,start=c(2017))
plot.ts(data1, main="Plot Jumlah Penumpang Kereta Sumatera Utara", xlab="periode",ylab="Volume penumpang",col="blue")
# atau autoplot(data1) 

pacf(data2, lag.max = 50) 
#lag nya ada 2 yaitu 1 sama 16

#function untuk membuat dataframe target dan lag-lag nya 
makeinput = function(x, lags = NULL){ 
  n = length(x) 
  a = matrix(0, nrow = length(x)-lags, ncol = 1+lags) 
  a[,1] = x[-c(1:lags)] 
  a[,1+lags] = x[-c((n-lags+1):n)] 
  for (i in 1:(lags-1)) { 
    a[,i+1] = x[-c(1:(lags-i),(n+1-i):n)] 
  } 
  Ytarget = a[,1] 
  Xinput = a[,(2:(lags+1))] 
  a = data.frame(Ytarget,Xinput) 
  return(a) 
} 
data = makeinput(data1, lags = 2)
data 
Ytarget<-data[c(1)]
Xinput<-data[c(2,3)]
data<-data.frame(Ytarget,Xinput)

#linearitas
colnames(data)= c("Y","X1","X2")
View(data)
dim(data)
string(data)
library(tseries)
terasvirta.test(data$X1+X2,data$Y)

##Normalisasi
normalize <- function(x) {return ((x - min(x)) / (max(x) - min(x)))}
st.X<-normalize(Xinput)
st.Y<-normalize(Ytarget)
dtransform<-data.frame(st.Y,st.X)

##pembagian data 60% training 40% testing
#untuk data asli
rows<-1:round(nrow(data)*0.60)
dtraining<-as.data.frame(data[rows,])
dtesting<-as.data.frame(data[-rows,])
rows<-1:round(nrow(dtransform)*0.60)
datatraining<-as.data.frame(dtransform[rows,]) 
datatesting<-as.data.frame(dtransform[-rows,]) 



#===TAHAPAN TRAINING===#


#hidden neuron = 2
nn2 <- neuralnet(st.Y ~ X1+X2, 
                     data = datatraining, 
                     hidden=c(2), 
                     linear.output=FALSE, 
                     algorithm = 'backprop',
                     learningrate = 0.001,
                     err.fct = 'sse',
                     act.fct ='logistic',
                     startweights = NULL,
                     stepmax = 100000)

nn3 <- neuralnet(st.Y ~ X1+X2, 
                     data = datatraining, 
                     hidden=c(3), 
                     linear.output=FALSE, 
                     algorithm = 'backprop',
                     learningrate = 0.001,
                     err.fct = 'sse',
                     act.fct ='logistic',
                     startweights = NULL,
                     stepmax = 100000)
#hidden neuron =4
nn4 <- neuralnet(st.Y ~ X1+X2, 
                     data = datatraining, 
                     hidden=c(4), 
                     linear.output=FALSE, 
                     algorithm = 'backprop',
                     learningrate = 0.001,
                     err.fct = 'sse',
                     act.fct ='logistic',
                     startweights = NULL,
                     stepmax = 100000)
#hidden neuron =5
nn5 <- neuralnet(st.Y ~ X1+X2, 
                     data = datatraining, 
                     hidden=c(5), 
                     linear.output=FALSE, 
                     algorithm = 'backprop',
                     learningrate = 0.001,
                     err.fct = 'sse',
                     act.fct ='logistic',
                     startweights = NULL,
                     stepmax = 100000)
#hidden neuron = 6
nn6 <- neuralnet(st.Y ~ X1+X2, 
                     data = datatraining, 
                     hidden=c(6), 
                     linear.output=FALSE, 
                     algorithm = 'backprop',
                     learningrate = 0.001,
                     err.fct = 'sse',
                     act.fct ='logistic',
                     startweights = NULL,
                     stepmax = 100000)
#hidden neuron = 7
nn7 <- neuralnet(st.Y ~ X1+X2, 
                     data = datatraining, 
                     hidden=c(7), 
                     linear.output=FALSE, 
                     algorithm = 'backprop',
                     learningrate = 0.001,
                     err.fct = 'sse',
                     act.fct ='logistic',
                     startweights = NULL,
                     stepmax = 100000)

#hidden neuron = 8
nn8 <- neuralnet(st.Y ~ X1+X2, 
                     data = datatraining, 
                     hidden=c(8), 
                     linear.output=FALSE, 
                     algorithm = 'backprop',
                     learningrate = 0.001,
                     err.fct = 'sse',
                     act.fct ='logistic',
                     startweights = NULL,
                     stepmax = 100000)
#hidden neuron = 9
nn9 <- neuralnet(st.Y ~ X1+X2, 
                     data = datatraining, 
                     hidden=c(9), 
                     linear.output=FALSE, 
                     algorithm = 'backprop',
                     learningrate = 0.001,
                     err.fct = 'sse',
                     act.fct ='logistic',
                     startweights = NULL,
                     stepmax = 100000)

#==HASIL TRAINING
training <- subset(datatraining, select = c("X1","X2"))

#nn2
nn.results.training2 <- compute(nn2, training)
results.training2 <- data.frame(actual = datatraining$st.Y, 
                                prediction = 
                                  nn.results.training2$net.result)
results.training2


#nn3
nn.results.training3 <- compute(nn3, training)
results.training3 <- data.frame(actual = datatraining$st.Y, 
                                prediction = 
                                  nn.results.training3$net.result)
results.training3

#nn4
nn.results.training4 <- compute(nn4, training)
results.training4 <- data.frame(actual = datatraining$st.Y, 
                                prediction = 
                                  nn.results.training4$net.result)
results.training4

#nn5
nn.results.training5 <- compute(nn5, training)
results.training5 <- data.frame(actual = datatraining$st.Y, 
                                prediction = 
                                  nn.results.training5$net.result)
results.training5

#nn6
nn.results.training6 <- compute(nn6, training)
results.training6 <- data.frame(actual = datatraining$st.Y, 
                                prediction = 
                                  nn.results.training6$net.result)
results.training6

#nn7
nn.results.training7 <- compute(nn7, training)
results.training7 <- data.frame(actual = datatraining$st.Y, 
                                prediction = 
                                  nn.results.training7$net.result)
results.training7

#nn8
nn.results.training8 <- compute(nn8, training)
results.training8 <- data.frame(actual = datatraining$st.Y, 
                                prediction = 
                                  nn.results.training8$net.result)
results.training8

#nn9
nn.results.training9 <- compute(nn9, training)
results.training9 <- data.frame(actual = datatraining$st.Y, 
                                prediction = 
                                  nn.results.training9$net.result)
results.training9


#==AKURASI TRAINING

#nn2
actual2=results.training2$actual
predicted2=results.training2$prediction
e2=actual2-predicted2
mse_training2=mean((e2)^2)
mse_training2
rs_training2 = 1-(var(e2)/var(actual2))
rs_training2

#nn3
actual3=results.training3$actual
predicted3=results.training3$prediction
e3=actual3-predicted3
mse_training3=mean((e3)^2)
mse_training3
rs_training3 = 1-(var(e3)/var(actual3))
rs_training3

#nn4
actual4=results.training4$actual
predicted4=results.training4$prediction
e4=actual4-predicted4
mse_training4=mean((e4)^2)
mse_training4
rs_training4 = 1-(var(e4)/var(actual4))
rs_training4

#nn5
actual5=results.training5$actual
predicted5=results.training5$prediction
e5=actual5-predicted5
mse_training5=mean((e5)^2)
mse_training5
rs_training5 = 1-(var(e5)/var(actual5))
rs_training5

#nn6
actual6=results.training6$actual
predicted6=results.training6$prediction
e6=actual6-predicted6
mse_training6=mean((e6)^2)
mse_training6
rs_training6 = 1-(var(e6)/var(actual6))
rs_training6

#nn7
actual7=results.training7$actual
predicted7=results.training7$prediction
e7=actual7-predicted7
mse_training7=mean((e7)^2)
mse_training7
rs_training7 = 1-(var(e7)/var(actual7))
rs_training7
MAPE=(sum(abs(e7/data))/65)*100
MAPE

#nn8
actual8=results.training8$actual
predicted8=results.training8$prediction
e8=actual8-predicted8
mse_training8=mean((e8)^2)
mse_training8
rs_training8 = 1-(var(e8)/var(actual8))
rs_training8
MAPE=(sum(abs(e8/data))/65)*100
MAPE

#nn9
actual9=results.training9$actual
predicted9=results.training9$prediction
e9=actual9-predicted9
mse_training9=mean((e9)^2)
mse_training9
rs_training9 = 1-(var(e9)/var(actual9))
rs_training9
MAPE=(sum(abs(e9/data))/65)*100
MAPE

#===TAHAPAN TESTING===#
nntest2 <- neuralnet(st.Y ~ X1+X2, 
                        data = datatesting, 
                        hidden=c(2), 
                        linear.output=FALSE, 
                        algorithm = 'backprop',
                        learningrate = 0.01,
                        err.fct = 'sse',
                        act.fct ='logistic',
                        startweights = NULL,
                        stepmax = 100000)

nntest3 <- neuralnet(st.Y ~ X1+X2, 
                        data = datatesting, 
                        hidden=c(3), 
                        linear.output=FALSE, 
                        algorithm = 'backprop',
                        learningrate = 0.01,
                        err.fct = 'sse',
                        act.fct ='logistic',
                        startweights = NULL,
                        stepmax = 100000)
nntest4 <- neuralnet(st.Y ~ X1+X2, 
                        data = datatesting, 
                        hidden=c(4), 
                        linear.output=FALSE, 
                        algorithm = 'backprop',
                        learningrate = 0.01,
                        err.fct = 'sse',
                        act.fct ='logistic',
                        startweights = NULL,
                        stepmax = 100000)
nntest5 <- neuralnet(st.Y ~ X1+X2, 
                        data = datatesting, 
                        hidden=c(5), 
                        linear.output=FALSE, 
                        algorithm = 'backprop',
                        learningrate = 0.01,
                        err.fct = 'sse',
                        act.fct ='logistic',
                        startweights = NULL,
                        stepmax = 100000)
nntest6 <- neuralnet(st.Y ~ X1+X2, 
                        data = datatesting, 
                        hidden=c(6), 
                        linear.output=FALSE, 
                        algorithm = 'backprop',
                        learningrate = 0.01,
                        err.fct = 'sse',
                        act.fct ='logistic',
                        startweights = NULL,
                        stepmax = 100000)
nntest7 <- neuralnet(st.Y ~ X1+X2, 
                        data = datatesting, 
                        hidden=c(7), 
                        linear.output=FALSE, 
                        algorithm = 'backprop',
                        learningrate = 0.01,
                        err.fct = 'sse',
                        act.fct ='logistic',
                        startweights = NULL,
                        stepmax = 100000)
nntest8 <- neuralnet(st.Y ~ X1+X2, 
                        data = datatesting, 
                        hidden=c(8), 
                        linear.output=FALSE, 
                        algorithm = 'backprop',
                        learningrate = 0.01,
                        err.fct = 'sse',
                        act.fct ='logistic',
                        startweights = NULL,
                        stepmax = 100000)
nntest9 <- neuralnet(st.Y ~ X1+X2, 
                        data = datatesting, 
                        hidden=c(9), 
                        linear.output=FALSE, 
                        algorithm = 'backprop',
                        learningrate = 0.01,
                        err.fct = 'sse',
                        act.fct ='logistic',
                        startweights = NULL,
                        stepmax = 100000)

#==HASIL Testing
testing <- subset(datatesting, select = c("X1","X2"))

#nn2
nn.results.testing2 <- compute(nn2, testing)
results.testing2 <- data.frame(actual = datatesting$st.Y, 
                                prediction = 
                                  nn.results.testing2$net.result)
results.testing2


#nn3
nn.results.testing3 <- compute(nn3, testing)
results.testing3 <- data.frame(actual = datatesting$st.Y, 
                                prediction = 
                                  nn.results.testing3$net.result)
results.testing3

#nn4
nn.results.testing4 <- compute(nn4, testing)
results.testing4 <- data.frame(actual = datatesting$st.Y, 
                                prediction = 
                                  nn.results.testing4$net.result)
results.testing4

#nn5
nn.results.testing5 <- compute(nn5, testing)
results.testing5 <- data.frame(actual = datatesting$st.Y, 
                                prediction = 
                                  nn.results.testing5$net.result)
results.testing5

#nn6
nn.results.testing6 <- compute(nn6, testing)
results.testing6 <- data.frame(actual = datatesting$st.Y, 
                                prediction = 
                                  nn.results.testing6$net.result)
results.testing6

#nn7
nn.results.testing7 <- compute(nn7,testing)
results.testing7 <- data.frame(actual = datatesting$st.Y, 
                                prediction = 
                                  nn.results.testing7$net.result)
results.testing7

#nn8
nn.results.testing8 <- compute(nn8, testing)
results.testing8 <- data.frame(actual = datatesting$st.Y, 
                                prediction = 
                                  nn.results.testing8$net.result)
results.testing8
#nn9
nn.results.testing9 <- compute(nn9, testing)
results.testing9 <- data.frame(actual = datatesting$st.Y, 
                                prediction = 
                                  nn.results.testing9$net.result)
results.testing9



#akurasi
#nn2
actual2=results.testing2$actual
predicted2=results.testing2$prediction
e2=actual2-predicted2
mse_testing2=mean((e2)^2)
mse_testing2
rs_testing2 = 1-(var(e2)/var(actual2))
rs_testing2
MAPE=(sum(abs(e2/data))/65)*100
MAPE

#nn2
predictedtesting2=results.testing2$prediction
actualtesting2=results.testing2$actual
etesting2=actualtesting2-predictedtesting2
mse_testing2<-mean((etesting2)^2)
mse_testing2
rs_testing2 = 1-(var(etesting2)/var(actualtesting2))
rs_testing2
MAPE=(sum(abs(etesting2/data))/65)*100
MAPE

#nntest3
predictedtesting3=results.testing3$prediction
actualtesting3=results.testing3$actual
etesting3=actualtesting3-predictedtesting3
mse_testing3<-mean((etesting3)^2)
mse_testing3
rs_testing3 = 1-(var(etesting3)/var(actualtesting3))
rs_testing3
MAPE=(sum(abs(etesting3/data))/65)*100
MAPE

#nntest4
predictedtesting4=results.testing4$prediction
actualtesting4=results.testing4$actual
etesting4=actualtesting4-predictedtesting4
mse_testing4<-mean((etesting4)^2)
mse_testing4
rs_testing4 = 1-(var(etesting4)/var(actualtesting4))
rs_testing4
MAPE=(sum(abs(etesting4/data))/65)*100
MAPE

#nntest5
predictedtesting5=results.testing5$prediction
actualtesting5=results.testing5$actual
etesting5=actualtesting5-predictedtesting5
mse_testing5<-mean((etesting5)^2)
mse_testing5
rs_testing5 = 1-(var(etesting5)/var(actualtesting5))
rs_testing5
MAPE=(sum(abs(etesting5/data))/65)*100
MAPE

#nntest6
predictedtesting6=results.testing6$prediction
actualtesting6=results.testing6$actual
etesting6=actualtesting6-predictedtesting6
mse_testing6<-mean((etesting6)^2)
mse_testing6
rs_testing6 = 1-(var(etesting6)/var(actualtesting6))
rs_testing6
MAPE=(sum(abs(etesting6/data))/65)*100
MAPE

#nntest7
predictedtesting7=results.testing7$prediction
actualtesting7=results.testing7$actual
etesting7=actualtesting7-predictedtesting7
mse_testing7<-mean((etesting7)^2)
mse_testing7
rs_testing7 = 1-(var(etesting7)/var(actualtesting7))
rs_testing7
MAPE=(sum(abs(etesting7/data))/65)*100
MAPE

#nntest8
predictedtesting8=results.testing8$prediction
actualtesting8=results.testing8$actual
etesting8=actualtesting8-predictedtesting8
mse_testing8<-mean((etesting8)^2)
mse_testing8
rs_testing8 = 1-(var(etesting8)/var(actualtesting8))
rs_testing8
MAPE=(sum(abs(etesting8/data))/65)*100
MAPE

#nntest9
predictedtesting9=results.testing9$prediction
actualtesting9=results.testing9$actual
etesting9=actualtesting9-predictedtesting9
mse_testing9<-mean((etesting9)^2)
mse_testing9
rs_testing9 = 1-(var(etesting9)/var(actualtesting9))
rs_testing9
MAPE=(sum(abs(etesting9/data))/65)*100
MAPE

#Plot data dari jaringan dengan error terkecil (2 hidden neuron)
#arsitektur jaringan (2-3-1)

#PLOT TRAINING
aslidetransform=as.ts(actual3*((max(Xinput)-
                                  min(Xinput)))+min(Xinput))
ramalandetransform=as.ts(predicted3*((max(Ytarget)-
                                        min(Ytarget)))+min(Ytarget))
ts.plot(aslidetransform,ramalandetransform,lty=c(1,3),col=c(
  3,7),main="Plot Hasil Ramalan Data Training Jumlah Penumpang Kereta Api Divre I 2014-2022")

#PLOT TESTING
aslitestingdetransform=as.ts(actualtesting3*((max(Xinput)-
                                                min(Xinput)))+min(Xinput))
ramalantestingdetransform=as.ts(predictedtesting3*((max(Ytarget)-
                                                      min(Ytarget)))+min(Ytarget))
ts.plot(aslitestingdetransform,ramalantestingdetransform,lty
        =c(1,3),col=c(3,7),main= "Plot Hasil Ramalan Data Testing Jumlah Penumpang Kereta Api Divre I 2014-2022 ")

nn2$result.matrix

#FORECAST
#menggunakan 7 data terakhir
p = as.matrix(dtransform[57:63,])
p
prediksi=predict(nn3,p,n.ahead=7)
prediksi
q = as.matrix(Ytarget[57:63])
q
prediksi2=prediksi*((max(q)-min(q)))+min(q)
prediksi2



