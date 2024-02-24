library(bigtime)

## H-Lag penalty

path = "./Data/"
file.names <- dir(path,pattern = ".csv")

prederro = matrix(data = NA,nrow = 1, ncol = length(file.names),byrow = FALSE,dimnames = NULL)
for(j in 1:length(file.names))
{
  Y<-read.csv(paste0(path,file.names[j]),head=FALSE)
  num_samp <-length(Y)
  y <-  matrix(data = NA, nrow = 1, ncol = num_samp,byrow = FALSE,dimnames = NULL)
  AR_root <-matrix(data = NA, nrow = 1, ncol = num_samp,byrow = FALSE,dimnames = NULL)
  MA_root <-matrix(data = NA, nrow = 1, ncol = num_samp,byrow = FALSE,dimnames = NULL)
  P<-matrix(data = NA, nrow = 1, ncol = num_samp,byrow = FALSE,dimnames = NULL)
  Q<-matrix(data = NA, nrow = 1, ncol = num_samp,byrow = FALSE,dimnames = NULL)
  phitheta =matrix(data = NA, nrow = num_samp, ncol = 20,byrow = FALSE,dimnames = NULL) 
  for(i in 1:num_samp)
  {
    VARMAfit <-sparseVARMA(Y=as.matrix(Y[1:200,i]),VARpen = 'HLag',VARMApen = 'HLag')
    y[i] <- directforecast(fit=VARMAfit,model = 'VARMA',h=1)
    AR_root[i] <-max(abs(1/polyroot(c(1,-VARMAfit$Phihat))))
    MA_root[i] <-max(abs(1/polyroot(c(1,VARMAfit$Thetahat))))
    P[i] <- colSums(as.matrix(VARMAfit$Phihat) !=0)
    Q[i] <- colSums(as.matrix(VARMAfit$Thetahat) !=0)
    phitheta[i,] <-c(VARMAfit$Phihat,VARMAfit$Thetahat)
  }
  write.csv(rbind(P,Q),file= paste0("./R_result/","H_PQ_",file.names[j]))
  write.csv(phitheta,file = paste0("./R_result/","H_phitheta_",file.names[j]))
  






## L1 penalty
path = "./Data/"
file.names <- dir(path,pattern = ".csv")
prederro = matrix(data = NA,nrow = 1, ncol = length(file.names),byrow = FALSE,dimnames = NULL)
for(j in 1:length(file.names))
{
Y<-read.csv(paste0(path,file.names[j]),head=FALSE)
num_samp <-length(Y)
y <-  matrix(data = NA, nrow = 1, ncol = num_samp,byrow = FALSE,dimnames = NULL)
AR_root <-matrix(data = NA, nrow = 1, ncol = num_samp,byrow = FALSE,dimnames = NULL)
MA_root <-matrix(data = NA, nrow = 1, ncol = num_samp,byrow = FALSE,dimnames = NULL)
P<-matrix(data = NA, nrow = 1, ncol = num_samp,byrow = FALSE,dimnames = NULL)
Q<-matrix(data = NA, nrow = 1, ncol = num_samp,byrow = FALSE,dimnames = NULL)
phitheta =matrix(data = NA, nrow = num_samp, ncol = 20,byrow = FALSE,dimnames = NULL)
for(i in 1:num_samp)
{
VARMAfit <-sparseVARMA(Y=as.matrix(Y[1:200,i]),VARpen = 'L1',VARMApen = 'L1')
y[i] <- directforecast(fit=VARMAfit,model = 'VARMA',h=1)
AR_root[i] <-max(abs(1/polyroot(c(1,-VARMAfit$Phihat))))
MA_root[i] <-max(abs(1/polyroot(c(1,VARMAfit$Thetahat))))
P[i] <- colSums(as.matrix(VARMAfit$Phihat) !=0)
Q[i] <- colSums(as.matrix(VARMAfit$Thetahat) !=0)
phitheta[i,] <-c(VARMAfit$Phihat,VARMAfit$Thetahat)
}
write.csv(rbind(P,Q),file= paste0("./R_result/","L_PQ_",file.names[j]))
write.csv(phitheta,file = paste0("./R_result/","L_phitheta_",file.names[j]))
prederro[j] <-mean(as.matrix(y - Y[201,])^2)
}
#erro=matrix(data=NA,nrow=7,ncol=1,byrow=FALSE,dimnames = NULL)
#erro[1:4]=prederro[4:7]
#erro[5:7]=prederro[1:3]
write.csv(prederro,file = "./R_result/Lerro.csv")
  prederro[j] <-mean(as.matrix(y - Y[201,])^2)
  
}
#erro=matrix(data=NA,nrow=7,ncol=1,byrow=FALSE,dimnames = NULL)
#erro[1:4]=prederro[4:7]
#erro[5:7]=prederro[1:3]
write.csv(prederro,file = "./R_result/Herro.csv")