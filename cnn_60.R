library(mxnet)  
 
wholedata=read.table("C:/D/peptide-mhc/MSfeature60.txt",stringsAsFactors=FALSE)
name=unique(wholedata[,1])
pl=matrix(0,length(name),1)
for (i in 1:length(name)){
index=which(wholedata[,1]==name[i])
pl[i]=length(index)
}
index=which(pl>20)
name=name[index]
err_rate=matrix(0,length(name),5)
for (ii in 2:193) {
print(ii)
	index=which(wholedata[,1]==name[ii])
length(index)
	dat=wholedata[index,3:63]
	dat=dat[sample(nrow(dat)),]
	testnum=floor(length(index)/5)
	for (kk in 1:5) {
		index_test=((kk-1)*testnum+1):(kk*testnum)
		test=dat[index_test,]
		total=1:length(index)
		index_train=setdiff(total,index_test)
		train=dat[index_train,]
		train <- data.matrix(train)  
		train_x <- t(train[, -1])  
		train_y <- train[, 1]  
		for (i in 1:length(train_y)){
			if (train_y[i]>500){
				train_y[i]=0
			}
			else{
				train_y[i]=1
			}
		}
	
	train_array <- train_x  
	dim(train_array) <- c(15,4, 1, ncol(train_x))
	test_x <- t(test[,-1])  
	test_y <- test[, 1]
	for (i in 1:length(test_y)){
			if (test_y[i]>500){
				test_y[i]=0
			}
			else{
				test_y[i]=1
			}
		}
	test_array <- test_x  
	dim(test_array) <- c(15, 4, 1, ncol(test_x))  
	
	data <- mx.symbol.Variable('data')  


conv_1 <- mx.symbol.Convolution(data = data, kernel = c(2, 2), num_filter = 180)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "relu")  
  

pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))  
  

  
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(1, 1), num_filter = 80)  
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "relu")

pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(1, 1), stride = c(2, 2)) 
 


flatten1 <- mx.symbol.Flatten(data = pool_2) 
fc_2 <- mx.symbol.FullyConnected(data = flatten1, num_hidden = 2) 
 


NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)  
 

  
devices <- mx.cpu() 
if (length(train_y)<=1280){
	bsize=floor(length(train_y)/10)
	nround=30
	}
if (length(train_y)>1280){
	bsize=floor(length(train_y)/20)
	nround=30
	}
if (length(train_y)>10000){
	bsize=floor(length(train_y)/100)
	nround=50
	} 

 
model <- mx.model.FeedForward.create(NN_model,  
                                     X = train_array,  
                                     y = train_y,  
                                     ctx = devices,  
                                     num.round = nround,
                                     array.batch.size = bsize ,
                                     learning.rate = 0.2, 
                                     momentum = 0.9, 
                                     eval.metric = mx.metric.accuracy,  
                                     epoch.end.callback = mx.callback.log.train.metric(100))

 
predicted <- predict(model, test_array)  
predicted_labels <- max.col(t(predicted)) -1
err=predicted_labels-test_y
err_rate[ii,kk]=length(which(err!=0))/length(test_y)
pp=paste('featur1e',ii)
write.table(file=pp,data.frame(t(predicted),test_y),append=T, row.names = FALSE,col.names = FALSE,quote = FALSE)
print(ii)
}
}



