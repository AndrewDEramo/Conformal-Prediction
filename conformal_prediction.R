# Conformal Prediction
# Softmax class output

set.seed(4)
# Matrix with two samples and three classes
A = matrix(rnorm(2*3),
           nrow=2,
           ncol=3)
(probA <- exp(A)/rowSums(exp(A)))

# Labels
labels <- matrix(c(2,1))

# Scores
scores <- probA[cbind(1:2, labels)]

#############

# Create toy data
library(MASS)
set.seed(5)
class1_mean <- c(2,3)
class1_cov <- matrix(c(1,-.5,-.5,1),
                     byrow=TRUE,
                     nrow=2)
class1 <- MASS::mvrnorm(200,class1_mean,class1_cov)

class2_mean <- c(6,5)
class2_cov <- matrix(c(1,-.5,-.5,1),
                     byrow=TRUE,
                     nrow=2)
class2 <- MASS::mvrnorm(200,class2_mean,class2_cov)

class3_mean <- c(2,5)
class3_cov <- matrix(c(1,0,0,1),
                     byrow=TRUE,
                     nrow=2)
class3 <- MASS::mvrnorm(200,class3_mean,class3_cov)

df <- data.frame(
  x1 = c(class1[,1], class2[,1], class3[,1]),
  x2 = c(class1[,2], class2[,2], class3[,2]),
  class = factor(rep(c("class1", "class2", "class3"), each=200))
)
plot(df$x2 ~ df$x1,
     col=df$class,
     pch=16,
     xlab="Feature 1",ylab="Feature 2")


group <- sample(cut(
  seq(nrow(df)), #300
  nrow(df)*cumsum(c(0, rep(1/3,3))),
  labels=(c("train","calibrate","test"))
))
sets <- split(df,group)
lapply(sets, dim)

library(nnet)
mod <- nnet::multinom(class~x1+x2,
                      data=sets$train)
preds <- predict(mod, 
                 newdata=data.frame(sets$calibrate[,1:2]),
                 type="probs")
                 )

n2 <- dim(sets$calibrate)[1]
actual_probs <- preds[cbind(1:nrow(preds),
                            match(sets$calibrate$class, 
                            colnames(preds))
                            )
                      ]
(cal_scores <- 1-actual_probs)

# qhat function
find_quantile <- function(cal_scores, n2, alpha, ...){
  q_level <- ceiling((n2+1)*(1-alpha))/n2
  qhat <- quantile(cal_scores, q_level, type=3)
  return(qhat)
}

hist(cal_scores,
     main="Histogram of Calibration Scores")
alphas <- c(0.05,0.10,0.20)
q <- c()
for(i in 1:length(alphas)){
  q <- c(q,find_quantile(cal_scores=cal_scores, 
                         n2=n2,
                         alpha=alphas[i])
  )
}
abline(v=q,
       lty=2,
       col=c(2,3,4))
legend("topright",
       legend=paste("alpha =", rev(alphas)),
       col=c(2,3,4),
       lty=2)

# qhat at alpha=0.05
qhat <- find_quantile(cal_scores, n2, alpha=0.05)
test_preds <- predict(mod,
                      newdata=data.frame(sets$test[,1:2]),
                      type="probs") >= 1-qhat
head(test_preds)
# Mean width of the prediction set
rowSums(test_preds) |>
  mean()
t <- test_preds[cbind(1:nrow(test_preds),
                 match(sets$test$class,
                       colnames(test_preds))
                 )
           ]
length(which(t==TRUE))/length(t)

# qhat at alpha=0.1
qhat <- find_quantile(cal_scores, n2, alpha=0.1)
test_preds <- predict(mod,
                      newdata=data.frame(sets$test[,1:2]),
                      type="probs") >= 1-qhat
head(test_preds)
# Mean width of the prediction set
rowSums(test_preds) |>
  mean()
t <- test_preds[cbind(1:nrow(test_preds),
                      match(sets$test$class,
                            colnames(test_preds))
                      )
                ]
length(which(t==TRUE))/length(t)
