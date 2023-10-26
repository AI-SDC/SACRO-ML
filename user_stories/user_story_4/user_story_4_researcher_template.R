##**********************************************************************
##  Example of model to be released in R                            ####
##**********************************************************************

# Set random seed ####
set.seed(39295)

# Load data ####
data(infert)

# Add extra columns of noise to data ####
n_add <- 2 # Add this many rows
ndata <- nrow(infert)
cnames <- colnames(infert)
for (i in 1:n_add) infert <- cbind(infert, rnorm(nrow(infert)))
colnames(infert) <- c(cnames, paste0("X", 1:n_add))

# Proportion of samples in training set ####
p_train <- 0.5

# Establish indices of samples in training and test sets ####
n_train <- round(ndata * p_train)
n_test <- ndata - n_train
index_train <- sample(ndata, n_train)
index_test <- setdiff(1:ndata, index_train)

# Split dataset into training and test sets ####
infert_train <- infert[index_train, ]
infert_test <- infert[index_test, ]

# Overfitted logistic regression model ####
model <- suppressWarnings(glm(case ~ (education + age + .)^2,
                              data = infert_train,
                              family = binomial(link = "logit")))

# Predictions on training and test sets ####
pred_train <- suppressWarnings(predict(model, infert_train, type = "response"))
pred_test <- suppressWarnings(predict(model, infert_test, type = "response"))

# Plot CDFs ####
pdf("cdf.pdf", width = 4, height = 4)
plot(0, type = "n",
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Quantile", ylab = "Value")
lines((1:n_train) / (1 + n_train), sort(pred_train))
lines((1:n_test) / (1 + n_test), sort(pred_test), col = "red")
legend("topleft", c("Train", "Test"), lty = 1, col = c("black", "red"))
dev.off()

# Define matrices of train and test data ####
output_train <- cbind(index_train, infert_train$case, pred_train)
colnames(output_train) <- c("rowID", "true_label", "probability")
output_test <- cbind(index_test, infert_test$case, pred_test)
colnames(output_test) <- c("rowID", "true_label", "probability")

# Save data to file (in working directory) ####
write.csv(output_train, file = "./output_train.csv", row.names = FALSE)
write.csv(output_test, file = "./output_test.csv", row.names = FALSE)

# Save model to file (in working directory) ###
saveRDS(model, file = "./model.rds")
