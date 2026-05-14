# install dependencies (will only install them for the first time)
list.of.packages = c(
    "devtools", "lme4", "parallel", "cAIC4", "magrittr","dplyr",
    "mgcv", "MASS", "lsei", "refund","stringr", "Matrix", "mvtnorm", 
    "arrangements", "progress", "ggplot2", "gridExtra", "here", "Rfast"
)
new.packages = list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)){
  print("Installing new packages:")
  print(new.packages)
  chooseCRANmirror(ind=75)
  install.packages(new.packages, dependencies = TRUE)
}else{
  print("All dependencies already installed!")
}

if(!("fastFMM" %in% installed.packages()[,"Package"])){
  print("Installing fastFMM")
  chooseCRANmirror(ind=75)
  install.packages("fastFMM", dependencies = TRUE)
}else{
  print("fastFMM already installed!")
}