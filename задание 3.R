library('ISLR')
library('GGally')
library('MASS')

my.seed <- 123
train.percent <- 0.75
options("ggmatrix.progress.bar" = FALSE)

#загружаем данные PimaIndiansDiabetes-----
library('mlbench')
data(PimaIndiansDiabetes)
head(PimaIndiansDiabetes)

#pregnant - количество беременных
#glucose - концентрация глюкозы
#pressure - артериальное давление
#triceps - толщина кожи
#insulin - количество инсулина в крови
#mass - масса тела индекс
#pedigree - предраспололоженность к диабету
#age - возраст
#diabets - тест на диабет (pos - есть, neg - нет)

str(PimaIndiansDiabetes)

# графики разброса
ggp <- ggpairs(PimaIndiansDiabetes)
print(ggp, progress = FALSE)

# доли наблюдений в столбце default
table(PimaIndiansDiabetes$diabetes) / sum(table(PimaIndiansDiabetes$diabetes))

#отбираем наблюдения в обучающую выборку------ 
set.seed(my.seed)
inTrain <- sample(seq_along(PimaIndiansDiabetes$diabetes),
                  nrow(PimaIndiansDiabetes)*train.percent)
df <- PimaIndiansDiabetes[inTrain, ]
# фактические значения на обучающей выборке
Факт <- df$diabetes

#логистическая регрессия
model.logit <- glm(diabetes ~ glucose, data = df, family = 'binomial')
summary(model.logit)

# прогноз: вероятности принадлежности классу 'Yes' (дефолт)
p.logit <- predict(model.logit, df, type = 'response')
Прогноз <- factor(ifelse(p.logit > 0.5, 2, 1),
                  levels = c(1, 2),
                  labels = c('No', 'Yes'))

# матрица неточностей
conf.m <- table(Факт, Прогноз)
conf.m

# чувствительность
conf.m[2, 2] / sum(conf.m[2, ])

# специфичность
conf.m[1, 1] / sum(conf.m[1, ])

# верность
sum(diag(conf.m)) / sum(conf.m)

#LDA------
model.lda <- lda(diabetes ~ glucose, data = PimaIndiansDiabetes[inTrain, ])
model.lda


# прогноз: вероятности принадлежности классу 'Yes' (дефолт)
p.lda <- predict(model.lda, df, type = 'response')
Прогноз <- factor(ifelse(p.lda$posterior[, 'pos'] > 0.5, 
                         2, 1),
                  levels = c(1, 2),
                  labels = c('neg', 'pos'))
# матрица неточностей
conf.m <- table(Факт, Прогноз)
conf.m

# чувствительность
conf.m[2, 2] / sum(conf.m[2, ])

# специфичность
conf.m[1, 1] / sum(conf.m[1, ])

# верность
sum(diag(conf.m)) / sum(conf.m)

#ROC-кривая----
#LDA-model
# считаем 1-SPC и TPR для всех вариантов границы отсечения
x <- NULL    # для (1 - SPC)
y <- NULL    # для TPR
# заготовка под матрицу неточностей
tbl <- as.data.frame(matrix(rep(0, 4), 2, 2))
rownames(tbl) <- c('fact.No', 'fact.Yes')
colnames(tbl) <- c('predict.No', 'predict.Yes')
# вектор вероятностей для перебора
p.vector <- seq(0, 1, length = 501)
# цикл по вероятностям отсечения
for (p in p.vector){
  # прогноз
  Прогноз <- factor(ifelse(p.lda$posterior[, 'pos'] > p, 
                           2, 1),
                    levels = c(1, 2),
                    labels = c('neg', 'pos'))
  
  # фрейм со сравнением факта и прогноза
  df.compare <- data.frame(Факт = Факт, Прогноз = Прогноз)
  
  # заполняем матрицу неточностей
  tbl[1, 1] <- nrow(df.compare[df.compare$Факт == 'neg' & df.compare$Прогноз == 'neg', ])
  tbl[2, 2] <- nrow(df.compare[df.compare$Факт == 'pos' & df.compare$Прогноз == 'pos', ])
  tbl[1, 2] <- nrow(df.compare[df.compare$Факт == 'neg' & df.compare$Прогноз == 'pos', ])
  tbl[2, 1] <- nrow(df.compare[df.compare$Факт == 'pos' & df.compare$Прогноз == 'neg', ])
  
  # считаем характеристики
  TPR <- tbl[2, 2] / sum(tbl[2, 2] + tbl[2, 1])
  y <- c(y, TPR)
  SPC <- tbl[1, 1] / sum(tbl[1, 1] + tbl[1, 2])
  x <- c(x, 1 - SPC)
}
# строим ROC-кривую
par(mar = c(5, 5, 1, 1))
# кривая
plot(x, y, type = 'l', col = 'blue', lwd = 3,
     xlab = '(1 - SPC)', ylab = 'TPR', 
     xlim = c(0, 1), ylim = c(0, 1))
# прямая случайного классификатора
abline(a = 0, b = 1, lty = 3, lwd = 2)

#точка для вероятности 0.5
points(x[p.vector == 0.5], y[p.vector == 0.5], pch = 16)
text(x[p.vector == 0.5], y[p.vector == 0.5], 'p = 0.5', pos = 4)

# точка для вероятности 0.2
points(x[p.vector == 0.2], y[p.vector == 0.2], pch = 16)
text(x[p.vector == 0.2], y[p.vector == 0.2], 'p = 0.2', pos = 4)

Прогноз <- factor(ifelse(p.lda$posterior[, 'Yes'] > 0.2, 
                         2, 1),
                  levels = c(1, 2),
                  labels = c('No', 'Yes'))
conf.m <- table(Факт, Прогноз)
conf.m
# чувствительность
conf.m[2, 2] / sum(conf.m[2, ])
# специфичность
conf.m[1, 1] / sum(conf.m[1, ])
# верность
sum(diag(conf.m)) / sum(conf.m)