(LABR - Arabic Book Reviews
ASTD - Arabic Sentiment Tweets 
IMDB Movie Reviews
TESA - Twitter Entity Sentiment Analysis)




LABR :
analyse avant preparation , 
preparation de fichier, (elimination de class 3 (neutre))
analyse 
pretraitement,
balancing,
vectorisation,

   -> creation de 4 fichier , normal ,normal+balanced, preproceced , preproceced+balanced 
aplication de 8 algorithms de ML .. 

Multinomial Naive Bayes
Complement Naive Bayes	
SVM (Linear)
SVM (NuSVC)
LogisticRegression
RandomForest
KNN
AdaBoostClassifier

	resultat pour chaque algorithme / chaque 3 fichier : 
		"Confusion Matrix"
   	 	"Accuracy"
        	"Precision"
        	"Recall"
        	"Specificity"
        	"F1 Score"
        	"False Positive Rate"
        	"False Negative Rate"
        	"Negative Predictive Value"
        	"Number of Positive Samples"
        	"Number of Negative Samples"

analyse de changement de F1 en fonction de 'parametre' pour chaque algorithme

(parametre :
Algorithm		Important Parameter(s)	Purpose					Typical Values

MultinomialNB		alpha			Smoothing				0.0001 to 10

ComplementNB		alpha			Same as MNB				0.0001 to 10

SVC (Linear)		C			Regularization Strength			0.01 to 100

NuSVC			nu			Max Training Error / Support Vectors %	0.01 to 0.5

LogisticRegression	C			Regularization Strength			0.01 to 100

RandomForestClassifier	n_estimators		Number of Trees				50 to 500

RandomForestClassifier	max_depth		Depth of Trees	None or 		5 to 50

KNeighborsClassifier	n_neighbors		Number of Neighbors			3 to 20

AdaBoostClassifier	n_estimators		Number of Weak Learners			50 to 500

AdaBoostClassifier	learning_rate		Step Size				0.01 to 2
)


sauvegarde tt les codes/resultat dans LABR.html 



IMDB : 
 -------------------- pretraitement..

TESA : 
-----------------------praitraitement.. 

ASTD (same as LABR  .. delete classe : neutre,irrelevant) : 
COMPLETED 
