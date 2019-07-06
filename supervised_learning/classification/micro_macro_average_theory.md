
In Micro-average method, you sum up the individual true positives, false positives, and false negatives of the system for different sets and the apply them to get the statistics.

Tricky, but I found this very interesting. There are two methods by which you can get such average statistic of information retrieval and classification.
1. Micro-average Method

In Micro-average method, you sum up the individual true positives, false positives, and false negatives of the system for different sets and the apply them to get the statistics. For example, for a set of data, the system's

True positive (TP1)  = 12
False positive (FP1) = 9
False negative (FN1) = 3

Then precision (P1) and recall (R1) will be 57.14%=TP1TP1+FP1
and 80%=TP1TP1+FN1

and for a different set of data, the system's

True positive (TP2)  = 50
False positive (FP2) = 23
False negative (FN2) = 9

Then precision (P2) and recall (R2) will be 68.49 and 84.75

Now, the average precision and recall of the system using the Micro-average method is

Micro-average of precision=TP1+TP2TP1+TP2+FP1+FP2=12+5012+50+9+23=65.96

Micro-average of recall=TP1+TP2TP1+TP2+FN1+FN2=12+5012+50+3+9=83.78

The Micro-average F-Score will be simply the harmonic mean of these two figures.
2. Macro-average Method

The method is straight forward. Just take the average of the precision and recall of the system on different sets. For example, the macro-average precision and recall of the system for the given example is

Macro-average precision=P1+P22=57.14+68.492=62.82
Macro-average recall=R1+R22=80+84.752=82.25

The Macro-average F-Score will be simply the harmonic mean of these two figures.

Suitability Macro-average method can be used when you want to know how the system performs overall across the sets of data. You should not come up with any specific decision with this average.

On the other hand, micro-average can be a useful measure when your dataset varies in size.



#### WHAT TO USE? MICRO OR MACRO F1 SCORE? 


* If you think all the labels are more or less equally sized (have roughly the same number of instances), use any.

* If you think there are labels with more instances than others and if you want to bias your metric towards the most populated ones, use micromedia. 

* If you think there are labels with more instances than others and if you want to bias your metric toward the least populated ones (or at least you don't want to bias toward the most populated ones), use macromedia.

* If the micromedia result is significantly lower than the macromedia one, it means that you have some gross misclassification in the most populated labels, whereas your smaller labels are probably correctly classified.

* If the macromedia result is significantly lower than the micromedia one, it means your smaller labels are poorly classified, whereas your larger ones are probably correctly classified.
