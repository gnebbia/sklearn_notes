# Inferential Statistics that can be useful in ML


## Note about p-values

The interpretation of the pvalue may follow two ways:

* **Neyman-Pearson** decision making framework. does not give an additional meaning
  to pvalue other than setting a threshold and telling if we can either reject
  or accept the null hypothesis
* **Fisher** interpretation instead, allows us to compare two p-values and allows us
  to say that in the case of a pvalue of 0.9 we are more confident in not being
  able to reject the null hypothesis with respect to a pvalue of 0.6


It's a bit messy. In practice, they appear to be used simultaneously. But
everyone implicitly claims to use Neyman-Pearson, while interpreting p-values
as some sort of strength-of-evidence.


## How is Inferential Statistics used in Machine Learning

It is a good practice to gather a population of results when comparing multiple
machine learning algorithms or when comparing the same algorithm but with
different hyperparameters or in general configurations.

A good rule of thumb is to gather 30 or more results, once we have gathered
these results we can use inferential statistics to compare models/systems.

For example we could gather 30 results and then compute the mean expected
performance. But at this point, how can we know if the differences among the
means are significant?

Inferential statistics helps us in this, and tells us how to interpret these
results.

When we gather data like this, we first want to know if it is normal or not,
with some normality test, and then if it is normal we apply some parametric
test, while if it is not normal we apply some nonparametric test.

Let's see a practical example in python:

```python
from scipy.stats import normaltest
import pandas as pd

result1 = pd.read_csv('results1.csv', header=None)
result2 = pd.read_csv('results2.csv', header=None)

# Check normality of first set of data
value, p = normaltest(results1.values)
print(value, p)
if p >= 0.05:
    print('It is probable that result1 is normal')
else:
    print('It is NOT probable that result1 is normal, we reject the null hypothesis')

# Check normality of second set of data
value, p = normaltest(result2.values)
print(value, p)
if p >= 0.05:
    print('It is likely that result2 is normal')
else:
    print('It is unlikely that result2 is normal')
```
Notice that by default scipy uses the D'Agostino and Pearsons normality test.

Now with this we can see that they are both normal, we can check their standard
deviation/variance, and if this is equal, we can use the Student t-test to
understand if both sets of results come from the same distribution (i.e., there
is no statistical difference in results). Student t-test assumes normality and 
can be applied on two samples of normal distributions which have the same
variance, in python this can be done with:
```python

from scipy.stats import ttest_ind

value, pvalue = ttest_ind(values1, values2, equal_var=True)
print(value, pvalue)
if pvalue > 0.05:
    print('Samples are probably drawn from the same distributions')
else:
    print('Samples are probably NOT drawn from different distributions')

```
If the variances and hence standard deviations are different, we can use a
modified version of the Student t-test which is called Welch's t-test.
In python we can do this with:

```python
from scipy.stats import ttest_ind

value, pvalue = ttest_ind(values1, values2, equal_var=False)
print(value, pvalue)
if pvalue > 0.05:
    print('Samples are probably drawn from the same distributions')
else:
    print('Samples are probably NOT drawn from different distributions')
```

Notice that: The closer two distributions are, the larger the sample we need 
in these kind of tests to tell them apart.

We can show this effect by calculating the statistical test on different sized
sub-samples of each set of results and plotting the p-values vs the sample
size.

We would expect the p-value to get smaller with the increase sample size. We can
also draw a line at the 95% level (0.05) and show at what point the sample size
is large enough to indicate these two populations are significantly different.

While if we have two samples which are not normal, in order to understand if
they come from the same distribution, we can use a nonparametric test called 
'2 samples Kolmogorov-Smirnov test'.

```python
from scipy.stats import ks_2samp

value, pvalue = ks_2samp(values1, values2)
print(value, pvalue)
if pvalue > 0.05:
    print('Samples are likely drawn from the same distributions (fail to reject H0)')
else:
    print('Samples are likely drawn from different distributions (reject H0)
```

Notice that: This test can be used on Gaussian data, but will have less
statistical power and may require large samples.


## P-value

What Are P Values?

P values evaluate how well the sample data support the devil’s advocate argument that the null 
hypothesis is true. It measures how compatible your data are with the null hypothesis. 
How likely is the effect observed in your sample data if the null hypothesis is true?

* High P values: your data are likely with a true null.
* Low P values: your data are unlikely with a true null.

A low P value suggests that your sample provides enough evidence that you can reject the 
null hypothesis for the entire population.

### How Do You Interpret P Values?

In technical terms, a P value is the probability of obtaining an effect at least as extreme as 
the one in your sample data, assuming the truth of the null hypothesis.
For example, suppose that a vaccine study produced a P value of 0.04. This P value indicates 
that if the vaccine had no effect, you’d obtain the observed difference or more in 4% of 
studies due to random sampling error.

P values address only one question: how likely are your data, assuming a true null hypothesis?
It does not measure support for the alternative hypothesis. This limitation leads 
us into the next section to cover a very common misinterpretation of P values.

### When do we reject null hypothesis?

We define alpha as 1 - p-value and call it 'level of significance'.

Let's make some example:
* For results with a 90% level of confidence, the value of alpha is 1 - 0.90 = 0.10.
* For results with a 95% level of confidence, the value of alpha is 1 - 0.95 = 0.05.
* For results with a 99% level of confidence, the value of alpha is 1 - 0.99 = 0.01.
* And in general, for results with a C% level of confidence, the value of alpha is 1 – C/100.

The rejection of the null hypothesis depends on the choice of alpha, and this choice
depends on the domain knowledge, for example physics experiment generally prefer a very high alpha
while social studies may keep a lower alpha.





## 2 Sample Kolmogorov Smirnov Test

This can be used to check if a sample is representative of a specific
population, or better, if two sample come from the same population.

Key facts about the Kolmogorov-Smirnov test

* The two sample Kolmogorov-Smirnov test is a nonparametric test that compares the 
cumulative distributions of two data sets(1,2).

* The test is nonparametric. It does not assume that data are sampled from Gaussian 
distributions (or any other defined distributions).

* The results will not change if you transform all the values to logarithms or reciprocals or any 
transformation. The KS test report the maximum difference between the two cumulative distributions,
and calculates a P value from that and the sample sizes. A transformation will stretch (even 
rearrange if you pick a strange transformation) the X axis of the frequency distribution, 
but cannot change the maximum distance between two frequency distributions.

* Converting all values to their ranks also would not change the maximum difference between the 
cumulative frequency distributions (pages 35-36 of Lehmann, reference 2). Thus, although the 
test analyzes the actual data, it is equivalent to an analysis of ranks. Thus the test is 
fairly robust to outliers (like the Mann-Whitney test).

* The null hypothesis is that both groups were sampled from populations with identical distributions. It tests for any violation of that null hypothesis -- different medians, different variances, or different distributions.

* Because it tests for more deviations from the null hypothesis than does the Mann-Whitney test, 
it has less power to detect a shift in the median but more power to detect changes in the shape of the distributions (Lehmann, page 39).

* Since the test does not compare any particular parameter (i.e. mean or median), 
it does not report any confidence interval.

* Don't use the Kolmogorov-Smirnov test if the outcome (Y values) are categorical, with many ties. 
Use it only for ratio or interval data, where ties are rare.

* The concept of one- and two-tail P values only makes sense when you are looking at an outcome that 
has two possible directions (i.e. difference between two means). Two cumulative distributions can differ 
in lots of ways, so the concept of tails is not really appropriate. the P value reported by 
Prism essentially has many tails. Some texts call this a two-tail P value.
Interpreting the P value

### Interpretation of the P value

The P value is the answer to this question:

If the two samples were randomly sampled from identical populations, what is the probability that the 
two cumulative frequency distributions would be as far apart as observed? More precisely, what is 
the chance that the value of the Komogorov-Smirnov D statistic would be as large or larger than observed?

If the P value is small, conclude that the two groups were sampled from populations with different distributions.
The populations may differ in median, variability or the shape of the distribution. 

### Performing the 2 Sample KS test in Python

```python
from scipy.stats import ks_2samp
ks = []
pvalues = []
for i in range(ds.shape[1]):
    ks_i, pvalue_i = ks_2samp(sample.iloc[:,i], ds.iloc[:,i])
    ks.append(ks_i)
    pvalues.append(pvalue_i)

print(pvalues)
```


## Tests on Variance (Homoscedasticity and Heteroscedasticity)

Certain tests (e.g. ANOVA) require that the variances of different populations
are equal. This can be determined by the following approaches:

Comparison of graphs (esp. box plots)
Comparison of variance, standard deviation and IQR statistics
Statistical tests

The F test presented in Two Sample Hypothesis Testing of Variances
can be used to determine whether the variances of two populations
are equal. For three or more variables the following statistical
tests for homogeneity of variances are commonly used:

* Bartlett’s test, for normal distributions
* Levene’s test, for not strictly normal distributions
* Fligner Killeen test, nonparametric tests

Levene's test is an alternative to the Bartlett test. The Levene test is less
sensitive than the Bartlett test to departures from normality. If you have
strong evidence that your data do in fact come from a normal, or nearly normal,
distribution, then Bartlett's test has better performance. 
The Fligner Killeen test is a non-parametric test for homogeneity of group
variances based on ranks. It is useful when the data is non-normal or where
there are outliers. We present the median-centering version of this test.



