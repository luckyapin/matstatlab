import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import math


def normal_interval(n, n1, n2, m, mu1, mu2, sigma1, sigma2, a):
    """
    Computes the proportion of times the difference between two normal distributions
    falls within the confidence interval based on repeated sampling.

    Parameters:
    n (int): Number of samples for each distribution.
    n1 (int): Sample size for the first distribution.
    n2 (int): Sample size for the second distribution.
    m (int): Number of iterations for the simulation.
    mu1 (float): Mean of the first normal distribution.
    mu2 (float): Mean of the second normal distribution.
    sigma1 (float): Standard deviation of the first normal distribution.
    sigma2 (float): Standard deviation of the second normal distribution.
    a (float): Significance level for the confidence interval (e.g., 0.05 for 95% CI).

    Returns:
    float: Proportion of times the true difference (mu1 - mu2) falls within the confidence interval.
    """
    c = 0
    for i in range(m):
        sample1 = np.random.normal(mu1, sigma1, n)
        sample2 = np.random.normal(mu2, sigma2, n)
        q = stats.t.ppf(1 - a / 2, n - 1)
        s1 = np.var(sample1)
        s2 = np.var(sample2)
        x1 = np.mean(sample1)
        x2 = np.mean(sample2)
        t_l = (
            ((((n1 + n2) * (n1 * s1 + n2 * s2)) / (n1 * n2 * (n1 + n2 - 2))) ** 0.5) * q
            + x1
            - x2
        )
        t_r = (
            ((((n1 + n2) * (n1 * s1 + n2 * s2)) / (n1 * n2 * (n1 + n2 - 2))) ** 0.5)
            * (-q)
            + x1
            - x2
        )

        if t_l >= mu1 - mu2 >= t_r:
            c += 1

    return c / m


def poisson_interval(l, alpha, n, m):
    """
    Computes the proportion of times the true value (l^2 + l) falls within the
    confidence interval based on repeated sampling from a Poisson distribution.

    Parameters:
    l (float): The lambda parameter for the Poisson distribution.
    alpha (float): Significance level for the confidence interval (e.g., 0.05 for 95% CI).
    n (int): Number of samples drawn from the Poisson distribution.
    m (int): Number of iterations for the simulation.

    Returns:
    float: Proportion of times the true value falls within the confidence interval.
    """

    c = 0
    for _ in range(m):
        sample = np.random.poisson(l, n)
        moment = np.mean(sample**2)
        std = np.std(sample**2) / np.sqrt(n)
        q = stats.norm.ppf(1 - alpha / 2)
        l_l = moment - q * std
        l_r = moment + q * std
        if l_l <= l**2 + l and l**2 + l <= l_r:
            c += 1

    return c / m


def analyze_bmi_distribution(db):
    """
    Analyzes the BMI distribution from the given DataFrame.
    The function performs the following tasks:
    - Groups the BMI values into bins.
    - Calculates the mean and standard deviation of the BMI.
    - Computes the expected frequency of BMI values under a normal distribution.
    - Performs a Chi-squared goodness-of-fit test to compare observed and expected frequencies.

    Parameters:
    db (pd.DataFrame): DataFrame containing a 'bmi' column.

    Returns:
    str: Result of the Chi-squared test.
    """

    bmi = db["bmi"]
    n_groups = 20
    bmi_groups = np.linspace(min(bmi), max(bmi), n_groups + 1)

    # Count the number of BMI values in each group
    bmi_count = []
    for i in range(len(bmi_groups) - 1):
        lower_bound = bmi_groups[i]
        upper_bound = bmi_groups[i + 1]
        count = np.sum((bmi >= lower_bound) & (bmi <= upper_bound))
        bmi_count.append(count)

    # Calculate midpoints of BMI groups
    bmi_middle_groups = [
        (bmi_groups[i] + bmi_groups[i + 1]) / 2 for i in range(len(bmi_groups) - 1)
    ]

    # Calculate mean and standard deviation of BMI
    n = sum(bmi_count)
    bmi_mean = (
        sum(
            [bmi_middle_groups[i] * bmi_count[i] for i in range(len(bmi_middle_groups))]
        )
        / n
    )
    bmi_dispersion = (
        sum(
            [
                bmi_middle_groups[i] ** 2 * bmi_count[i]
                for i in range(len(bmi_middle_groups))
            ]
        )
        / n
        - bmi_mean**2
    )
    bmi_std = bmi_dispersion**0.5

    # Calculate expected frequencies under a normal distribution
    h = bmi_groups[1] - bmi_groups[0]
    p = 1 / (math.pi * 2) ** 0.5
    expected = [
        (p * math.exp(-(((i - bmi_mean) / bmi_std) ** 2) / 2) * h * n / bmi_std)
        for i in bmi_middle_groups
    ]

    # Perform Chi-squared goodness-of-fit test
    m = len(bmi_middle_groups)
    a = 0.05
    chi_nabl = [
        (bmi_count[i] - expected[i]) ** 2 / expected[i]
        for i in range(len(bmi_middle_groups))
    ]
    chi_nabl_sum = sum(chi_nabl)
    chi_exp = stats.chi2.ppf(1 - a, m)

    # Return the result of the Chi-squared test
    if chi_nabl_sum < chi_exp:
        return "Нет оснований отвергнуть"  # No reason to reject
    else:
        return "Отвергаем"  # Reject the null hypothesis


def analyze_bmi_by_smoker_status(db):
    """
    Analyzes the BMI distribution based on smoking status and performs a chi-squared test
    to determine if there is a significant difference in BMI distributions between smokers and non-smokers.

    Parameters:
    db (pd.DataFrame): DataFrame containing the data with 'bmi' and 'smoker' columns.

    Returns:
    str: Result of the chi-squared test indicating whether to reject or accept the null hypothesis.
    """
    # Separate BMI values by smoking status
    yes = list(db[db["smoker"] == "yes"]["bmi"])
    no = list(db[db["smoker"] == "no"]["bmi"])

    n_groups = 10
    bmi_groups = np.linspace(min(db["bmi"]), max(db["bmi"]), n_groups + 1)

    yes_count = []
    no_count = []

    # Count occurrences of BMI values in each group for smokers and non-smokers
    for i in range(len(bmi_groups) - 1):
        lower_bound = bmi_groups[i]
        upper_bound = bmi_groups[i + 1]

        count_yes = sum(lower_bound <= value <= upper_bound for value in yes)
        count_no = sum(lower_bound <= value <= upper_bound for value in no)

        yes_count.append(count_yes)
        no_count.append(count_no)

    yes_count_abs = [count / len(yes) for count in yes_count]
    no_count_abs = [count / len(no) for count in no_count]

    # Total counts for each group
    yes_no = [yes_count[i] + no_count[i] for i in range(n_groups)]
    chi = [
        len(yes) * len(no) * (yes_count_abs[i] - no_count_abs[i]) ** 2 / yes_no[i]
        for i in range(n_groups)
    ]

    # Perform chi-squared test
    p = stats.chi2.sf(sum(chi), n_groups - 1)

    # Return the result of the hypothesis test
    if p < 0.05:
        return "Отвергаем"
    else:
        return "Принимаем"


def analyze_bmi_by_gender(db):
    """
    Analyzes the distribution of BMI by gender using Chi-Squared test.

    Parameters:
    db (DataFrame): DataFrame containing 'bmi' and 'sex' columns.

    Returns:
    str: Result of the Chi-Squared test, either 'Отвергаем' or 'Принимаем'.
    """
    male = list(db[db["sex"] == "male"]["bmi"])
    female = list(db[db["sex"] == "female"]["bmi"])

    n_groups = 10
    bmi_groups = np.linspace(min(db["bmi"]), max(db["bmi"]), n_groups + 1)

    male_count = []
    female_count = []

    for i in range(len(bmi_groups) - 1):
        lower_bound = bmi_groups[i]
        upper_bound = bmi_groups[i + 1]

        male_count.append(sum(lower_bound <= value <= upper_bound for value in male))
        female_count.append(
            sum(lower_bound <= value <= upper_bound for value in female)
        )

    male_female = [male_count[i] + female_count[i] for i in range(n_groups)]
    male_sum = sum(male_count)
    female_sum = sum(female_count)
    n = male_sum + female_sum

    male_exp = [male_sum * male_female[i] / n for i in range(n_groups)]
    female_exp = [female_sum * male_female[i] / n for i in range(n_groups)]

    male_oe = [
        (male_count[i] - male_exp[i]) ** 2 / male_exp[i] for i in range(n_groups)
    ]
    female_oe = [
        (female_count[i] - female_exp[i]) ** 2 / female_exp[i] for i in range(n_groups)
    ]

    chi = sum(male_oe) + sum(female_oe)
    p = stats.chi2.sf(chi, (n_groups - 1) * 1)

    return "Отвергаем" if p < 0.05 else "Принимаем"
