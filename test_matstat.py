import unittest
import matstat
import pandas as pd


class TestIntervalFunction(unittest.TestCase):

    def test_interval_proportion(self):
        n = n1 = n2 = 1000
        m = 1000
        mu1 = 2
        mu2 = 1
        sigma1 = 1
        sigma2 = 1
        a = 1 - 0.95
        proportion = matstat.normal_interval(n, n1, n2, m, mu1, mu2, sigma1, sigma2, a)

        self.assertAlmostEqual(proportion, 0.95, delta=0.05)

    def test_poisson_interval_proportion(self):
        l = 1
        alpha = 0.05
        n = 1000
        m = 1000
        proportion = matstat.poisson_interval(l, alpha, n, m)

        self.assertAlmostEqual(proportion, 0.95, delta=0.05)

    def test_analyze_bmi_distribution(self):
        db = pd.read_csv("sex_bmi_smokers.csv")
        result = matstat.analyze_bmi_distribution(db)
        expected_result = "Нет оснований отвергнуть"

        self.assertEqual(result, expected_result)

    def test_analyze_bmi_by_smoker_status(self):
        db = pd.read_csv("sex_bmi_smokers.csv")
        result = matstat.analyze_bmi_by_smoker_status(db)
        expected_result = "Принимаем"

        self.assertEqual(result, expected_result)

    def test_analyze_bmi_by_gender(self):
        db = pd.read_csv("sex_bmi_smokers.csv")
        result = matstat.analyze_bmi_by_gender(db)
        self.assertEqual(result, "Принимаем")


if __name__ == "__main__":
    unittest.main()
