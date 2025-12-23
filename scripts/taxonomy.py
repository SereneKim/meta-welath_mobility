import graph

short_names = {
    "Regression‐based Measures": "M1",
    "Rank‐based Measures": "M2",
    "Transition Matrix / Probability Measures": "M3",
    "Absolute Mobility Measures": "M4",
    "Multigenerational Measures": "M5",
    "Decomposition / Structural Approaches": "M6",
    "Non‐parametric Approaches": "M7",
    "Others_Measure": "M8",
    "Panel/Longitudinal Surveys": "D1",
    "Administrative/Registry Data": "D2",
    "National Survey Data": "D3",
    "Opportunity Atlas": "D4",
    "Natural/Experimental Data": "D5",
    "Linked Administrative Data": "D6",
    "International Panel Data": "D7",
    "Rich List Data": "D8",
    "University/Institution Data" : "D9",
    "Pseudo-Panel/Household Budget Survey": "D10",
    "Archival/Historical Data": "D11",
    "Big Data": "D12",
    "No dataset": "D13",
    "Others_DataType": "D14",
    "Measurement and Methodological Advances": "R1",
    "Empirical Estimates and Determinants": "R2",
    "Policy, Institutional, and Geographic Impacts": "R3",
    "Intergenerational Wealth Mobility and Inheritance": "R4",
    "Demographic Differences in Mobility (Race, Gender, etc.)": "R5",
    "Mobility and Non-Income Outcomes (Health, Wellbeing, etc.)": "R6",
    "Theoretical and Structural Models": "R7",
    "Perceptions of Mobility and Attitudes": "R8",
    "Others_RqType": "R9"} 

short_names = {graph.norm(k): v for k, v in short_names.items()}


NODE_HIGHLIGHTS = {
    "Intergenerational Wealth Mobility and Inheritance": "R4",
    "Regression-based Measures": "M1",
    "Empirical Estimates and Determinants": "R2",
    "No dataset": "D13",
    "Panel/Longitudinal Surveys": "D1"
}

PAIR_HIGHLIGHTS = {
    'Regression‐based Measures - Intergenerational Wealth Mobility and Inheritance': "M1-R4",
    'Regression‐based Measures - Empirical Estimates and Determinants': "M1-R2",
    'Regression‐based Measures - Panel/Longitudinal Surveys': "M1-D1",
    'Panel/Longitudinal Surveys - Intergenerational Wealth Mobility and Inheritance': "D1-R4",
    'Panel/Longitudinal Surveys - Empirical Estimates and Determinants': "D1-R2"
}

TRIANGLE_HIGHLIGHTS = {
    '[Panel/Longitudinal Surveys, Regression‐based Measures, Empirical Estimates and Determinants]': "D1-M1-R2",
    '[Panel/Longitudinal Surveys, Regression‐based Measures, Intergenerational Wealth Mobility and Inheritance]': "D1-M1-R4",
    '[Linked Administrative Data, Regression‐based Measures, Intergenerational Wealth Mobility and Inheritance]': "D6-R4-M1",
    '[National Survey Data, Regression‐based Measures, Empirical Estimates and Determinants]': "D3-M1-R2",
    '[National Survey Data, Regression‐based Measures, Intergenerational Wealth Mobility and Inheritance]': "D3-M1-R4"
}

NODE_HIGHLIGHTS = {graph.norm(k): v for k, v in NODE_HIGHLIGHTS.items()}
PAIR_HIGHLIGHTS = {graph.norm(k): v for k, v in PAIR_HIGHLIGHTS.items()}
TRIANGLE_HIGHLIGHTS = {graph.norm(k): v for k, v in TRIANGLE_HIGHLIGHTS.items()}


period_order = ["-2000", "2001-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2025"]
period_labels = {
    "-2000": "-2000 (T1)",
    "2001-2005": "2001-2005 (T2)",
    "2006-2010": "2006-2010 (T3)",
    "2011-2015": "2011-2015 (T4)",
    "2016-2020": "2016-2020 (T5)",
    "2021-2025": "2021-2025 (T6)",
}
