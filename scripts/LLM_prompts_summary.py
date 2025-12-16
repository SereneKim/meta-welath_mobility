
"""
GPT Prompts Used to Filter Irrelevant Papers
"""

# 1. Empirical & Theoretical 
gpt_model = "gpt-4o-2024-08-06" # "gpt-4o-mini-2024-07-18" # (Also tried on this)
messages=[
	{
		"role": "assistant", 
		"content": f"""You're an helpful assistant who can recognize whether this paper {url} studies intergenerational mobility in wealth, income or earnings empirically with real-life datasets or theoretically by delving into the models to measure the mobility. This is the abstract of the paper: {abstract}"""
	},
	{
		"role": "user", 
		"content": 
				"""
				If the study is empirical return a term: "empirical".
				If the study is theoretical return a term: "theoretical".
				If the study is both empirical and theoretical return a term: "both".
				If the study is not at all about intergenerational wealth mobility return a term: "false".
				```plaintext
				"""
	}
]

# Filtered out "false" or "None"

# 2. 4 Questions regarding the relevance to our RQ 

from pydantic import BaseModel


class GetInfo(BaseModel):
    Q1: str
    Q1_1: str
    Q2: str
    Q2_1: str
    Q2_2: str
    Q3: str
    Q4: str


gpt_model = "o3-mini"

messages=[
	{ 
	"role": "system",
	"content": "Extract relevant data."
	
	},
	{
		"role": "user", 
		"content":f"""
				Consider this academic paper in the Social Sciences domain: 
				- Title: {row.title}
				- Year: {row.year}
				- Abstract: {row.abstract_sm}
				- DOI: {row.doi}

				Use this information to answer the following questions. You can try to access the full paper using the DOI provided.:
				Q1. Does the paper discuss inter/multigenerational wealth/income/earning mobility? Yes or No.
				Q1_1. If 'yes' in Q1, give me a sentence or two about what the paper says about inter/multigenerational wealth/income/earning mobility. If 'no' in Q1, what is the main focus of the paper?
				Q2. If 'yes' in Q1, does this paper mention some measures to study inter/multigenerational wealth/income/earning mobility? Yes or No. 
				Q2_1. If 'yes' in Q2, what are the measures mentioned in the paper? If 'no' in Q2, return 'N/A'.
				Q2_2. If 'no' in Q2, give me a reason. If 'yes' in Q2, return 'N/A'.
				Q3. If 'yes' in Q2, does this paper empirically study inter/multigenerational wealth/income/earning mobility by applying the measures to some real data? Yes or No. 
				Q4. If 'No' in Q3, does this paper theoretically study the measures of inter/multigenerational wealth/income/earning mobility? Yes or No. 

				Answer with 'Yes' or 'No', or if irrelevant, 'N/A', for each main questions (Q1,Q2,Q3,Q4). For subquestions (Q1_1, Q2_1, Q2_2), provide a brief explanation. Don't hallucinate. 
				"""
	}
]


## Q2 == 'Yes' were selected

# 3. (After generating all the taxonomies) Last check ---> After this, we arrive at 617 papers
from pydantic import BaseModel

class GetCat(BaseModel):
    Q1: str
    Q2: str


messages=[
		{ 
		"role": "assistant",
		"content": "You are a research assistant in the Social Sciences domain. You are going to read an abstract of a paper that presumably studied inter/multi generational wealth/income/earning mobility. You are going to help me, the researcher, categorize the paper based on its title and abstract."
		
		},
		{
			"role": "user", 
			"content":f"""
					Read ```Title and ```Abstract of the paper below.
					```Title
					{row['title']}
					```                 
					```Abstract
					{abs_clean}
					```
					Answer the following questions with 'Yes' or 'No':
					Q1. Does the paper study inter/multi generational wealth/income/earning mobility? Not mobility in social status, occupation, or other categorical variables.
					Q2. Is the inter/multi generational wealth/income/earning mobility core to the paper? Being core to the paper means that the paper's main research question is about inter/multi generational wealth/income/earning mobility, rather than using the mobility as a variable without discussing how it is calculated e.g., to see a relationship between inequality and mobility, or an impact of mobility on other variables, or an impact of some variables on mobility. In brief, the paper uses an empirical or synthetic dataset to study it or the paper studies a methodology to study it theoretically.
				"""
		}
	]


"""
Category Generation (Taxonomy Generation)
Using 742 Papers (where Q2 == 'Yes')
"""

# 1) Measures
gpt_model = "o3-mini" 
messages=[
	{ 
	"role": "assistant",
	"content": "You are a research assistant in the Social Sciences domain. I, the researcher, have a list of abstracts of papers that studied inter/multi generational wealth/income/earning mobility. I want you to categorize the measures that the papers used based on the abstracts."
	
	},
	{
		"role": "user", 
		"content":f"""                        
				```Abstracts
				{json_abs}
				```
				First, find the representative categories of the measures that the papers used based on the abstracts. Here, 'measures' mean an estimate from a model or an equation that quantifies the inter/multi generational wealth/income/earning mobility.
				Then, return the categories and lists of id's that belong to each in a JSON format.
				For the id's that do not belong to any category, return them in a separate category named 'Others'.
				"""
	}
]

# 2) Data Types
messages=[
	{ 
	"role": "assistant",
	"content": "You are a research assistant in the Social Sciences domain. I, the researcher, have a list of abstracts of papers that studied inter/multi generational wealth/income/earning mobility empirically. I want you to categorize the datasets that the papers used based on the abstracts."
	
	},
	{
		"role": "user", 
		"content":f"""                        
				```Abstracts
				{json_abs}
				```
				First, find the representative categories of datasets that the papers used based on the abstracts. 
				Then, return the categories and lists of id's that belong to each in a JSON format.
				For the id's without datasets, return them in a separate category named 'No dataset'.
				For the id's that do not belong to any category, return them in a separate category named 'Others'.
				"""
	}
]

# 3) Research Question Types
messages=[
            { 
            "role": "assistant",
            "content": "You are a research assistant in the Social Sciences domain. I, the researcher, have a list of abstracts of papers that studied inter/multi generational wealth/income/earning mobility. I want you to categorize the research questions that the papers aimed to answer based on the abstracts."
            
            },
            {
                "role": "user", 
                "content":f"""                        
                        ```Abstracts
                        {json_abs}
                        ```
                        First, find the representative categories of research questions that the papers aimed to answer based on the abstracts. 
                        Then, return the categories and lists of id's that belong to each in a JSON format.
                        For the id's that do not belong to any category, return them in a separate category named 'Others'.
                        """
            }
        ]




"""
Category Assignment (To each paper, one by one)
"""



# 1) Measures
from pydantic import BaseModel

class GetCat(BaseModel):
    category: int
    measure: str
    
    
messages=[
	{ 
	"role": "assistant",
	"content": "You are a research assistant in the Social Sciences domain. I, the researcher, have an abstract of a paper that studied inter/multi generational wealth/income/earning mobility. I want you to categorize the measure(s) that the paper used based on the abstract."
	
	},
	{
		"role": "user", 
		"content":f"""                        
				```Abstracts
				{json_abs[str(i)]}
				```
				First, recognize what measure(s) the papers used based on the abstracts. Here, 'measures' mean an estimate from a model or an equation that quantifies the inter/multi generational wealth/income/earning mobility.
				Does it belong to any of the following categories?
				1. Regression‐based Measures 
				2. Rank‐based Measures
				3. Transition Matrix / Probability Measures
				4. Absolute Mobility Measures
				5. Multigenerational Measures
				6. Decomposition / Structural Approaches
				7. Non‐parametric Approaches
				8. Others
				
				Return the category that the measure(s) belong to as a number, and the exact name of the measure(s) if possible. For example, if it belongs to '1', return also the name of the measure(s) such as 'Intergenerational Elasticity (IGE)'. 
				Don't repeat the name of category here like 'Regression‐based Measures - IGE'.
				If the measure(s) do not belong to any category, return '8' and the exact name of the measure.
				"""
	}
]


# 2) Data Types
messages=[
		{ 
		"role": "assistant",
		"content": "You are a research assistant in the Social Sciences domain. I, the researcher, have an abstract of a paper that studied inter/multi generational wealth/income/earning mobility. I want you to categorize the datasets that the paper used based on the abstract."
		
		},
		{
			"role": "user", 
			"content":f"""                        
					```Abstracts
					{abs_clean}
					```
					First, recognize the types of datasets the papers used based on the abstracts. 
					Does it belong to any of the following categories?
					1.	Panel/Longitudinal Surveys
					2.	Administrative/Registry Data
					3.	National Survey Data
					4.	Opportunity Atlas
					5.	Natural/Experimental Data
					6.	Linked Administrative Data
					7.	International Panel Data
					8.	Rich List Data
					9.	University/Institution Data
					10.	Pseudo-Panel/Household Budget Survey
					11.	Archival/Historical Data
					12.	Big Data
					13.	No dataset
					14.	Others
					
					Return the category that the type(s) belong to as a number, and the exact dataset name if possible. For example, if it belongs to '1', return also the name such as '2015 UK Longitudinal Survey on Wealth Mobility'. 
					Don't repeat the name of category here like 'Panel/Longitudinal Surveys - 2015 UK Longitudinal Survey on Wealth Mobility'.
					If the dataset does not belong to any category, return '14' and the exact dataset name.
					If the paper does not use any dataset, return '13' and N/A as the exact dataset name.
					"""
		}
	]

# 3) Research Question Types
messages=[
                { 
                "role": "assistant",
                "content": "You are a research assistant in the Social Sciences domain. I, the researcher, have an abstract of a paper that studied inter/multi generational wealth/income/earning mobility. I want you to categorize the research questions that the paper aimed to answer based on the abstract."
                
                },
                {
                    "role": "user", 
                    "content":f"""                        
                            ```Abstracts
                            {abs_clean}
                            ```
                            First, recognize the types of research questions the papers aimed to answer based on the abstracts. 
                            Does it belong to any of the following categories?
                            1.	Measurement and Methodological Advances
                            2.	Empirical Estimates and Determinants
                            3.	Policy, Institutional, and Geographic Impacts
                            4.	Intergenerational Wealth Mobility and Inheritance
                            5.	Demographic Differences in Mobility (Race, Gender, etc.)
                            6.	Mobility and Non-Income Outcomes (Health, Wellbeing, etc.)
                            7.	Theoretical and Structural Models
                            8.	Perceptions of Mobility and Attitudes
                            9.	Others
                            
                            Return the category that the research questions belong to as a number, and the exact research question if possible. 
                            If the dataset does not belong to any category, return '9' and the exact research question.                            """
                }
            ]
