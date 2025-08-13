import json
import os
import logging

# 1. Get a list of categories based on the abstracts of papers.

system_types = "You are a research assistant in the Social Sciences domain. I, the researcher, have a list of abstracts of papers that studied inter/multi generational wealth/income/earning mobility. I want you to categorize the measures that the papers used based on the abstracts."
prompt_types =f"""                        
                        ```Abstracts
                        {json_abs}
                        ```
                        First, find the representative categories of the measures that the papers used based on the abstracts. Here, 'measures' mean an estimate from a model or an equation that quantifies the inter/multi generational wealth/income/earning mobility.
                        Then, return the categories and lists of id's that belong to each in a JSON format.
                        For the id's that do not belong to any category, return them in a separate category named 'Others'.
                        """
                        
# 2. Get a list of categories based on the titles of papers.
system_titles = "You are a research assistant in the Social Sciences domain. I, the researcher, have a list of titles of papers that studied inter/multi generational wealth/income/earning mobility. I want you to categorize the measures that the papers used based on the titles."
prompt_titles = " "

