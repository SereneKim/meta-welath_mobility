import re

class Preprocess:
    @staticmethod
    def clean_abs(abs_text):
        abs_text = re.sub(r'[^\x00-\x7F]+', '', abs_text)  # Remove non-ASCII characters (like Korean)
        abs_text = re.sub(r'\bp\.\s*\d+\b', '', abs_text)  # Remove page numbers
        abs_text = re.sub(r'[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)*(?:,\s+[A-Z][a-z]+)*\s*\(\d{4}\)', '', abs_text)  # Remove parenthesis citation
        abs_text = re.sub(r'[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)*(?:,\s+[A-Z][a-z]+)*(?:,\s*\d{4})', '', abs_text)  # Remove text citation
        abs_text = re.sub(r'<[^>]+>', '', abs_text)  # Remove HTML tags
        abs_text = re.sub(r'\b(?:ABSTRACT|Abstract)\b', '', abs_text)  # Remove 'ABSTRACT' or 'Abstract'
        abs_text = re.sub(r'https?://\S+', '', abs_text)  # Remove URLs
        abs_text = re.sub(r'www\.\S+', '', abs_text)  # Remove www links
        abs_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '', abs_text)  # Remove emails
        abs_text = re.sub(r'intergenerational elasticity', 'IGE', abs_text, flags=re.IGNORECASE)  # Replace IGE

        abs_text = re.sub(r'(?<!\w)[.,]+(?!\w)', '', abs_text)  # Remove isolated periods and commas
        abs_text = re.sub(r'\s+', ' ', abs_text).strip()  # Normalize spaces

        return abs_text

class CostCalculator:
    @staticmethod
    def estimate_cost(prompt_toks, completion_toks, model="o3-mini"):
        pricing = {
            "o3-mini":{"prompt": 0.0011, "completion": 0.0044},  # pricing per 1K tokens
        }
        rate = pricing[model]
        return (prompt_toks / 1000) * rate["prompt"] + (completion_toks / 1000) * rate["completion"]