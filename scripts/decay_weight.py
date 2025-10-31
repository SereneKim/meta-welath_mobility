import pandas as pd
import numpy as np
import math

def parse_list_like(s):
    """Safely parse strings like [https://a, https://b] into a real list."""
    if not isinstance(s, str) or s.strip() in ("", "[]"):
        return []
    # remove brackets
    s = s.strip()[1:-1]
    # split by comma
    items = [i.strip() for i in s.split(",") if i.strip()]
    return items

# Make sure that you have data frame with the right columns:

# Have 1 key value (e.g., for pairs)
def fill_missing_years(df_summarized, year_col='year', key_col = 'triplet_key',  count_col='paper_count', identifier_col='identifier',key2_col = None, is_triplet=True):
    rows = []
    # all_years = sorted({int(row['year']) for row in df_summarized.to_dict('records')})
    min_years = df_summarized[year_col].astype(int).min()
    max_years = df_summarized[year_col].astype(int).max()
    all_years = list(range(min_years, max_years+1))

    if key2_col is not None:
        for t, group in df_summarized.groupby([key_col, key2_col]):
            existing_years = set(group[year_col].astype(int))
            missing = set(all_years) - existing_years
            group[year_col] = pd.to_numeric(group[year_col], errors='coerce')
            
            # In is_key2 case, we assume pairwise key, so no measure/datatype/rqtype
            for y in missing:
                rows.append({'key1': t[0], 'key2': t[1], 'year': int(y), 'identifier': "filled", "paper_count": 0})
            for y in existing_years:
                rows.append({'key1': t[0], 'key2': t[1], 'year': int(y), 'identifier': "appeared", "paper_count": group['paper_count'].iloc[0]})

        df_long_filled = pd.DataFrame(rows)
        df_long_filled = df_long_filled.sort_values(['key1', 'key2', 'year']).reset_index(drop=True)
        df_long_filled.head()
        return df_long_filled
    
    else:
        for t, group in df_summarized.groupby(key_col):
            existing_years = set(group[year_col].astype(int))
            missing = set(all_years) - existing_years
            group[year_col] = pd.to_numeric(group[year_col], errors='coerce')

            if is_triplet:
                for y in missing:
                    rows.append({'key': t, 'year': int(y), 'identifier': "filled", "measure": group['measure'].iloc[0], 
                                "datatype": group['datatype'].iloc[0], "rqtype": group['rqtype'].iloc[0], "paper_count": 0})
                for y in existing_years:
                    rows.append({'key': t, 'year': int(y), 'identifier': "appeared", "measure": group['measure'].iloc[0], 
                                "datatype": group['datatype'].iloc[0], "rqtype": group['rqtype'].iloc[0], "paper_count": group['paper_count'].iloc[0]})
            else:
                for y in missing:
                    rows.append({'key': t, 'year': int(y), 'identifier': "filled", "paper_count": 0})
                for y in existing_years:
                    rows.append({'key': t, 'year': int(y), 'identifier': "appeared", "paper_count": group['paper_count'].iloc[0]})
        df_long_filled = pd.DataFrame(rows)
        df_long_filled = df_long_filled.sort_values(['key', 'year']).reset_index(drop=True)
        df_long_filled.head()
        return df_long_filled
    
    
#     # Assume that ew column is added to df_long_filled
# def compute_decaying_weights(df_long_filled, lam=math.log(2)/5, year_col='year', key_col = 'key', key2_col = None, count_col='paper_count', identifier_col='identifier', is_weighted=True):
#     if key2_col is None:
#         for key, group in df_long_filled.groupby(key_col):
#             group = group.sort_values('year')

#             ew_values = []
#             for i, (index, year) in enumerate(group[year_col].astype(int).items()):
#                 if is_weighted:
#                     print("Weighted decaying weight computation")
#                     if i == 0:
#                         if group[group[year_col] == year][identifier_col].iloc[0] == 'appeared':
#                             ew = group.iloc[i][count_col]  # initial edge weight
#                         else:
#                             ew = 0.0
#                     else:
#                         decay = math.exp(-lam * 1) #Always decay by 1 year
#                         if group.iloc[i][identifier_col] == 'appeared':
#                             ew = ew_values[-1] * decay + group.iloc[i][count_col]
#                         else:
#                             ew = ew_values[-1] * decay
#                 else:
#                     print("Unweighted decaying weight computation")
#                     if i == 0:
#                         if group[group[year_col] == year][identifier_col].iloc[0] == 'appeared':
#                             ew = 1.0  # initial edge weight
#                         else:
#                             ew = 0.0
#                     else:
#                         decay = math.exp(-lam * 1)
#                         if group.iloc[i][identifier_col] == 'appeared':
#                             ew = ew_values[-1] * decay + 1.0
#                         else:
#                             ew = ew_values[-1] * decay
#                 ew_values.append(ew)
#                 df_long_filled.loc[group.index[i], 'ew'] = ew
                
#         df_long_filled = df_long_filled.sort_values(['key', 'year']).reset_index(drop=True)
#         return df_long_filled
#     else:
#         for (key1, key2), group in df_long_filled.groupby([key_col, key2_col]):
#             group = group.sort_values('year')

#             ew_values = []
#             for i, (index, year) in enumerate(group[year_col].astype(int).items()):
#                 if is_weighted:
#                     print("Weighted decaying weight computation")
#                     if i == 0:
#                         if group[group[year_col] == year][identifier_col].iloc[0] == 'appeared':
#                             ew = group.iloc[i][count_col]  # initial edge weight
#                         else:
#                             ew = 0.0
#                     else:
#                         decay = math.exp(-lam * 1)
#                         if group.iloc[i][identifier_col] == 'appeared':
#                             ew = ew_values[-1] * decay + group.iloc[i][count_col]
#                         else:
#                             ew = ew_values[-1] * decay
#                 else:
#                     print("Unweighted decaying weight computation")
#                     if i == 0:
#                         if group[group[year_col] == year][identifier_col].iloc[0] == 'appeared':
#                             ew = 1.0  # initial edge weight
#                         else:
#                             ew = 0.0
#                     else:
#                         decay = math.exp(-lam * 1)
#                         if group.iloc[i][identifier_col] == 'appeared':
#                             ew = ew_values[-1] * decay + 1.0
#                         else:
#                             ew = ew_values[-1] * decay
#                 ew_values.append(ew)
#                 df_long_filled.loc[group.index[i], 'ew'] = ew
                
#         df_long_filled = df_long_filled.sort_values([key_col, key2_col, 'year']).reset_index(drop=True)
#         return df_long_filled


def compute_decaying_weights(df_long_filled, lam=math.log(2)/5, year_col='year',
                             key_col='key', key2_col=None, count_col='paper_count',
                             identifier_col='identifier', is_weighted=True):
    
    decay = math.exp(-lam)  # constant per year
    
    if key2_col is None:
        for key, group in df_long_filled.groupby(key_col):
            group = group.sort_values(year_col)
            ew_values = []

            for i, (idx, row) in enumerate(group.iterrows()):
                if i == 0:
                    # initialize
                    ew = row[count_col] if (row[identifier_col] == 'appeared' and is_weighted) else (1.0 if row[identifier_col] == 'appeared' else 0.0)
                else:
                    prev = ew_values[-1]
                    print(prev, decay)
                    if row[identifier_col] == 'appeared':
                        ew = prev + (row[count_col] if is_weighted else 1.0)
                    else:
                        ew = prev * decay
                ew_values.append(ew)
                df_long_filled.loc[idx, 'ew'] = ew

        df_long_filled = df_long_filled.sort_values([key_col, year_col]).reset_index(drop=True)
        return df_long_filled

    else:
        for (key1, key2), group in df_long_filled.groupby([key_col, key2_col]):
            group = group.sort_values(year_col)
            ew_values = []

            for i, (idx, row) in enumerate(group.iterrows()):
                if i == 0:
                    ew = row[count_col] if (row[identifier_col] == 'appeared' and is_weighted) else (1.0 if row[identifier_col] == 'appeared' else 0.0)
                else:
                    prev = ew_values[-1]
                    if row[identifier_col] == 'appeared':
                        ew = prev + (row[count_col] if is_weighted else 1.0)
                    else:
                        ew = prev * decay
                ew_values.append(ew)
                df_long_filled.loc[idx, 'ew'] = ew

        df_long_filled = df_long_filled.sort_values([key_col, key2_col, year_col]).reset_index(drop=True)
        return df_long_filled
