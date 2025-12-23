import pandas as pd
import numpy as np
import os
import graph
import taxonomy

path = os.path.dirname(os.getcwd())

# Load the data and then ensure that each dataset have necessary columns 
# (incl. periods (ordered), short_name - m, dt, and r short names if pair or triangle dfs)


# Main data
main = pd.read_csv(f'{path}/data_abstracts/true_mobility_studies_617_forKGs_cleaned.csv')
main['period'] = pd.cut(main['year'], bins=[1900, 2000, 2005, 2010, 2015, 2020, 2025], right=True, labels=["-2000", "2001-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2025"])

# Normalize categories
main['category_1'] = main['category_1'].replace({'Others': 'Others_Measure'}).apply(graph.norm)
main['data_cat']   = main['data_cat'].replace({'Others': 'Others_DataType'}).apply(graph.norm)
main['rq_cat']     = main['rq_cat'].replace({'Others': 'Others_RqType'}).apply(graph.norm)

main['m_short_name'] = main['category_1'].map(taxonomy.short_names)
main['dt_short_name'] = main['data_cat'].map(taxonomy.short_names)
main['rq_short_name'] = main['rq_cat'].map(taxonomy.short_names)



# Load all top degree and top strength data across periods
top_degree_all = {}
for file in os.listdir(f"{path}/results/feature-only-KG/periods/top_degree"):
    if file.endswith(".csv"):
        period = file.split("_")[0]
        df = pd.read_csv(f"{path}/results/feature-only-KG/periods/top_degree/{file}")
        df['period'] = period
        top_degree_all[period] = df
all_top_degree_df = pd.concat(top_degree_all.values(), ignore_index=True)
all_top_degree_df['short_name'] = all_top_degree_df['node'].apply(graph.norm).map(taxonomy.short_names)
all_top_degree_df['color_group'] = all_top_degree_df['node'].map(taxonomy.NODE_HIGHLIGHTS).fillna("Other")

top_strength_all = {}
for file in os.listdir(f"{path}/results/feature-only-KG/periods/top_strength"):
    if file.endswith(".csv"):
        period = file.split("_")[0]
        df = pd.read_csv(f"{path}/results/feature-only-KG/periods/top_strength/{file}")
        df['period'] = period
        top_strength_all[period] = df
all_top_strength_df = pd.concat(top_strength_all.values(), ignore_index=True)
all_top_strength_df['short_name'] = all_top_strength_df['node'].apply(graph.norm).map(taxonomy.short_names)
all_top_strength_df['color_group'] = all_top_strength_df['node'].apply(graph.norm).map(taxonomy.NODE_HIGHLIGHTS).fillna("Other")

norm_degree_all = {}
for file in os.listdir(f"{path}/results/feature-only-KG/periods/degree_normalized"):
    if file.endswith(".csv"):
        period = file.split("_")[0]
        df = pd.read_csv(f"{path}/results/feature-only-KG/periods/degree_normalized/{file}")
        df['period'] = period
        norm_degree_all[period] = df
all_norm_degree_df = pd.concat(norm_degree_all.values(), ignore_index=True)
all_norm_degree_df['short_name'] = all_norm_degree_df['node'].apply(graph.norm).map(taxonomy.short_names)
all_norm_degree_df['color_group'] = all_norm_degree_df['node'].apply(graph.norm).map(taxonomy.NODE_HIGHLIGHTS).fillna("Other")

# Load all the betweenness data across periods with normalization
top_betweenness = {}
for file in os.listdir(f"{path}/results/feature-only-KG/periods/top_betweenness"):
    if file.endswith(".csv"):
        period = file.split("_")[0]
        df = pd.read_csv(f"{path}/results/feature-only-KG/periods/top_betweenness/{file}")
        df['period'] = period
        
        # --- Normalization ---
        n_nodes = df.shape[0]  
        if n_nodes > 2:
            factor = 1 / ((n_nodes - 1) * (n_nodes - 2))
            df['score_norm'] = df['score'] * factor
        else:
            df['score_norm'] = df['score']  # fallback if very small graph
        top_betweenness[period] = df
all_top_betweenness_df = pd.concat(top_betweenness.values(), ignore_index=True)
all_top_betweenness_df['short_name'] = all_top_betweenness_df['node'].apply(graph.norm).map(taxonomy.short_names)
all_top_betweenness_df['color_group'] = all_top_betweenness_df['node'].apply(graph.norm).map(taxonomy.NODE_HIGHLIGHTS).fillna("Other")

edge_betweenness = {}
for file in os.listdir(f"{path}/results/feature-only-KG/periods/edge_betweenness"):
    if file.endswith(".csv"):
        period = file.split("_")[0]
        df = pd.read_csv(f"{path}/results/feature-only-KG/periods/edge_betweenness/{file}")
        df['period'] = period
        edge_betweenness[period] = df
all_edge_betweenness_df = pd.concat(edge_betweenness.values(), ignore_index=True)
all_edge_betweenness_df['from_short_name'] = all_edge_betweenness_df['u'].apply(graph.norm).map(taxonomy.short_names)
all_edge_betweenness_df['to_short_name'] = all_edge_betweenness_df['v'].apply(graph.norm).map(taxonomy.short_names)  
all_edge_betweenness_df['short_name'] = all_edge_betweenness_df['from_short_name'] + "-" + all_edge_betweenness_df['to_short_name']
all_edge_betweenness_df['name'] = all_edge_betweenness_df['u'] + " - " + all_edge_betweenness_df['v']
all_edge_betweenness_df['color_group'] = all_edge_betweenness_df['name'].apply(graph.norm).map(taxonomy.PAIR_HIGHLIGHTS).fillna("Other")

top_betweenness_noweight = {}
for file in os.listdir(f"{path}/results/feature-only-KG/periods/top_betweenness_noweight"):
    if file.endswith(".csv"):
        period = file.split("_")[0]
        df = pd.read_csv(f"{path}/results/feature-only-KG/periods/top_betweenness_noweight/{file}")
        df['period'] = period
        
        # --- Normalization ---
        n_nodes = df.shape[0]  
        if n_nodes > 2:
            factor = 1 / ((n_nodes - 1) * (n_nodes - 2))
            df['score_norm'] = df['score'] * factor
        else:
            df['score_norm'] = df['score']  # fallback if very small graph
        top_betweenness_noweight[period] = df
all_top_betweenness_noweight_df = pd.concat(top_betweenness_noweight.values(), ignore_index=True)
all_top_betweenness_noweight_df['short_name'] = all_top_betweenness_noweight_df['node'].apply(graph.norm).map(taxonomy.short_names)
all_top_betweenness_noweight_df['color_group'] = all_top_betweenness_noweight_df['node'].apply(graph.norm).map(taxonomy.NODE_HIGHLIGHTS).fillna("Other")

pairs = pd.read_csv(f"{path}/results/feature-only-KG/pair_counts_perYear.csv")
pairs['from_short_name'] = pairs['from_name'].apply(graph.norm).map(taxonomy.short_names)
pairs['to_short_name'] = pairs['to_name'].apply(graph.norm).map(taxonomy.short_names)
pairs['short_name'] = pairs['from_short_name'] + "-" + pairs['to_short_name']
pairs['name'] = pairs['from_name'] + " - " + pairs['to_name']

# build lookup from (short_name, period) -> score
str_lookup = all_top_strength_df.set_index(['short_name', 'period'])['strength'].to_dict()
deg_lookup = all_top_degree_df.set_index(['short_name', 'period'])['score'].to_dict()
edbtw_lookup = all_edge_betweenness_df.set_index(['short_name', 'period'])['edge_betweenness'].to_dict()
wedbtw_lookup = all_edge_betweenness_df.set_index(['short_name', 'period'])['edge_betweenness_weighted'].to_dict()
# map to/from short names + period into degrees
pairs['to_strength'] = pairs.apply(lambda r: str_lookup.get((r['to_short_name'], r['period']), np.nan), axis=1)
pairs['from_strength'] = pairs.apply(lambda r: str_lookup.get((r['from_short_name'], r['period']), np.nan), axis=1)
pairs['to_degree'] = pairs.apply(lambda r: deg_lookup.get((r['to_short_name'], r['period']), np.nan), axis=1)
pairs['from_degree'] = pairs.apply(lambda r: deg_lookup.get((r['from_short_name'], r['period']), np.nan), axis=1)
pairs['edge_betweenness'] = pairs.apply(lambda r: edbtw_lookup.get((r['short_name'], r['period']), np.nan), axis=1)
pairs['edge_betweenness_weighted'] = pairs.apply(lambda r: wedbtw_lookup.get((r['short_name'], r['period']), np.nan), axis=1)
pairs['color_group'] = pairs['name'].apply(graph.norm).map(taxonomy.PAIR_HIGHLIGHTS).fillna("Other")


tri = pd.read_csv(f"{path}/results/feature-only-KG/triangle_counts_papers.csv")
tri['m_short_name'] = tri['m.name'].apply(graph.norm).map(taxonomy.short_names)
tri['dt_short_name'] = tri['dt.name'].apply(graph.norm).map(taxonomy.short_names)
tri['rq_short_name'] = tri['rq.name'].apply(graph.norm).map(taxonomy.short_names)
tri['short_name'] = tri['m_short_name'] + "-" + tri['dt_short_name'] + "-" + tri['rq_short_name']
tri['name'] = "[" +tri['m.name'] + "," + tri['dt.name'] + "," + tri['rq.name'] + "]"
tri['color_group'] = tri['name'].apply(graph.norm).map(taxonomy.TRIANGLE_HIGHLIGHTS).fillna("Other")



