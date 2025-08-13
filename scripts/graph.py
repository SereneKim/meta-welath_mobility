import plotly.graph_objects as go
import networkx as nx
import math
import numpy as np

def scale_weight(w, min_w, max_w, min_thick=1, max_thick=8):
    if max_w == min_w:
        return (min_thick + max_thick) / 2
    return min_thick + (w - min_w) / (max_w - min_w) * (max_thick - min_thick)


color_labels = {
                # Measure Categories
                "Regression‐based Measures" : "#B5DBA5",
                "Rank‐based Measures": "#B5DBA5",
                "Transition Matrix / Probability Measures": "#B5DBA5",
                "Absolute Mobility Measures": "#B5DBA5",
                "Multigenerational Measures": "#B5DBA5",
                "Decomposition / Structural Approaches": "#B5DBA5",
                "Non‐parametric Approaches": "#B5DBA5",
                "Others_Measure": "#B5DBA5",
                
                # Data Categories
                "Panel/Longitudinal Surveys": "#339999",
                "Administrative/Registry Data": "#339999",
                "National Survey Data": "#339999",
                "Opportunity Atlas": "#339999",
                "Natural/Experimental Data": "#339999",
                "Linked Administrative Data": "#339999",
                "International Panel Data": "#339999",
                "Rich List Data": "#339999",
                "University/Institution Data": "#339999",
                "Pseudo-Panel/Household Budget Survey": "#339999",
                "Archival/Historical Data": "#339999",
                "Big Data": "#339999",
                "No dataset": "#339999",
                "Others_DataType": "#339999",
                
                # RQ Categories
                "Measurement and Methodological Advances": "#F4A988",
                "Empirical Estimates and Determinants": "#F4A988",
                "Policy, Institutional, and Geographic Impacts": "#F4A988",
                "Intergenerational Wealth Mobility and Inheritance": "#F4A988",
                "Demographic Differences in Mobility (Race, Gender, etc.)" : "#F4A988",
                "Mobility and Non-Income Outcomes (Health, Wellbeing, etc.)": "#F4A988",
                "Theoretical and Structural Models": "#F4A988",
                "Perceptions of Mobility and Attitudes": "#F4A988",
                "Others_RqType": "#F4A988"}

color_labels2 = {
                # Measure Categories
                "Regression‐based Measures" : "#B5DBA5",
                "Rank‐based Measures": "#A0C995",
                "Transition Matrix / Probability Measures": "#8DBB85",
                "Absolute Mobility Measures": "#7BAD76",
                "Multigenerational Measures": "#689F66",
                "Decomposition / Structural Approaches": "#558F56",
                "Non‐parametric Approaches": "#447F47",
                "Others_Measure": "#336F37",
                
                # Data Categories
                "Panel/Longitudinal Surveys": "#66B2B2",
                "Administrative/Registry Data": "#5CA3A3",
                "National Survey Data": "#529393",
                "Opportunity Atlas": "#488484",
                "Natural/Experimental Data": "#3F7575",
                "Linked Administrative Data": "#366666",
                "International Panel Data": "#2E5757",
                "Rich List Data": "#264848",
                "University/Institution Data": "#1F3A3A",
                "Pseudo-Panel/Household Budget Survey": "#173030",
                "Archival/Historical Data": "#102525",
                "Big Data": "#091B1B",
                "No dataset": "#051212",
                "Others_DataType": "#000A0A",
                
                # RQ Categories
                "Measurement and Methodological Advances": "#F4A988",
                "Empirical Estimates and Determinants": "#E99874",
                "Policy, Institutional, and Geographic Impacts": "#DE875F",
                "Intergenerational Wealth Mobility and Inheritance": "#D3764A",
                "Demographic Differences in Mobility (Race, Gender, etc.)": "#C86636",
                "Mobility and Non-Income Outcomes (Health, Wellbeing, etc.)": "#BD5521",
                "Theoretical and Structural Models": "#B2440C",
                "Perceptions of Mobility and Attitudes": "#A73300",
                "Others_RqType": "#8F2A00",
}

discrete_colors_31 = [
    "#B5DBA5", "#F4A988", "#CB7729", "#C04E3F", "#F7A7A1",
    "#339999", "#005F56", "#F4E1A1", "#998250", "#A6D785", "#FFD4B2",
    "#B36200", "#9B2F2F", "#FEC5C1", "#5FBFC0", "#004B47", "#F9EAB0",
    "#B3A369", "#6DA47B", "#D4816E", "#A35C18", "#932A25", "#D7827F",
    "#227C7C", "#003B36", "#E6D9A3", "#7E6E3F", "#CCCCCC", "#999999",
    "#666666", "#333333"
]


def plot_feature_graph_single(df_edges, title="Feature Graph", color_labels=None):
    G = nx.DiGraph()
    for _, row in df_edges.iterrows():
        G.add_edge(row["from_val"], row["to_val"], weight=row["weight"])

    pos = nx.spring_layout(G, seed=42, k=5 / math.sqrt(G.order()))

    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    min_w, max_w = min(edge_weights, default=1), max(edge_weights, default=1)
    scaled_widths = [scale_weight(w, min_w, max_w) for w in edge_weights]

    # Edge traces
    edge_traces = []
    for (u, v), width in zip(G.edges(), scaled_widths):
        edge_traces.append(go.Scatter(
            x=[pos[u][0], pos[v][0], None],
            y=[pos[u][1], pos[v][1], None],
            mode='lines',
            line=dict(width=width, color='gray'),
            showlegend=False,
            hoverinfo='text',
            text=[f'{u} → {v}<br>Weight: {G[u][v]["weight"]}']
        ))

    # Node colors
    node_labels = list(G.nodes())
    x_pos = [pos[n][0] for n in node_labels]
    y_pos = [pos[n][1] for n in node_labels]

    if color_labels is None:
        node_to_color = {label: i for i, label in enumerate(sorted(node_labels))}
        node_colors = [discrete_colors_31[node_to_color[label] % len(discrete_colors_31)] for label in node_labels]
    else:
        node_colors = [color_labels.get(label, "#cccccc") for label in node_labels]

    # Node traces
    node_traces = []
    for i, label in enumerate(node_labels):
        node_traces.append(go.Scatter(
            x=[x_pos[i]],
            y=[y_pos[i]],
            mode='markers',
            name=label,
            marker=dict(
                size=20,
                color=node_colors[i],
                line=dict(width=2, color='darkblue')
            ),
            hoverinfo='text',
            text=[label],
            showlegend=True
        ))

    return go.Figure(data=edge_traces + node_traces, layout=go.Layout(
        title=title,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        font=dict(size=16, color='black'),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor='rgba(255,255,255,0)',
        font_family="Times New Roman"
    ))

def plot_centrality_subgraphs_single(df_edges, color_labels, top_k=7, center_radius=0.1):
    G = nx.DiGraph()
    for _, row in df_edges.iterrows():
        G.add_edge(row["from_val"], row["to_val"], weight=row["weight"])

    pos = nx.spring_layout(G, seed=42, k=5 / math.sqrt(G.order()))
    betweenness = nx.betweenness_centrality(G)
    degree = dict(G.degree())
    
    top_betw = sorted(betweenness, key=betweenness.get, reverse=True)[:top_k]
    top_deg = sorted(degree, key=degree.get, reverse=True)[:top_k]
    geom_sorted = sorted(
        [n for n, (x, y) in pos.items() if abs(x) <= center_radius and abs(y) <= center_radius],
        key=lambda n: pos[n][0]**2 + pos[n][1]**2
    )

    def make_fig(sub_nodes, title):
        G_sub = G.subgraph(sub_nodes)
        pos_sub = {n: pos[n] for n in G_sub.nodes()}
        edge_weights = [G_sub[u][v]['weight'] for u, v in G_sub.edges()]
        min_w, max_w = min(edge_weights, default=1), max(edge_weights, default=1)
        scaled_widths = [scale_weight(w, min_w, max_w) for w in edge_weights]

        edge_traces = [
            go.Scatter(
                x=[pos_sub[u][0], pos_sub[v][0], None],
                y=[pos_sub[u][1], pos_sub[v][1], None],
                mode='lines',
                line=dict(width=w, color='gray'),
                showlegend=False
            ) for (u, v), w in zip(G_sub.edges(), scaled_widths)
        ]

        node_traces = [
            go.Scatter(
                x=[pos_sub[n][0]],
                y=[pos_sub[n][1]],
                mode='markers+text',
                text=[str(i + 1)],
                textposition='middle center',
                textfont=dict(size=12, color='white'),
                marker=dict(
                    size=22,
                    color=color_labels.get(n, "#cccccc"),
                    line=dict(width=2, color='darkblue')
                ),
                hoverinfo='text',
                name=n
            ) for i, n in enumerate(sub_nodes)
        ]

        return go.Figure(data=edge_traces + node_traces, layout=go.Layout(
            title=title,
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='rgba(255,255,255,0)',
            margin=dict(b=20, l=5, r=5, t=40),
            font=dict(size=16, color='black'),
            font_family="Times New Roman",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        ))

    return (
        make_fig(geom_sorted, "Geometric Center Nodes"),
        make_fig(top_betw, "Top Betweenness Centrality"),
        make_fig(top_deg, "Top Degree Centrality")
    )



#-----------------------------------------------------------------------------------
# For multple years
#-----------------------------------------------------------------------------------

def plot_spring_graph(df_feature_edges, seed=47, k=5, iterations = 50, color_labels = color_labels, font_size=16, title=None, size = 10 , font_family="Times New Roman"):
    # Build the graph
    G = nx.DiGraph()
    for _, row in df_feature_edges.iterrows():
        G.add_edge(row['from_val'], row['to_val'], weight=row['weight'])

    # Layout
    pos = nx.spring_layout(G, seed=seed, k=k / math.sqrt(G.order()), iterations=iterations)

    # Edge thickness based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    min_w, max_w = min(edge_weights), max(edge_weights)

    scaled_widths = [scale_weight(w, min_w, max_w) for w in edge_weights]


    edge_traces = []
    for (u, v), width in zip(G.edges(), scaled_widths):
        edge_traces.append(go.Scatter(
            x=[pos[u][0], pos[v][0], None],
            y=[pos[u][1], pos[v][1], None],
            mode='lines',
            line=dict(width=width, color='gray'),
            showlegend=False,
            name=f'{u} → {v}',
            hoverinfo='text',
            text=[f'{u} → {v}<br>Weight: {G[u][v]["weight"]}']
        ))

    # Map node labels to color
    node_labels = list(G.nodes())
    x_pos = [pos[node][0] for node in node_labels]
    y_pos = [pos[node][1] for node in node_labels]


    # Create individual node traces with legends
    node_traces = []
    
    for i, label in enumerate(node_labels):
        node_size = size[label] if isinstance(size, dict) else size
        node_traces.append(go.Scatter(
            x=[x_pos[i]],
            y=[y_pos[i]],
            mode='markers',
            name=label,
            marker=dict(
                size=node_size,
                color=color_labels[label],
                line=dict(width=2, color=color_labels[label])
            ),
            hoverinfo='text',
            text=[label],
            showlegend=True
        ))


    # Final plot
    if title:
        fig = go.Figure(data=edge_traces + node_traces, layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            # paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(b=20, l=5, r=5, t=40),
            font=dict(size=font_size, color='black'),
            font_family=font_family,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        ))
    else:
        fig = go.Figure(data=edge_traces + node_traces, layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            # paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(b=20, l=5, r=5, t=40),
            font=dict(size=font_size, color='black'),
            font_family=font_family,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        ))

    return fig


def calculate_movement_size(df_feature_edges):
    node_positions_by_year = {}

    for year in df_feature_edges['to_year'].unique():
        df_year = df_feature_edges[df_feature_edges['to_year'] == year]
        G = nx.DiGraph()
        for _, row in df_year.iterrows():
            G.add_edge(row['from_val'], row['to_val'], weight=row['weight'])
        pos = nx.spring_layout(G, seed=47, k=5 / math.sqrt(G.order()))
        node_positions_by_year[year] = pos

    # movement = {}
    sorted_years = sorted(df_feature_edges['to_year'].unique())

    for i, year in enumerate(sorted_years):
        cumulative_movement = {}
        for node in set().union(*[pos.keys() for pos in node_positions_by_year.values()]):
            dist = 0
            for y1, y2 in zip(sorted_years[:i], sorted_years[1:i+1]):
                p1 = node_positions_by_year[y1].get(node)
                p2 = node_positions_by_year[y2].get(node)
                if p1 is not None and p2 is not None:
                    dist += np.linalg.norm(np.array(p2) - np.array(p1))
            cumulative_movement[node] = dist
            
    return node_positions_by_year, cumulative_movement



def plot_top_k(df_feature_edges, top_k = 4, color_labels = color_labels): # Betweenness & Degree Centrality
    # Build the graph
    G = nx.DiGraph()
    for _, row in df_feature_edges.iterrows():
        G.add_edge(row['from_val'], row['to_val'], weight=row['weight'])

    # Layout
    pos = nx.spring_layout(G, seed=42, k=5 / math.sqrt(G.order()))

    # --- CENTRALITY METRICS ---
    betweenness = nx.betweenness_centrality(G)
    degree = dict(G.degree())

    # How many top central nodes to show
    top_k = top_k 

    # Top-k nodes by betweenness centrality
    top_betweenness_nodes = sorted(betweenness, key=betweenness.get, reverse=True)[:top_k]
    # Top-k nodes by degree centrality
    top_degree_nodes = sorted(degree, key=degree.get, reverse=True)[:top_k]

    # --- CREATE SUBGRAPH FUNCTION ---
    def make_subgraph_plot(subgraph_nodes_ordered, title):
        G_sub = G.subgraph(subgraph_nodes_ordered).copy()
        pos_sub = {n: pos[n] for n in G_sub.nodes()}
        
        edge_weights = [G_sub[u][v]['weight'] for u, v in G_sub.edges()]
        min_w, max_w = min(edge_weights, default=1), max(edge_weights, default=1)
        scaled_widths = [scale_weight(w, min_w, max_w) for w in edge_weights]
        
        # Edge traces
        edge_traces = []
        for (u, v), width in zip(G_sub.edges(), scaled_widths):
            edge_traces.append(go.Scatter(
                x=[pos_sub[u][0], pos_sub[v][0], None],
                y=[pos_sub[u][1], pos_sub[v][1], None],
                mode='lines',
                line=dict(width=width, color='gray'),
                showlegend=False,
            ))

        # Node traces with correct rank labels
        node_traces = []
        for i, node in enumerate(subgraph_nodes_ordered):
            x, y = pos_sub[node]
            rank_label = str(i + 1)  # ranked order (1 = highest centrality, or closest to center)
            node_traces.append(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                text=[rank_label],
                textposition='middle center',
                textfont=dict(size=18, color='black'),
                marker=dict(
                    size=30,
                    color=color_labels.get(node, '#cccccc'),
                    line=dict(width=2, color=color_labels.get(node, '#cccccc')),            ),
                hovertext=[f'{rank_label}: {node}'],
                hoverinfo='text',
                name=node,
                showlegend=True
            ))

        return go.Figure(data=edge_traces + node_traces, layout=go.Layout(
            title=title,
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='rgba(255,255,255,0)',
            margin=dict(b=20, l=5, r=5, t=40),
            font=dict(size=16, color='black'),
            # font_family="Times New Roman",
            font_family = "Avenir",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        ))

    # --- DISPLAY THREE PLOTS ---
    fig_betw = make_subgraph_plot(top_betweenness_nodes, 'Top Betweenness Centrality Nodes')
    fig_deg = make_subgraph_plot(top_degree_nodes, 'Top Degree Centrality Nodes')

    return fig_betw, fig_deg

