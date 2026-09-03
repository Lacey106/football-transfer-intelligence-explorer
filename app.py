from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
FALLBACK_DATA_DIR = Path(r"D:\jupyter\Manchester\final essay")
REQUIRED_FILES = {
    "edges": "club_edges.csv",
    "clubs": "club_embeddings.csv",
    "transfers": "cleaned_transfers.csv",
    "network_summary": "network_summary.csv",
    "cluster_summary": "louvain_cluster_summary.csv",
    "cluster_countries": "top_countries_by_louvain_cluster.csv",
}


MAP_COUNTRY_ALIASES = {
    "England": "United Kingdom",
    "Scotland": "United Kingdom",
    "Wales": "United Kingdom",
    "Northern Ireland": "United Kingdom",
    "Hongkong": "China",
}

MAP_DISPLAY_DUPLICATES = {
    "China": ["Taiwan"],
}


st.set_page_config(
    page_title="Football Transfer Intelligence Explorer",
    page_icon="soccer",
    layout="wide",
    initial_sidebar_state="expanded",
)


def resolve_data_dir() -> Path:
    if all((APP_DIR / filename).exists() for filename in REQUIRED_FILES.values()):
        return APP_DIR
    return FALLBACK_DATA_DIR


DATA_DIR = resolve_data_dir()


@st.cache_data(show_spinner="Loading transfer network data...")
def load_data(data_dir: str):
    base = Path(data_dir)
    edges = pd.read_csv(base / REQUIRED_FILES["edges"])
    clubs = pd.read_csv(base / REQUIRED_FILES["clubs"])
    transfers = pd.read_csv(base / REQUIRED_FILES["transfers"])

    optional = {}
    for key in ["network_summary", "cluster_summary", "cluster_countries"]:
        path = base / REQUIRED_FILES[key]
        optional[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()

    for df in [edges, clubs, transfers]:
        for col in df.select_dtypes("object").columns:
            df[col] = df[col].fillna("").astype(str)

    money_cols = [
        "total_transfer_fee",
        "average_transfer_fee",
        "max_transfer_fee",
        "transfer_fee_clean",
        "market_value_clean",
        "total_fee_involved",
        "total_fee_in",
        "total_fee_out",
    ]
    for df in [edges, clubs, transfers]:
        for col in money_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["season", "first_season", "last_season", "louvain_cluster"]:
        for df in [edges, clubs, transfers]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return edges, clubs, transfers, optional


club_edges, club_embeddings, transfers, optional_tables = load_data(str(DATA_DIR))


def money(value) -> str:
    value = float(value or 0)
    sign = "-" if value < 0 else ""
    euro = "\N{EURO SIGN}"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{sign}{euro}{abs_value / 1_000_000_000:.2f}bn"
    if abs_value >= 1_000_000:
        return f"{sign}{euro}{abs_value / 1_000_000:.2f}m"
    return f"{sign}{euro}{abs_value:,.0f}"


def add_money_display(df, columns):
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[f"{col}_display"] = result[col].apply(money)
    return result


def format_window(value: str) -> str:
    return {"s": "Summer", "w": "Winter"}.get(str(value).lower(), str(value))


def selected_customdata(selection):
    try:
        points = selection.selection.points
    except AttributeError:
        return None
    if not points:
        return None
    data = points[0].get("customdata")
    if isinstance(data, list) and data:
        return data[0]
    return data


def selectable_chart(fig, key, height=None):
    if height:
        fig.update_layout(height=height)
    try:
        return st.plotly_chart(
            fig,
            use_container_width=True,
            key=key,
            on_select="rerun",
            selection_mode="points",
        )
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)
        return None


def set_linked_country(country):
    if country and st.session_state.get("linked_country") != country:
        st.session_state["linked_country"] = country
        st.session_state["linked_country_members"] = country_members(country)
        st.rerun()


def set_linked_club(club):
    if club and st.session_state.get("linked_club") != club:
        st.session_state["linked_club"] = club
        st.rerun()


def set_linked_edge(source, target):
    if source and target:
        st.session_state["linked_edge_source"] = source
        st.session_state["linked_edge_target"] = target


def clean_club_list(series):
    return sorted([value for value in series.dropna().unique() if str(value).strip()])


def build_country_summary(clubs, edges, transfer_rows):
    clubs = clubs.copy()
    edges = edges.copy()
    transfer_rows = transfer_rows.copy()
    clubs["map_country"] = clubs["country"].replace(MAP_COUNTRY_ALIASES)
    edges["source_map_country"] = edges["source_country"].replace(MAP_COUNTRY_ALIASES)
    edges["target_map_country"] = edges["target_country"].replace(MAP_COUNTRY_ALIASES)
    transfer_rows["source_map_country"] = transfer_rows["source_country"].replace(MAP_COUNTRY_ALIASES)
    transfer_rows["target_map_country"] = transfer_rows["target_country"].replace(MAP_COUNTRY_ALIASES)

    country = (
        clubs.groupby("map_country", as_index=False)
        .agg(
            clubs=("club", "nunique"),
            total_transfers=("total_transfers", "sum"),
            total_fee=("total_fee_involved", "sum"),
            avg_pagerank=("pagerank", "mean"),
            clusters=("louvain_cluster", "nunique"),
            original_countries=("country", lambda values: ", ".join(sorted(set(values)))),
        )
        .query("map_country != ''")
        .rename(columns={"map_country": "country"})
    )

    outbound = edges.groupby("source_map_country", as_index=False)["transfer_count"].sum()
    outbound = outbound.rename(columns={"source_map_country": "country", "transfer_count": "outbound_links"})
    inbound = edges.groupby("target_map_country", as_index=False)["transfer_count"].sum()
    inbound = inbound.rename(columns={"target_map_country": "country", "transfer_count": "inbound_links"})

    fees_out = transfer_rows.groupby("source_map_country", as_index=False)["transfer_fee_clean"].sum()
    fees_out = fees_out.rename(columns={"source_map_country": "country", "transfer_fee_clean": "player_fee_out"})
    fees_in = transfer_rows.groupby("target_map_country", as_index=False)["transfer_fee_clean"].sum()
    fees_in = fees_in.rename(columns={"target_map_country": "country", "transfer_fee_clean": "player_fee_in"})

    for add in [outbound, inbound, fees_out, fees_in]:
        country = country.merge(add, on="country", how="left")

    fill_cols = ["outbound_links", "inbound_links", "player_fee_out", "player_fee_in"]
    country[fill_cols] = country[fill_cols].fillna(0)
    country["net_fee_flow"] = country["player_fee_in"] - country["player_fee_out"]
    country["hover"] = country.apply(
        lambda row: (
            f"<b>{row['country']}</b><br>"
            f"Dataset countries: {row['original_countries']}<br>"
            f"Clubs: {int(row['clubs']):,}<br>"
            f"Club-level transfers: {int(row['total_transfers']):,}<br>"
            f"Network clusters: {int(row['clusters']):,}<br>"
            f"Incoming fees: {money(row['player_fee_in'])}<br>"
            f"Outgoing fees: {money(row['player_fee_out'])}"
        ),
        axis=1,
    )
    return country.sort_values("clubs", ascending=False)


def country_members(map_country):
    return [
        country for country in all_countries
        if MAP_COUNTRY_ALIASES.get(country, country) == map_country
    ]


def country_map_display_rows(summary):
    rows = summary.copy()
    rows["map_location"] = rows["country"]
    rows["display_country"] = rows["country"]
    rows["linked_country"] = rows["country"]
    additions = []
    for country, duplicate_locations in MAP_DISPLAY_DUPLICATES.items():
        match = rows[rows["country"] == country]
        if match.empty:
            continue
        for location in duplicate_locations:
            duplicate = match.iloc[0].copy()
            duplicate["map_location"] = location
            duplicate["display_country"] = country
            duplicate["linked_country"] = country
            duplicate["original_countries"] = f"{duplicate['original_countries']}, {location} (map display)"
            additions.append(duplicate)
    if additions:
        rows = pd.concat([rows, pd.DataFrame(additions)], ignore_index=True)
    return rows


def add_map_countries(df):
    result = df.copy()
    if "source_country" in result.columns:
        result["source_map_country"] = result["source_country"].replace(MAP_COUNTRY_ALIASES)
    if "target_country" in result.columns:
        result["target_map_country"] = result["target_country"].replace(MAP_COUNTRY_ALIASES)
    if "country" in result.columns:
        result["map_country"] = result["country"].replace(MAP_COUNTRY_ALIASES)
    return result


def build_country_flows(transfer_rows, min_fee=0, top_n=18):
    rows = add_map_countries(transfer_rows)
    rows = rows[
        (rows["source_map_country"] != "")
        & (rows["target_map_country"] != "")
        & (rows["source_map_country"] != rows["target_map_country"])
    ].copy()
    if min_fee > 0:
        rows = rows[rows["transfer_fee_clean"] >= min_fee]
    flows = (
        rows.groupby(["source_map_country", "target_map_country"], as_index=False)
        .agg(transfers=("transfer_id", "count"), total_fee=("transfer_fee_clean", "sum"))
        .sort_values(["transfers", "total_fee"], ascending=False)
        .head(top_n)
    )
    return flows


def make_sankey(flows):
    if flows.empty:
        fig = go.Figure()
        fig.update_layout(height=520, margin=dict(l=8, r=8, t=24, b=8))
        return fig
    flows = add_money_display(flows, ["total_fee"])
    labels = sorted(set(flows["source_map_country"]).union(set(flows["target_map_country"])))
    index = {label: i for i, label in enumerate(labels)}
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=24,
                    thickness=20,
                    line=dict(color="#334155", width=0.5),
                    label=labels,
                    color="#64748b",
                ),
                link=dict(
                    source=flows["source_map_country"].map(index),
                    target=flows["target_map_country"].map(index),
                    value=flows["transfers"],
                    customdata=flows[["total_fee_display"]],
                    hovertemplate=(
                        "%{source.label} to %{target.label}<br>"
                        "Transfers: %{value:,}<br>"
                        "Total fee: %{customdata[0]}<extra></extra>"
                    ),
                    color="rgba(37, 99, 235, 0.30)",
                ),
            )
        ]
    )
    fig.update_layout(height=560, margin=dict(l=8, r=8, t=24, b=8), font=dict(size=13))
    return fig


def build_cluster_profiles(clubs, edges):
    profiles = []
    for cluster_id, group in clubs.groupby("louvain_cluster"):
        if pd.isna(cluster_id):
            continue
        top_countries = (
            group["country"].value_counts().head(4).rename_axis("country").reset_index(name="clubs")
        )
        leaders = group.sort_values("pagerank", ascending=False).head(5)
        member_clubs = set(group["club"])
        internal_edges = edges[
            edges["source_club"].isin(member_clubs) & edges["target_club"].isin(member_clubs)
        ]
        external_edges = edges[
            edges["source_club"].isin(member_clubs) ^ edges["target_club"].isin(member_clubs)
        ]
        if not external_edges.empty:
            direction_rows = external_edges.copy()
            direction_rows["direction"] = direction_rows.apply(
                lambda row: (
                    f"{row['source_country']} to {row['target_country']}"
                    if row["source_club"] in member_clubs
                    else f"{row['target_country']} to {row['source_country']}"
                ),
                axis=1,
            )
            main_directions = (
                direction_rows.groupby("direction")["transfer_count"]
                .sum()
                .sort_values(ascending=False)
                .head(3)
            )
            main_directions_text = ", ".join(
                f"{direction} ({int(count)})" for direction, count in main_directions.items()
            )
        else:
            main_directions_text = "No external direction"
        profiles.append(
            {
                "cluster": int(cluster_id),
                "label": str(group["louvain_cluster_label"].mode().iloc[0])
                if "louvain_cluster_label" in group and not group["louvain_cluster_label"].mode().empty
                else f"Cluster {int(cluster_id)}",
                "clubs": group["club"].nunique(),
                "countries": group["country"].nunique(),
                "avg_pagerank": group["pagerank"].mean(),
                "avg_degree": group["degree"].mean(),
                "internal_relationships": len(internal_edges),
                "external_relationships": len(external_edges),
                "top_countries": ", ".join(
                    f"{row.country} ({row.clubs})" for row in top_countries.itertuples()
                ),
                "leading_clubs": ", ".join(leaders["club"].tolist()),
                "main_transfer_directions": main_directions_text,
            }
        )
    return pd.DataFrame(profiles).sort_values("clubs", ascending=False)


def similar_clubs(clubs, club_name, limit=12):
    if not club_name or club_name not in set(clubs["club"]):
        return pd.DataFrame()
    emb_cols = [col for col in clubs.columns if col.startswith("emb_")]
    if not emb_cols:
        return pd.DataFrame()
    matrix = clubs[emb_cols].fillna(0)
    target = matrix.loc[clubs["club"] == club_name].iloc[0]
    distances = ((matrix - target) ** 2).sum(axis=1) ** 0.5
    result = clubs[["club", "country", "louvain_cluster_label", "pagerank", "degree", "total_transfers"]].copy()
    result["embedding_distance"] = distances
    return result[result["club"] != club_name].sort_values("embedding_distance").head(limit)


def filter_clubs(clubs, countries, clusters, search):
    result = clubs.copy()
    if countries:
        result = result[result["country"].isin(countries)]
    if clusters:
        result = result[result["louvain_cluster"].isin(clusters)]
    if search:
        result = result[result["club"].str.contains(search, case=False, na=False)]
    return result


def edge_subset(edges, clubs, min_count, selected_club=""):
    allowed = set(clubs["club"])
    result = edges[
        edges["source_club"].isin(allowed)
        & edges["target_club"].isin(allowed)
        & (edges["transfer_count"] >= min_count)
    ].copy()
    if selected_club:
        result = result[
            (result["source_club"] == selected_club)
            | (result["target_club"] == selected_club)
        ]
    return result


def make_network(edges, clubs, selected_club, max_edges):
    top_edges = edges.sort_values(["transfer_count", "total_transfer_fee"], ascending=False).head(max_edges)
    graph = nx.Graph()
    for row in top_edges.itertuples():
        graph.add_edge(
            row.source_club,
            row.target_club,
            weight=float(row.transfer_count),
            fee=float(row.total_transfer_fee),
            first=int(row.first_season) if pd.notna(row.first_season) else "",
            last=int(row.last_season) if pd.notna(row.last_season) else "",
        )

    if graph.number_of_edges() == 0:
        return go.Figure()

    pos = nx.spring_layout(graph, seed=42, k=0.55, iterations=90, weight="weight")
    club_lookup = clubs.set_index("club").to_dict("index")
    max_weight = max([data["weight"] for _, _, data in graph.edges(data=True)] or [1])
    max_pr = max(clubs["pagerank"].max(), 0.000001)
    focus_neighbors = set()
    if selected_club and selected_club in graph:
        focus_neighbors = set(graph.neighbors(selected_club))

    edge_traces = []
    for source, target, data in graph.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        is_focus = selected_club and selected_club in {source, target}
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(
                    width=0.8 + (data["weight"] / max_weight) * (6 if is_focus else 2.0),
                    color="rgba(255, 196, 71, 0.92)" if is_focus else "rgba(74, 98, 138, 0.18)",
                ),
                hoverinfo="text",
                text=(
                    f"<b>{source} - {target}</b><br>"
                    f"Transfers: {int(data['weight'])}<br>"
                    f"Total fee: {money(data['fee'])}<br>"
                    f"Seasons: {data['first']} - {data['last']}"
                ),
                showlegend=False,
            )
        )

    node_rows = []
    for node in graph.nodes:
        info = club_lookup.get(node, {})
        node_rows.append(
            {
                "club": node,
                "x": pos[node][0],
                "y": pos[node][1],
                "country": info.get("country", ""),
                "cluster": info.get("louvain_cluster", -1),
                "pagerank": info.get("pagerank", 0),
                "degree": info.get("degree", 0),
                "transfers": info.get("total_transfers", 0),
                "size": 8 + (float(info.get("pagerank", 0)) / max_pr) * 30,
                "is_focus": node == selected_club,
                "is_neighbor": node in focus_neighbors,
            }
        )
    nodes = pd.DataFrame(node_rows)
    palette = px.colors.qualitative.Safe + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    countries = sorted(nodes["country"].fillna("").unique().tolist())
    country_colors = {country: palette[index % len(palette)] for index, country in enumerate(countries)}
    nodes["color"] = nodes["country"].map(country_colors).fillna("#94a3b8")
    nodes["display_size"] = nodes["size"]
    nodes["opacity"] = 0.9
    nodes["line_width"] = 0.8
    if selected_club:
        nodes.loc[~(nodes["is_focus"] | nodes["is_neighbor"]), "opacity"] = 0.34
        nodes.loc[nodes["is_neighbor"], "line_width"] = 1.4
        nodes.loc[nodes["is_focus"], "display_size"] = nodes.loc[nodes["is_focus"], "display_size"] + 18
        nodes.loc[nodes["is_focus"], "line_width"] = 3.4

    node_trace = go.Scatter(
        x=nodes["x"],
        y=nodes["y"],
        mode="markers+text" if selected_club else "markers",
        text=nodes["club"].where(nodes["is_focus"], ""),
        textposition="top center",
        customdata=nodes[["club", "country", "pagerank", "degree", "transfers"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Country: %{customdata[1]}<br>"
            "PageRank: %{customdata[2]:.5f}<br>"
            "Degree: %{customdata[3]}<br>"
            "Transfers: %{customdata[4]}<extra></extra>"
        ),
        marker=dict(
            size=nodes["display_size"],
            color=nodes["color"],
            symbol=nodes["is_focus"].map({True: "star", False: "circle"}),
            opacity=nodes["opacity"],
            line=dict(width=nodes["line_width"], color="#ffffff"),
        ),
        showlegend=False,
    )

    fig = go.Figure(edge_traces + [node_trace])
    fig.update_layout(
        height=720,
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="closest",
    )
    return fig


def make_temporal_network(transfer_rows, clubs, year, selected_club="", max_edges=160):
    yearly = transfer_rows[transfer_rows["season"] == year].copy()
    if selected_club:
        yearly = yearly[
            (yearly["source_club"] == selected_club)
            | (yearly["target_club"] == selected_club)
        ]
    if yearly.empty:
        fig = go.Figure()
        fig.update_layout(
            height=680,
            margin=dict(l=8, r=8, t=24, b=8),
            annotations=[
                dict(
                    text=f"No transfers available for {year} under current filters.",
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                )
            ],
        )
        return fig

    year_edges = (
        yearly.groupby(["source_club", "target_club"], as_index=False)
        .agg(
            transfer_count=("transfer_id", "count"),
            total_fee=("transfer_fee_clean", "sum"),
            source_country=("source_country", "first"),
            target_country=("target_country", "first"),
        )
        .sort_values(["transfer_count", "total_fee"], ascending=False)
        .head(max_edges)
    )
    active = sorted(set(year_edges["source_club"]).union(set(year_edges["target_club"])))
    nodes = clubs[clubs["club"].isin(active)].copy()
    if nodes.empty:
        return go.Figure()

    pos = nodes.set_index("club")[["embed_x", "embed_y"]].to_dict("index")
    max_count = max(year_edges["transfer_count"].max(), 1)
    edge_traces = []
    for row in year_edges.itertuples():
        if row.source_club not in pos or row.target_club not in pos:
            continue
        x0, y0 = pos[row.source_club]["embed_x"], pos[row.source_club]["embed_y"]
        x1, y1 = pos[row.target_club]["embed_x"], pos[row.target_club]["embed_y"]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(
                    width=0.4 + 4 * row.transfer_count / max_count,
                    color="rgba(37, 99, 235, 0.26)",
                ),
                hoverinfo="text",
                text=(
                    f"<b>{row.source_club} to {row.target_club}</b><br>"
                    f"Season: {year}<br>"
                    f"Transfers: {int(row.transfer_count)}<br>"
                    f"Fee: {money(row.total_fee)}"
                ),
                showlegend=False,
            )
        )

    node_activity = (
        yearly.melt(
            id_vars=["transfer_id"],
            value_vars=["source_club", "target_club"],
            value_name="club",
        )
        .groupby("club", as_index=False)
        .agg(year_transfers=("transfer_id", "count"))
    )
    nodes = nodes.merge(node_activity, on="club", how="left")
    nodes["year_transfers"] = nodes["year_transfers"].fillna(0)
    max_activity = max(nodes["year_transfers"].max(), 1)

    node_trace = go.Scatter(
        x=nodes["embed_x"],
        y=nodes["embed_y"],
        mode="markers",
        customdata=nodes[["club", "country", "year_transfers", "pagerank", "louvain_cluster_label"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Country: %{customdata[1]}<br>"
            "Season transfers: %{customdata[2]}<br>"
            "PageRank: %{customdata[3]:.5f}<br>"
            "Community: %{customdata[4]}<extra></extra>"
        ),
        marker=dict(
            size=8 + 28 * nodes["year_transfers"] / max_activity,
            color=nodes["louvain_cluster"],
            colorscale="Turbo",
            opacity=0.88,
            line=dict(width=0.7, color="#ffffff"),
            colorbar=dict(title="Cluster"),
        ),
        showlegend=False,
    )
    fig = go.Figure(edge_traces + [node_trace])
    fig.update_layout(
        title=f"Transfer network evolution: {year}",
        height=680,
        margin=dict(l=8, r=8, t=48, b=8),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
    )
    return fig


st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
    h1, h2, h3 {letter-spacing: 0 !important;}
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 14px 16px;
        border-radius: 8px;
    }
    .section-note {
        color: #475569;
        font-size: 0.95rem;
        margin-top: -0.4rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("Football Transfer Intelligence Explorer")
st.caption(
    "A drill-down interface for country distribution, club communities, network relationships, and player-level transfer evidence."
)

season_min = int(transfers["season"].min())
season_max = int(transfers["season"].max())
all_countries = clean_club_list(club_embeddings["country"])
all_clusters = sorted(club_embeddings["louvain_cluster"].dropna().unique().astype(int).tolist())

for key, value in {
    "linked_country": "",
    "linked_country_members": [],
    "linked_club": "",
    "linked_edge_source": "",
    "linked_edge_target": "",
}.items():
    st.session_state.setdefault(key, value)

with st.sidebar:
    st.header("Explore")
    season_range = st.slider("Season range", season_min, season_max, (season_min, season_max))
    selected_countries = st.multiselect("Country filter", all_countries, default=[])
    club_search = st.text_input("Club search")
    with st.expander("Advanced options", expanded=False):
        selected_clusters = st.multiselect("Community filter", all_clusters, default=[])
        min_relationship = st.slider("Minimum number of transfers", 1, 20, 2)
        max_edges = st.slider("Network detail level", 50, 900, 350, step=50)
        st.caption("Higher detail shows more club-to-club links; higher minimum transfers keeps only stronger relationships.")
    if st.session_state["linked_country"] or st.session_state["linked_club"] or st.session_state["linked_edge_source"]:
        st.markdown("#### Linked selection")
        if st.session_state["linked_country"]:
            st.caption(f"Country: {st.session_state['linked_country']}")
        if st.session_state["linked_club"]:
            st.caption(f"Club: {st.session_state['linked_club']}")
        if st.session_state["linked_edge_source"]:
            st.caption(
                f"Edge: {st.session_state['linked_edge_source']} to {st.session_state['linked_edge_target']}"
            )
        if st.button("Clear linked selection"):
            st.session_state["linked_country"] = ""
            st.session_state["linked_country_members"] = []
            st.session_state["linked_club"] = ""
            st.session_state["linked_edge_source"] = ""
            st.session_state["linked_edge_target"] = ""
            st.rerun()

season_transfers = transfers[
    (transfers["season"] >= season_range[0]) & (transfers["season"] <= season_range[1])
].copy()
active_countries = selected_countries
if not active_countries and st.session_state["linked_country_members"]:
    active_countries = st.session_state["linked_country_members"]
filtered_clubs = filter_clubs(club_embeddings, active_countries, selected_clusters, club_search)
if club_search and not st.session_state["linked_club"]:
    search_matches = clean_club_list(filtered_clubs["club"])
    if len(search_matches) == 1:
        st.session_state["linked_club"] = search_matches[0]
filtered_edges = edge_subset(club_edges, filtered_clubs, min_relationship)
country_summary = build_country_summary(filtered_clubs, filtered_edges, season_transfers)
country_map_summary = country_map_display_rows(country_summary)
country_flows = build_country_flows(season_transfers, top_n=22)
cluster_profiles = build_cluster_profiles(filtered_clubs, filtered_edges)

metric_cols = st.columns(5)
metric_cols[0].metric("Clubs", f"{filtered_clubs['club'].nunique():,}")
metric_cols[1].metric("Countries", f"{filtered_clubs['country'].nunique():,}")
metric_cols[2].metric("Visible relationships", f"{len(filtered_edges):,}")
metric_cols[3].metric("Cleaned transfer records", f"{len(season_transfers):,}")
metric_cols[4].metric("Fees tracked", money(season_transfers["transfer_fee_clean"].sum()))

story_tab, overview, network_tab, temporal_tab, club_tab, community_tab, evidence_tab = st.tabs(
    [
        "Story Mode",
        "Country Overview",
        "Network Drill-down",
        "Temporal Evolution",
        "Club Intelligence",
        "Community Patterns",
        "Transfer Evidence",
    ]
)

with story_tab:
    st.subheader("From global football markets to individual transfers")
    st.markdown(
        '<div class="section-note">This opening view is designed for presentation: start with geography, reveal cross-border flows, then move into clusters, club positions, and player-level evidence.</div>',
        unsafe_allow_html=True,
    )

    story_left, story_right = st.columns([1.25, 1])
    with story_left:
        story_map = px.choropleth(
            country_map_summary,
            locations="map_location",
            locationmode="country names",
            color="clubs",
            hover_name="display_country",
            hover_data={
                "original_countries": True,
                "clubs": ":,",
                "total_transfers": ":,",
                "avg_pagerank": ":.5f",
                "clusters": ":,",
                "map_location": False,
                "linked_country": False,
            },
            color_continuous_scale="Tealrose",
            custom_data=["linked_country"],
        )
        story_map.update_geos(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
            lataxis_range=[25, 72],
            lonaxis_range=[-25, 55],
        )
        story_map.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0))
        story_selection = selectable_chart(story_map, "story_country_map")
        story_clicked_country = selected_customdata(story_selection)
        if story_clicked_country:
            set_linked_country(story_clicked_country)

    with story_right:
        lead_country = country_summary.iloc[0] if not country_summary.empty else None
        lead_cluster = cluster_profiles.iloc[0] if not cluster_profiles.empty else None
        st.markdown("#### Narrative checkpoints")
        if lead_country is not None:
            st.metric("Largest mapped market", lead_country["country"], f"{int(lead_country['clubs']):,} clubs")
        if lead_cluster is not None:
            st.metric(
                "Largest Louvain community",
                f"Cluster {int(lead_cluster['cluster'])}",
                f"{int(lead_cluster['clubs']):,} clubs",
            )
        st.markdown(
            """
            1. Geography shows where the transfer network is concentrated.
            2. Flow links show which national markets exchange players most often.
            3. Louvain clusters expose communities that are not always identical to countries.
            4. Embeddings place structurally similar clubs near each other.
            5. Player records verify every visible relationship.
            """
        )

    st.markdown("#### Cross-border transfer flows")
    flow_fig = make_sankey(country_flows)
    st.plotly_chart(flow_fig, use_container_width=True)

    flow_table = add_money_display(country_flows, ["total_fee"])
    if not flow_table.empty:
        flow_table["flow"] = flow_table["source_map_country"] + " to " + flow_table["target_map_country"]
        top_flow_fig = px.bar(
            flow_table.head(12).sort_values("transfers"),
            x="transfers",
            y="flow",
            orientation="h",
            color="transfers",
            color_continuous_scale="Tealrose",
            hover_data={"transfers": ":,", "total_fee_display": True, "flow": False},
        )
        top_flow_fig.update_layout(
            height=360,
            margin=dict(l=8, r=8, t=8, b=8),
            yaxis_title="",
            xaxis_title="Transfers",
            coloraxis_showscale=False,
        )
        st.plotly_chart(top_flow_fig, use_container_width=True)
        flow_display = flow_table[
            ["source_map_country", "target_map_country", "transfers", "total_fee_display"]
        ].rename(
            columns={
                "source_map_country": "source_country",
                "target_map_country": "target_country",
                "total_fee_display": "total_fee",
            }
        )
        st.dataframe(
            flow_display,
            use_container_width=True,
            hide_index=True,
        )

with overview:
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Country distribution")
        st.markdown(
            '<div class="section-note">Click a country on the map if your Streamlit version supports selection, or choose it from the selector below.</div>',
            unsafe_allow_html=True,
        )
        map_fig = px.choropleth(
            country_map_summary,
            locations="map_location",
            locationmode="country names",
            color="clubs",
            hover_name="display_country",
            hover_data={
                "clubs": ":,",
                "total_transfers": ":,",
                "total_fee": False,
                "avg_pagerank": ":.5f",
                "clusters": ":,",
                "original_countries": True,
                "map_location": False,
                "linked_country": False,
            },
            color_continuous_scale="Tealrose",
            custom_data=["linked_country"],
        )
        map_fig.update_geos(showframe=False, showcoastlines=True, projection_type="natural earth")
        map_fig.update_layout(height=560, margin=dict(l=0, r=0, t=8, b=0))
        map_selection = selectable_chart(map_fig, "country_map")
        clicked_country = selected_customdata(map_selection)
        if clicked_country:
            set_linked_country(clicked_country)

        country_options = [""] + country_summary["country"].tolist()
        default_country = st.session_state["linked_country"] if st.session_state["linked_country"] in country_options else ""
        country_focus = st.selectbox(
            "Focus country",
            country_options,
            index=country_options.index(default_country),
            format_func=lambda value: "All countries" if value == "" else value,
        )
        if country_focus and country_focus != st.session_state["linked_country"]:
            set_linked_country(country_focus)
        if country_focus:
            st.info("Next: open Club Intelligence to rank clubs from this country, or Network Drill-down to inspect club relationships.")

    with right:
        st.subheader("Top countries")
        top_country = country_summary.head(15).copy()
        bar_fig = px.bar(
            top_country.sort_values("clubs"),
            x="clubs",
            y="country",
            orientation="h",
            color="total_transfers",
            color_continuous_scale="Viridis",
            hover_data={"total_transfers": ":,", "player_fee_in": False, "player_fee_out": False},
        )
        bar_fig.update_layout(height=420, margin=dict(l=8, r=8, t=8, b=8), yaxis_title="", xaxis_title="Clubs")
        st.plotly_chart(bar_fig, use_container_width=True)

        country_table = add_money_display(country_summary, ["player_fee_in", "player_fee_out", "net_fee_flow"])
        st.dataframe(
            country_table[
                [
                    "country",
                    "original_countries",
                    "clubs",
                    "total_transfers",
                    "clusters",
                    "player_fee_in_display",
                    "player_fee_out_display",
                    "net_fee_flow_display",
                ]
            ].head(20).rename(columns={
                "player_fee_in_display": "incoming_fees",
                "player_fee_out_display": "outgoing_fees",
                "net_fee_flow_display": "net_fees",
            }),
            use_container_width=True,
            hide_index=True,
        )

    if country_focus:
        focus_country_members = country_members(country_focus)
        focus_clubs = filtered_clubs[filtered_clubs["country"].isin(focus_country_members)]
        focus_edges = filtered_edges[
            (filtered_edges["source_country"].isin(focus_country_members))
            | (filtered_edges["target_country"].isin(focus_country_members))
        ]
        st.subheader(f"{country_focus}: club and relationship detail")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clubs", f"{focus_clubs['club'].nunique():,}")
        c2.metric("Relationships", f"{len(focus_edges):,}")
        c3.metric("Transfers in", f"{int(focus_clubs['transfers_in'].sum()):,}")
        c4.metric("Transfers out", f"{int(focus_clubs['transfers_out'].sum()):,}")

        focus_clubs_plot = add_money_display(focus_clubs, ["total_fee_involved"])
        country_club_fig = px.scatter(
            focus_clubs_plot,
            x="embed_x",
            y="embed_y",
            size="pagerank",
            color="louvain_cluster_label",
            hover_name="club",
            hover_data=["degree", "transfers_in", "transfers_out", "total_fee_involved_display"],
            custom_data=["club"],
            size_max=36,
        )
        country_club_fig.update_layout(height=520, margin=dict(l=8, r=8, t=12, b=8))
        selectable_chart(country_club_fig, "country_club_embedding")

with network_tab:
    st.subheader("Relationship network")
    st.markdown(
        '<div class="section-note">Use the controls to move from the macro network into a single club ego network. Hover links for transfer counts, then use Edge detail to send the relationship to Transfer Evidence.</div>',
        unsafe_allow_html=True,
    )
    club_options = [""] + clean_club_list(filtered_clubs["club"])
    if st.session_state["linked_club"] not in club_options:
        st.session_state["linked_club"] = ""
    default_club_index = club_options.index(st.session_state["linked_club"]) if st.session_state["linked_club"] in club_options else 0
    selected_club = st.selectbox(
        "Selected club",
        club_options,
        index=default_club_index,
        format_func=lambda value: "Whole filtered network" if value == "" else value,
    )
    if selected_club and selected_club != st.session_state["linked_club"]:
        st.session_state["linked_club"] = selected_club
    network_context_clubs = filter_clubs(club_embeddings, active_countries, selected_clusters, "") if selected_club else filtered_clubs
    visible_edges = edge_subset(club_edges, network_context_clubs, min_relationship, selected_club)

    net_fig = make_network(visible_edges, network_context_clubs, selected_club, max_edges)
    network_left, network_right = st.columns([1.7, 1])
    with network_left:
        network_selection = selectable_chart(net_fig, "relationship_network")
        clicked_club = selected_customdata(network_selection)
        if clicked_club:
            set_linked_club(clicked_club)

    with network_right:
        st.markdown(
            """
            <div style="
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 14px 16px;
                background: #f8fafc;
                margin-bottom: 18px;
            ">
                <h4 style="margin: 0 0 10px 0;">How to read this network</h4>
                <p style="margin: 0 0 8px 0;"><b>Points</b> represent football clubs.</p>
                <p style="margin: 0 0 8px 0;"><b>Lines</b> represent transfer relationships between clubs.</p>
                <p style="margin: 0 0 8px 0;"><b>Larger points</b> indicate higher PageRank, meaning a more central club in the transfer network.</p>
                <p style="margin: 0 0 8px 0;"><b>Thicker lines</b> indicate more transfers between two clubs.</p>
                <p style="margin: 0 0 8px 0;"><b>Colours</b> separate countries within the current filtered network.</p>
                <p style="margin: 0 0 8px 0;"><b>Selected clubs</b> are enlarged with a star marker; connected lines become brighter.</p>
                <p style="margin: 0;"><b>Hover</b> for details, or <b>click a point</b> to show the club profile.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Linked club profile")
        if selected_club:
            profile = club_embeddings[club_embeddings["club"] == selected_club]
            if not profile.empty:
                profile = profile.iloc[0]
                st.metric("Club", profile["club"])
                st.metric("Country", profile["country"])
                r1, r2 = st.columns(2)
                r1.metric("PageRank", f"{profile['pagerank']:.5f}")
                r2.metric("Cluster", int(profile["louvain_cluster"]))
                r3, r4 = st.columns(2)
                r3.metric("Transfers in", f"{int(profile['transfers_in']):,}")
                r4.metric("Transfers out", f"{int(profile['transfers_out']):,}")
        else:
            st.caption("Click a node or choose a club to show its profile here.")

        st.markdown("#### Edge detail")
        edge_options = []
        edge_lookup = {}
        for row in visible_edges.sort_values(["transfer_count", "total_transfer_fee"], ascending=False).head(200).itertuples():
            label = f"{row.source_club} -> {row.target_club} ({int(row.transfer_count)} transfers)"
            edge_options.append(label)
            edge_lookup[label] = (row.source_club, row.target_club)
        edge_options = [""] + edge_options
        edge_choice = st.selectbox(
            "Relationship",
            edge_options,
            format_func=lambda value: "Select an edge" if value == "" else value,
        )
        if edge_choice:
            source_edge, target_edge = edge_lookup[edge_choice]
            set_linked_edge(source_edge, target_edge)
            st.success(f"Linked edge: {source_edge} to {target_edge}. Open Transfer Evidence to inspect the underlying player records.")
            edge_records = season_transfers[
                (season_transfers["source_club"] == source_edge)
                & (season_transfers["target_club"] == target_edge)
            ]
            edge_records_display = add_money_display(edge_records, ["transfer_fee_clean"])
            st.dataframe(
                edge_records_display.sort_values(["season", "transfer_fee_clean"], ascending=[False, False]).head(20)[
                    [
                        "season",
                        "window",
                        "player_name",
                        "player_pos",
                        "transfer_type",
                        "transfer_fee_clean_display",
                    ]
                ].rename(columns={"transfer_fee_clean_display": "transfer_fee"}),
                use_container_width=True,
                hide_index=True,
            )

    visible_edges_display = add_money_display(visible_edges, ["total_transfer_fee", "average_transfer_fee", "max_transfer_fee"])
    edge_table_columns = [
        "source_club",
        "target_club",
        "transfer_count",
        "total_transfer_fee_display",
        "average_transfer_fee_display",
        "max_transfer_fee_display",
        "first_season",
        "last_season",
        "source_country",
        "target_country",
    ]
    edge_table_columns = [col for col in edge_table_columns if col in visible_edges_display.columns]
    st.dataframe(
        visible_edges_display.sort_values(["transfer_count", "total_transfer_fee"], ascending=False)
        .head(150)
        [edge_table_columns].rename(columns={
            "total_transfer_fee_display": "total_fee",
            "average_transfer_fee_display": "average_fee",
            "max_transfer_fee_display": "max_fee",
        }),
        use_container_width=True,
        hide_index=True,
    )

with temporal_tab:
    st.subheader("Season-by-season network evolution")
    st.markdown(
        '<div class="section-note">This view keeps club positions fixed using the learned embedding coordinates, then reveals which transfer links are active in each season.</div>',
        unsafe_allow_html=True,
    )
    temporal_cols = st.columns([1, 1, 1])
    temporal_year = temporal_cols[0].slider("Playback season", season_min, season_max, season_range[0])
    temporal_edges = temporal_cols[1].slider("Edges shown in selected season", 30, 300, 140, step=10)
    temporal_focus = temporal_cols[2].checkbox("Use linked club as focus", value=bool(selected_club))
    temporal_club = selected_club if temporal_focus else ""

    temporal_rows = season_transfers.copy()
    allowed_clubs_for_time = set(filtered_clubs["club"])
    temporal_rows = temporal_rows[
        temporal_rows["source_club"].isin(allowed_clubs_for_time)
        | temporal_rows["target_club"].isin(allowed_clubs_for_time)
    ]
    temporal_fig = make_temporal_network(
        temporal_rows,
        filtered_clubs,
        temporal_year,
        selected_club=temporal_club,
        max_edges=temporal_edges,
    )
    temporal_selection = selectable_chart(temporal_fig, "temporal_network")
    temporal_clicked_club = selected_customdata(temporal_selection)
    if temporal_clicked_club:
        set_linked_club(temporal_clicked_club)

    year_summary = (
        temporal_rows.groupby("season", as_index=False)
        .agg(transfers=("transfer_id", "count"), fees=("transfer_fee_clean", "sum"))
        .sort_values("season")
    )
    if not year_summary.empty:
        year_summary = add_money_display(year_summary, ["fees"])
        year_summary["selected"] = year_summary["season"].eq(temporal_year)
        timeline_fig = px.bar(
            year_summary,
            x="season",
            y="transfers",
            color="selected",
            color_discrete_map={True: "#d97706", False: "#94a3b8"},
            hover_data={"fees_display": True, "fees": False, "selected": False},
        )
        timeline_fig.update_layout(
            height=260,
            margin=dict(l=8, r=8, t=12, b=8),
            showlegend=False,
            xaxis_title="Season",
            yaxis_title="Transfers",
        )
        st.plotly_chart(timeline_fig, use_container_width=True)

with club_tab:
    st.subheader("Club intelligence")
    rank_options = ["PageRank", "Transfers", "Total fees", "Degree"]
    if hasattr(st, "segmented_control"):
        ranking_mode = st.segmented_control("Rank by", rank_options, default="PageRank")
    else:
        ranking_mode = st.radio("Rank by", rank_options, horizontal=True)
    rank_col = {
        "PageRank": "pagerank",
        "Transfers": "total_transfers",
        "Total fees": "total_fee_involved",
        "Degree": "degree",
    }[ranking_mode]
    top_clubs = filtered_clubs.sort_values(rank_col, ascending=False).head(40)
    top_clubs_plot = add_money_display(top_clubs, ["total_fee_involved"])

    rank_fig = px.bar(
        top_clubs_plot.sort_values(rank_col),
        x=rank_col,
        y="club",
        color="country",
        orientation="h",
        hover_data=["louvain_cluster_label", "transfers_in", "transfers_out", "degree", "pagerank", "total_fee_involved_display"],
    )
    rank_fig.update_layout(height=760, margin=dict(l=8, r=8, t=12, b=8), yaxis_title="", xaxis_title=ranking_mode)
    if ranking_mode == "Total fees":
        rank_fig.update_xaxes(tickprefix="\N{EURO SIGN}", tickformat="~s")
    st.plotly_chart(rank_fig, use_container_width=True)

    detail_club_options = clean_club_list(filtered_clubs["club"])
    detail_club = ""
    if detail_club_options:
        detail_default = 0
        if selected_club in detail_club_options:
            detail_default = detail_club_options.index(selected_club)
        detail_club = st.selectbox("Club profile", detail_club_options, index=detail_default)
    else:
        st.warning("No clubs match the current filters.")
    if detail_club:
        profile = club_embeddings[club_embeddings["club"] == detail_club].iloc[0]
        st.info("Next: choose this club in Network Drill-down to see its strongest relationships, then open Transfer Evidence for player-level records.")
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Country", profile["country"])
        p2.metric("Cluster", int(profile["louvain_cluster"]))
        p3.metric("PageRank", f"{profile['pagerank']:.5f}")
        p4.metric("Transfers in", f"{int(profile['transfers_in']):,}")
        p5.metric("Transfers out", f"{int(profile['transfers_out']):,}")

        club_rel = club_edges[
            (club_edges["source_club"] == detail_club)
            | (club_edges["target_club"] == detail_club)
        ].copy()
        club_rel["direction"] = club_rel.apply(
            lambda row: "Outgoing" if row["source_club"] == detail_club else "Incoming",
            axis=1,
        )
        club_rel["partner"] = club_rel.apply(
            lambda row: row["target_club"] if row["source_club"] == detail_club else row["source_club"],
            axis=1,
        )
        club_rel_plot = add_money_display(club_rel, ["total_transfer_fee"])
        rel_fig = px.bar(
            club_rel_plot.sort_values("transfer_count", ascending=False).head(25).sort_values("transfer_count"),
            x="transfer_count",
            y="partner",
            color="direction",
            orientation="h",
            hover_data=["total_transfer_fee_display", "first_season", "last_season"],
        )
        rel_fig.update_layout(height=560, margin=dict(l=8, r=8, t=12, b=8), yaxis_title="", xaxis_title="Transfers")
        st.plotly_chart(rel_fig, use_container_width=True)

        similar = similar_clubs(club_embeddings, detail_club)
        if not similar.empty:
            st.markdown("#### Structurally similar clubs")
            st.caption(
                "Similarity is calculated from the learned Node2Vec embedding vectors, so it reflects network position rather than only country or transfer volume."
            )
            st.dataframe(
                similar,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "pagerank": st.column_config.NumberColumn("PageRank", format="%.5f"),
                    "embedding_distance": st.column_config.NumberColumn("Embedding distance", format="%.4f"),
                },
            )

with community_tab:
    st.subheader("Louvain community patterns")
    st.markdown(
        '<div class="section-note">This section turns clustering output into interpretable evidence: community size, dominant countries, leading clubs, and internal/external relationships.</div>',
        unsafe_allow_html=True,
    )
    if cluster_profiles.empty:
        st.warning("No community data is available for the current filters.")
    else:
        cluster_fig = px.scatter(
            cluster_profiles,
            x="external_relationships",
            y="internal_relationships",
            size="clubs",
            color="label",
            hover_name="label",
            hover_data=["cluster", "clubs", "countries", "avg_pagerank", "avg_degree", "top_countries"],
            size_max=58,
        )
        cluster_fig.update_layout(
            height=520,
            margin=dict(l=8, r=8, t=16, b=8),
            xaxis_title="External relationships",
            yaxis_title="Internal relationships",
        )
        st.plotly_chart(cluster_fig, use_container_width=True)

        st.dataframe(
            cluster_profiles[
                [
                    "cluster",
                    "label",
                    "clubs",
                    "countries",
                    "avg_pagerank",
                    "avg_degree",
                    "internal_relationships",
                    "external_relationships",
                    "top_countries",
                    "leading_clubs",
                    "main_transfer_directions",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "avg_pagerank": st.column_config.NumberColumn("Avg PageRank", format="%.5f"),
                "avg_degree": st.column_config.NumberColumn("Avg degree", format="%.2f"),
            },
        )

        selected_cluster_profile = st.selectbox(
            "Inspect one community",
            cluster_profiles["cluster"].tolist(),
            format_func=lambda value: f"Cluster {value}",
        )
        cluster_clubs = filtered_clubs[filtered_clubs["louvain_cluster"] == selected_cluster_profile]
        if not cluster_clubs.empty:
            cluster_clubs_plot = add_money_display(cluster_clubs, ["total_fee_involved"])
            cluster_embed = px.scatter(
                cluster_clubs_plot,
                x="embed_x",
                y="embed_y",
                size="pagerank",
                color="country",
                hover_name="club",
                hover_data=["degree", "transfers_in", "transfers_out", "total_fee_involved_display"],
                size_max=34,
            )
            cluster_embed.update_layout(height=520, margin=dict(l=8, r=8, t=12, b=8))
            st.plotly_chart(cluster_embed, use_container_width=True)

with evidence_tab:
    st.subheader("Player-level transfer evidence")
    st.markdown(
        '<div class="section-note">Every row below is the evidence behind a club edge: player, season, direction, fee, transfer type, source club, and target club.</div>',
        unsafe_allow_html=True,
    )
    e1, e2, e3 = st.columns([1, 1, 1])
    evidence_club_options = [""] + clean_club_list(filtered_clubs["club"])
    source_default = (
        evidence_club_options.index(st.session_state["linked_edge_source"])
        if st.session_state["linked_edge_source"] in evidence_club_options
        else 0
    )
    target_default = (
        evidence_club_options.index(st.session_state["linked_edge_target"])
        if st.session_state["linked_edge_target"] in evidence_club_options
        else 0
    )
    source_filter = e1.selectbox("Source club", evidence_club_options, index=source_default)
    target_filter = e2.selectbox("Target club", evidence_club_options, index=target_default)
    player_filter = e3.text_input("Player search")

    evidence = season_transfers.copy()
    allowed_clubs = set(filtered_clubs["club"])
    evidence = evidence[
        evidence["source_club"].isin(allowed_clubs)
        | evidence["target_club"].isin(allowed_clubs)
    ]
    if source_filter:
        evidence = evidence[evidence["source_club"] == source_filter]
    if target_filter:
        evidence = evidence[evidence["target_club"] == target_filter]
    if player_filter:
        evidence = evidence[evidence["player_name"].str.contains(player_filter, case=False, na=False)]

    if selected_club and not source_filter and not target_filter:
        evidence = evidence[
            (evidence["source_club"] == selected_club)
            | (evidence["target_club"] == selected_club)
        ]

    if source_filter and target_filter:
        st.info(f"Showing linked edge evidence: {source_filter} to {target_filter}. These rows are the player-level records behind the selected club relationship.")

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Rows", f"{len(evidence):,}")
    t2.metric("Unique players", f"{evidence['player_name'].nunique():,}")
    t3.metric("Total fees", money(evidence["transfer_fee_clean"].sum()))
    t4.metric("Loans", f"{int(evidence['is_loan'].sum()) if 'is_loan' in evidence else 0:,}")

    by_year = evidence.groupby("season", as_index=False).agg(
        transfers=("transfer_id", "count"),
        fees=("transfer_fee_clean", "sum"),
    )
    if not by_year.empty:
        year_fig = go.Figure()
        year_fig.add_bar(x=by_year["season"], y=by_year["transfers"], name="Transfers", marker_color="#2563eb")
        year_fig.add_scatter(
            x=by_year["season"],
            y=by_year["fees"],
            name="Fees",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#b45309", width=3),
        )
        year_fig.update_layout(
            height=380,
            margin=dict(l=8, r=8, t=12, b=8),
            yaxis=dict(title="Transfers"),
            yaxis2=dict(title="Fees", overlaying="y", side="right", tickprefix="\N{EURO SIGN}", tickformat="~s"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(year_fig, use_container_width=True)

    evidence_display = add_money_display(evidence, ["transfer_fee_clean", "market_value_clean"])
    evidence_display["window"] = evidence_display["window"].map(format_window)
    st.dataframe(
        evidence_display.sort_values(["season", "transfer_fee_clean"], ascending=[False, False])
        [
            [
                "season",
                "window",
                "player_name",
                "player_age",
                "player_nation",
                "player_pos",
                "source_club",
                "target_club",
                "source_country",
                "target_country",
                "transfer_type",
                "transfer_fee_clean_display",
                "market_value_clean_display",
                "league_name",
            ]
        ].rename(columns={
            "transfer_fee_clean_display": "transfer_fee",
            "market_value_clean_display": "market_value",
        }),
        use_container_width=True,
        hide_index=True,
    )

