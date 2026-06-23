"""
Delhivery Network Intelligence Dashboard
=========================================
Run:  streamlit run app.py
Requires: final_delivery_data.csv, corridor_stats.csv, G_all.pkl
          ftl_reg.pkl, carting_reg.pkl, le_src.pkl, le_dst.pkl, le_tod.pkl
          (see README for how to export models from notebook 03)
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delhivery Network Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME HELPERS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "primary"  : "#E8401C",   # Delhivery brand red-orange
    "secondary": "#1C3557",   # deep navy
    "accent"   : "#F5A623",   # amber
    "good"     : "#27AE60",
    "warn"     : "#F39C12",
    "bad"      : "#E74C3C",
    "neutral"  : "#7F8C8D",
    "bg"       : "#F8F9FA",
}

PROFILE_COLORS = {
    "CRITICAL"                          : "#E74C3C",
    "SWITCH TO FTL"                     : "#E67E22",
    "CHRONIC DELAY — Monitor"           : "#F1C40F",
    "HIGH VOLUME — FTL Opportunity"     : "#1ABC9C",
    "STABLE"                            : "#27AE60",
}

def metric_card(label, value, delta=None, delta_label="", color=COLORS["secondary"]):
    """Render a styled metric inside a markdown block."""
    delta_html = ""
    if delta is not None:
        col = COLORS["bad"] if delta > 0 else COLORS["good"]
        arrow = "▲" if delta > 0 else "▼"
        delta_html = f'<div style="font-size:13px;color:{col}">{arrow} {delta_label}</div>'
    st.markdown(
        f"""
        <div style="background:#fff;border-radius:10px;padding:18px 22px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.08);border-left:4px solid {color}">
            <div style="font-size:13px;color:{COLORS['neutral']};margin-bottom:4px">{label}</div>
            <div style="font-size:26px;font-weight:700;color:{color}">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS  (cached so they run once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("final_delivery_data.csv")
    cs = pd.read_csv("corridor_stats.csv")
    return df, cs


@st.cache_resource
def load_graph():
    with open("G_all.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_models():
    """
    Load the trained models exported from notebook 03.
    Add this cell at the end of notebook 03 to export:

        import pickle
        pickle.dump(ftl_reg,     open("ftl_reg.pkl",     "wb"))
        pickle.dump(carting_reg, open("carting_reg.pkl", "wb"))
        pickle.dump(le_src,      open("le_src.pkl",      "wb"))
        pickle.dump(le_dst,      open("le_dst.pkl",      "wb"))
        pickle.dump(le_tod,      open("le_tod.pkl",      "wb"))
    """
    try:
        ftl_reg     = pickle.load(open("ftl_reg.pkl",     "rb"))
        carting_reg = pickle.load(open("carting_reg.pkl", "rb"))
        le_src      = pickle.load(open("le_src.pkl",      "rb"))
        le_dst      = pickle.load(open("le_dst.pkl",      "rb"))
        le_tod      = pickle.load(open("le_tod.pkl",      "rb"))
        return ftl_reg, carting_reg, le_src, le_dst, le_tod, True
    except FileNotFoundError:
        return None, None, None, None, None, False


@st.cache_data
def build_hub_table(_df):
    """Aggregate trip-level data to one row per source facility."""
    hub = _df.groupby(["source_center", "source_name"]).agg(
        trips          =("trip_uuid",       "count"),
        median_factor  =("segment_factor",  "median"),
        pct_delayed    =("is_delayed",       "mean"),
        total_excess   =("time_gap",         "sum"),
        out_corridors  =("destination_center","nunique"),
    ).reset_index()
    hub = hub.rename(columns={
        "source_center": "center",
        "source_name"  : "name",
    })
    hub["pct_delayed_fmt"]  = (hub["pct_delayed"] * 100).round(1)
    hub["excess_hrs"]       = (hub["total_excess"] / 60).round(0).astype(int)
    hub["median_factor_fmt"]= hub["median_factor"].round(2)
    return hub.sort_values("total_excess", ascending=False).reset_index(drop=True)


@st.cache_data
def build_corridor_table(_df, _cs):
    """Add intervention profile to corridor_stats."""
    q75_trips  = _cs["trip_count"].quantile(0.75)

    def profile(row):
        chronic     = row["median_segment_factor"] > 1.2
        high_volume = row["trip_count"] > q75_trips
        if chronic and high_volume:
            return "CRITICAL"
        elif chronic:
            return "CHRONIC DELAY — Monitor"
        elif high_volume:
            return "HIGH VOLUME — FTL Opportunity"
        else:
            return "STABLE"

    cs2 = _cs.copy()
    cs2["profile"] = cs2.apply(profile, axis=1)

    # Pull human-readable names from df
    src_names = (_df[["source_center","source_name"]]
                 .drop_duplicates("source_center")
                 .rename(columns={"source_center":"source_center","source_name":"src_name"}))
    dst_names = (_df[["destination_center","destination_name"]]
                 .drop_duplicates("destination_center")
                 .rename(columns={"destination_center":"destination_center","destination_name":"dst_name"}))
    cs2 = cs2.merge(src_names, on="source_center", how="left")
    cs2 = cs2.merge(dst_names, on="destination_center", how="left")
    cs2["corridor_label"] = cs2["src_name"] + "  →  " + cs2["dst_name"]
    cs2["excess_hrs"]     = (cs2["total_excess_time"] / 60).round(0).astype(int)
    return cs2


@st.cache_data
def build_graph_metrics(_df):
    """Compute betweenness and PageRank on G_all — cached so it runs once."""
    G = nx.DiGraph()
    cs_temp = _df.groupby(["source_center","destination_center"]).agg(
        weight    =("segment_factor", "median"),
        pct_delayed=("is_delayed",    "mean"),
    ).reset_index()
    for _, row in cs_temp.iterrows():
        G.add_edge(row["source_center"], row["destination_center"],
                   weight=row["weight"], pct_delayed=row["pct_delayed"])

    betweenness = nx.betweenness_centrality(G, weight="weight")
    pagerank    = nx.pagerank(G, weight="weight")
    return betweenness, pagerank


@st.cache_data
def get_facility_list(_df):
    fac = pd.concat([
        _df[["source_center","source_name"]].rename(
            columns={"source_center":"center","source_name":"name"}),
        _df[["destination_center","destination_name"]].rename(
            columns={"destination_center":"center","destination_name":"name"}),
    ]).drop_duplicates("center")
    fac["label"] = fac["name"] + "  [" + fac["center"] + "]"
    return fac.sort_values("name").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
df, cs = load_data()
G_all  = load_graph()
ftl_reg, carting_reg, le_src, le_dst, le_tod, models_ready = load_models()

hub_table       = build_hub_table(df)
corridor_table  = build_corridor_table(df, cs)
facilities      = get_facility_list(df)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAV
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center;padding:12px 0 20px">
            <span style="font-size:32px">🚚</span><br>
            <span style="font-size:18px;font-weight:700;color:{COLORS['primary']}">
                Delhivery
            </span><br>
            <span style="font-size:12px;color:{COLORS['neutral']}">
                Network Intelligence
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        [
            "📊  Network Overview",
            "🏭  Hub Explorer",
            "🛣️  Corridor Audit",
            "🤖  ETA & Route Predictor",
            "📈  Sensitivity Analysis",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:11px;color:{COLORS['neutral']}'>"
        "103,014 trips · 1,590 facilities · 2,508 corridors<br>"
        "Sep 2018 · Delhivery logistics network"
        "</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — NETWORK OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊  Network Overview":

    st.markdown(f"## 📊 Network Overview")
    st.markdown(
        "High-level health of the Delhivery logistics network across "
        "**103,014 trip segments** from September 2018."
    )

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Total Trips", "103,014", color=COLORS["secondary"])
    with c2:
        metric_card("Facilities (Nodes)", "1,590", color=COLORS["secondary"])
    with c3:
        metric_card("Corridors (Edges)", "2,508", color=COLORS["secondary"])
    with c4:
        metric_card("Trips Delayed", "84.4%",
                    delta=1, delta_label="vs OSRM estimate", color=COLORS["bad"])
    with c5:
        metric_card("Median Delay Factor", "1.69×",
                    delta=1, delta_label="actual vs OSRM", color=COLORS["warn"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: delay distribution + delay by hour ─────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Delay Factor Distribution")
        st.caption("segment_actual_time / segment_osrm_time. Values >1 mean slower than OSRM predicted.")
        capped = df[df["segment_factor"] <= df["segment_factor"].quantile(0.99)]
        fig = px.histogram(
            capped, x="segment_factor", nbins=80,
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig.add_vline(x=1.0, line_dash="dash", line_color=COLORS["good"],
                      annotation_text="OSRM baseline", annotation_position="top right")
        fig.add_vline(x=1.2, line_dash="dash", line_color=COLORS["bad"],
                      annotation_text="Chronic threshold", annotation_position="top left")
        fig.update_layout(
            xaxis_title="Delay Factor (actual / OSRM)",
            yaxis_title="Trip Count",
            margin=dict(t=10, b=40),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### Median Delay Factor by Hour of Day")
        st.caption("Hourly pattern of delays across all corridors and route types.")
        hour_delay = df.groupby("hour_of_day")["segment_factor"].median().reset_index()
        fig2 = px.line(
            hour_delay, x="hour_of_day", y="segment_factor",
            markers=True, color_discrete_sequence=[COLORS["primary"]],
        )
        fig2.add_hline(y=1.0, line_dash="dash", line_color=COLORS["good"])
        fig2.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Median Delay Factor",
            xaxis=dict(tickmode="linear", dtick=2),
            margin=dict(t=10, b=40),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 3: FTL vs Carting + excess time by day ────────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown("#### Delay Factor: FTL vs Carting by Hour")
        st.caption("Carting shows higher delay variability; FTL is faster but still 1.6–1.8× OSRM.")
        rtype_hour = (df.groupby(["route_type","hour_of_day"])["segment_factor"]
                      .median().reset_index())
        fig3 = px.line(
            rtype_hour, x="hour_of_day", y="segment_factor",
            color="route_type",
            color_discrete_map={"FTL": COLORS["secondary"], "Carting": COLORS["accent"]},
            markers=True,
        )
        fig3.add_hline(y=1.0, line_dash="dash", line_color=COLORS["good"])
        fig3.update_layout(
            xaxis_title="Hour of Day", yaxis_title="Median Delay Factor",
            xaxis=dict(tickmode="linear", dtick=2),
            margin=dict(t=10, b=40), plot_bgcolor="white",
            legend=dict(title="Route Type"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        st.markdown("#### Cumulative Excess Delay by Day of Week")
        st.caption("Total minutes above OSRM estimate accumulated each day.")
        df["dow_label"] = df["day_of_week"].map(
            {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
        excess_dow = df.groupby("dow_label")["time_gap"].sum().reset_index()
        order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        excess_dow["dow_label"] = pd.Categorical(excess_dow["dow_label"],
                                                  categories=order, ordered=True)
        excess_dow = excess_dow.sort_values("dow_label")
        fig4 = px.bar(
            excess_dow, x="dow_label", y="time_gap",
            color_discrete_sequence=[COLORS["secondary"]],
        )
        fig4.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Total Excess Delay (min)",
            margin=dict(t=10, b=40), plot_bgcolor="white",
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Summary callout ───────────────────────────────────────────────────────
    st.info(
        "⚠️  **20.7 million minutes** of cumulative excess delay in 17 days. "
        "The top 5 hubs alone account for **~70%** of all excess delay. "
        "See the **Hub Explorer** tab for a ranked breakdown."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HUB EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏭  Hub Explorer":

    st.markdown("## 🏭 Hub Explorer")
    st.markdown(
        "Rank and drill into individual facilities by their structural network "
        "position and SLA breach contribution."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        top_n = st.slider("Show top N hubs by excess delay", 10, 100, 30, step=10)
    with c2:
        min_trips = st.number_input("Min trips", value=100, step=50)
    with c3:
        sort_by = st.selectbox("Sort by", ["total_excess", "median_factor", "pct_delayed", "trips"])

    filtered = (hub_table[hub_table["trips"] >= min_trips]
                .sort_values(sort_by, ascending=False)
                .head(top_n))

    # ── Top-hub bar chart ──────────────────────────────────────────────────
    fig = px.bar(
        filtered.sort_values("total_excess"),
        x="total_excess", y="name",
        orientation="h",
        color="median_factor",
        color_continuous_scale=["#27AE60","#F39C12","#E74C3C"],
        hover_data={"trips":True,"pct_delayed_fmt":True,"median_factor_fmt":True},
        labels={
            "total_excess"   : "Total Excess Delay (min)",
            "name"           : "Facility",
            "median_factor"  : "Delay Factor",
        },
    )
    fig.update_layout(
        height=max(400, top_n * 22),
        margin=dict(t=10, b=40, l=260),
        plot_bgcolor="white",
        coloraxis_colorbar=dict(title="Delay Factor"),
        yaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Hub drilldown ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Corridor Drilldown")
    st.caption("Select a facility to see all outgoing corridors and their delay profiles.")

    hub_options = filtered["name"].tolist()
    selected_name = st.selectbox("Select facility", hub_options)
    selected_center = filtered[filtered["name"] == selected_name]["center"].values[0]

    # Filter corridors
    out_corr = corridor_table[corridor_table["source_center"] == selected_center].copy()
    in_corr  = corridor_table[corridor_table["destination_center"] == selected_center].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Outgoing corridors from {selected_name}** ({len(out_corr)} corridors)")
        if len(out_corr) > 0:
            fig_out = px.bar(
                out_corr.sort_values("total_excess_time", ascending=False).head(20),
                x="dst_name", y="median_segment_factor",
                color="pct_delayed",
                color_continuous_scale=["#27AE60","#E74C3C"],
                labels={
                    "dst_name"             : "Destination",
                    "median_segment_factor": "Delay Factor",
                    "pct_delayed"          : "% Delayed",
                },
            )
            fig_out.update_layout(
                xaxis_tickangle=-35,
                margin=dict(t=10, b=120),
                plot_bgcolor="white",
                height=380,
            )
            st.plotly_chart(fig_out, use_container_width=True)

            # KPIs for selected hub
            k1, k2, k3 = st.columns(3)
            with k1:
                metric_card(
                    "Total Excess Delay",
                    f"{out_corr['excess_hrs'].sum():,} hrs",
                    color=COLORS["bad"],
                )
            with k2:
                metric_card(
                    "Avg Delay Factor",
                    f"{out_corr['median_segment_factor'].mean():.2f}×",
                    color=COLORS["warn"],
                )
            with k3:
                metric_card(
                    "Chronic Corridors",
                    f"{out_corr['is_chronic_delay'].sum()} / {len(out_corr)}",
                    color=COLORS["bad"],
                )
        else:
            st.info("No outgoing corridors found for this facility.")

    with col2:
        st.markdown(f"**Incoming corridors to {selected_name}** ({len(in_corr)} corridors)")
        if len(in_corr) > 0:
            fig_in = px.bar(
                in_corr.sort_values("total_excess_time", ascending=False).head(20),
                x="src_name", y="median_segment_factor",
                color="pct_delayed",
                color_continuous_scale=["#27AE60","#E74C3C"],
                labels={
                    "src_name"             : "Source",
                    "median_segment_factor": "Delay Factor",
                    "pct_delayed"          : "% Delayed",
                },
            )
            fig_in.update_layout(
                xaxis_tickangle=-35,
                margin=dict(t=10, b=120),
                plot_bgcolor="white",
                height=380,
            )
            st.plotly_chart(fig_in, use_container_width=True)
        else:
            st.info("No incoming corridors found for this facility.")

    # ── Delay by hour for selected hub ────────────────────────────────────────
    st.markdown(f"#### ⏰ Delay Pattern by Hour — {selected_name}")
    hub_trips = df[df["source_center"] == selected_center]
    if len(hub_trips) > 0:
        hour_fig_data = (hub_trips.groupby(["hour_of_day","route_type"])["segment_factor"]
                         .median().reset_index())
        fig_hour = px.line(
            hour_fig_data, x="hour_of_day", y="segment_factor",
            color="route_type",
            color_discrete_map={"FTL": COLORS["secondary"], "Carting": COLORS["accent"]},
            markers=True,
            labels={"hour_of_day":"Hour of Day","segment_factor":"Median Delay Factor"},
        )
        fig_hour.add_hline(y=1.0, line_dash="dash", line_color=COLORS["good"])
        fig_hour.add_hline(y=1.2, line_dash="dot",  line_color=COLORS["bad"],
                           annotation_text="Chronic threshold")
        fig_hour.update_layout(
            xaxis=dict(tickmode="linear", dtick=2),
            margin=dict(t=10, b=40), plot_bgcolor="white",
        )
        st.plotly_chart(fig_hour, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CORRIDOR AUDIT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🛣️  Corridor Audit":

    st.markdown("## 🛣️ Corridor Audit")
    st.markdown(
        "Every corridor ranked by SLA breach contribution and classified "
        "into an intervention profile."
    )

    # ── Profile filter ────────────────────────────────────────────────────────
    all_profiles = corridor_table["profile"].unique().tolist()
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_profiles = st.multiselect(
            "Filter by intervention profile",
            all_profiles,
            default=all_profiles,
        )
    with col2:
        min_corridor_trips = st.number_input("Min corridor trips", value=50, step=25)

    filtered_corr = corridor_table[
        (corridor_table["profile"].isin(selected_profiles)) &
        (corridor_table["trip_count"] >= min_corridor_trips)
    ].copy()

    # ── Profile summary cards ─────────────────────────────────────────────────
    st.markdown("### Profile Summary")
    profile_counts = (corridor_table["profile"].value_counts()
                      .reindex(PROFILE_COLORS.keys(), fill_value=0))
    cols = st.columns(len(PROFILE_COLORS))
    for col, (prof, color) in zip(cols, PROFILE_COLORS.items()):
        with col:
            count = profile_counts.get(prof, 0)
            metric_card(prof, str(count) + " corridors", color=color)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scatter: delay factor vs trips, coloured by profile ───────────────────
    st.markdown("### Delay Factor vs Trip Volume")
    st.caption(
        "Each dot is a corridor. Size = total excess delay. "
        "Top-right quadrant = highest intervention priority."
    )

    # Clip size column to positive values only — negative excess time causes plotly error
    filtered_corr = filtered_corr.copy()
    filtered_corr["size_col"] = filtered_corr["total_excess_time"].clip(lower=0.1)

    fig_scatter = px.scatter(
        filtered_corr,
        x="trip_count",
        y="median_segment_factor",
        color="profile",
        size="size_col",
        size_max=30,
        color_discrete_map=PROFILE_COLORS,
        hover_data={
            "corridor_label"       : True,
            "trip_count"           : True,
            "median_segment_factor": ":.2f",
            "pct_delayed"          : ":.1%",
            "excess_hrs"           : True,
            "size_col"             : False,   # hide the internal size column from tooltip
        },
        labels={
            "trip_count"           : "Trip Count",
            "median_segment_factor": "Median Delay Factor",
            "profile"              : "Profile",
        },
    )
    fig_scatter.add_hline(y=1.2, line_dash="dash", line_color="grey",
                        annotation_text="Chronic threshold (1.2×)")
    fig_scatter.update_layout(
        height=500, plot_bgcolor="white",
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Top-20 corridors table ────────────────────────────────────────────────
    st.markdown("### Top 20 Corridors by Excess Delay")
    display_cols = {
        "corridor_label"       : "Corridor",
        "profile"              : "Profile",
        "trip_count"           : "Trips",
        "median_segment_factor": "Delay Factor",
        "pct_delayed"          : "% Delayed",
        "excess_hrs"           : "Excess Delay (hrs)",
    }
    top20 = (filtered_corr.nlargest(20, "total_excess_time")
             [display_cols.keys()].rename(columns=display_cols))
    top20["Delay Factor"] = top20["Delay Factor"].round(2)
    top20["% Delayed"]    = (top20["% Delayed"] * 100).round(1).astype(str) + "%"

    st.dataframe(
        top20,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Delay Factor"      : st.column_config.ProgressColumn(
                "Delay Factor", min_value=0, max_value=5, format="%.2f"),
            "Excess Delay (hrs)": st.column_config.NumberColumn(format="%d hrs"),
        },
    )

    # ── Download ──────────────────────────────────────────────────────────────
    csv_export = filtered_corr[display_cols.keys()].rename(columns=display_cols).to_csv(index=False)
    st.download_button(
        "⬇️  Download filtered corridor table",
        data=csv_export,
        file_name="corridor_audit.csv",
        mime="text/csv",
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ETA & ROUTE PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖  ETA & Route Predictor":

    st.markdown("## 🤖 ETA & Route Predictor")
    st.markdown(
        "Enter trip context to get a predicted delivery time and a "
        "cost-adjusted FTL vs Carting recommendation."
    )

    if not models_ready:
        st.warning(
            "⚠️  Trained model files not found (`ftl_reg.pkl`, `carting_reg.pkl`, "
            "`le_src.pkl`, `le_dst.pkl`, `le_tod.pkl`). "
            "Add this block at the end of notebook 03 and re-run it:\n\n"
            "```python\n"
            "import pickle\n"
            "pickle.dump(ftl_reg,     open('ftl_reg.pkl',     'wb'))\n"
            "pickle.dump(carting_reg, open('carting_reg.pkl', 'wb'))\n"
            "pickle.dump(le_src,      open('le_src.pkl',      'wb'))\n"
            "pickle.dump(le_dst,      open('le_dst.pkl',      'wb'))\n"
            "pickle.dump(le_tod,      open('le_tod.pkl',      'wb'))\n"
            "```"
        )

        # Show a demo with lookup-based estimates when models aren't available
        st.markdown("---")
        st.markdown("### 📊 Lookup-Based Corridor Estimate *(model not loaded)*")
        st.caption(
            "While the trained models aren't available, you can still look up "
            "historical median statistics for any corridor."
        )

    # ── Corridor lookup (always available) ────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        src_label = st.selectbox(
            "Source Facility",
            facilities["label"].tolist(),
            index=0,
        )
    with col2:
        dst_label = st.selectbox(
            "Destination Facility",
            facilities["label"].tolist(),
            index=min(5, len(facilities) - 1),
        )

    src_center = facilities[facilities["label"] == src_label]["center"].values[0]
    dst_center = facilities[facilities["label"] == dst_label]["center"].values[0]

    # Time/context inputs
    col3, col4, col5 = st.columns(3)
    with col3:
        hour = st.slider("Hour of Dispatch", 0, 23, 10)
    with col4:
        day  = st.slider("Day of Week (0=Mon)", 0, 6, 2)
    with col5:
        osrm_time = st.number_input("OSRM Estimated Time (min)", value=20, min_value=1, max_value=300)

    osrm_dist = st.number_input("OSRM Distance (km)", value=25.0, min_value=0.5, max_value=500.0)

    # Cost parameters
    with st.expander("⚙️  Cost Parameters (tune from your rate card)"):
        cost_per_min = st.slider("SLA Penalty Rate (₹ per delayed minute)", 0.5, 10.0, 2.0, 0.5)
        ftl_premium  = st.slider("FTL Cost Premium (₹ per km over Carting)", 1.0, 20.0, 8.0, 1.0)

    # ── Historical lookup ─────────────────────────────────────────────────────
    corr_match = corridor_table[
        (corridor_table["source_center"]      == src_center) &
        (corridor_table["destination_center"] == dst_center)
    ]

    st.markdown("---")

    if len(corr_match) > 0:
        row = corr_match.iloc[0]
        st.markdown("#### 📌 Historical Corridor Statistics")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            metric_card("Historical Trips",        f"{row['trip_count']:,}", color=COLORS["secondary"])
        with h2:
            metric_card("Median Delay Factor",     f"{row['median_segment_factor']:.2f}×", color=COLORS["warn"])
        with h3:
            metric_card("% Trips Delayed",         f"{row['pct_delayed']*100:.1f}%", color=COLORS["bad"])
        with h4:
            metric_card("Intervention Profile",    row["profile"],
                        color=PROFILE_COLORS.get(row["profile"], COLORS["neutral"]))

        # Historical estimate without model
        hist_estimated_actual = osrm_time * row["median_segment_factor"]
        st.markdown(
            f"**Historical estimate:** OSRM {osrm_time} min × delay factor "
            f"{row['median_segment_factor']:.2f} = **{hist_estimated_actual:.0f} min** expected actual time"
        )

    else:
        st.info("No historical data found for this corridor. "
                "Try a different source–destination combination.")

    # ── ML model prediction (if models loaded) ────────────────────────────────
    if models_ready:
        st.markdown("#### 🤖 ML Model Recommendation")

        tod_map = {0:"Night",1:"Night",2:"Night",3:"Night",4:"Night",5:"Night",
                   6:"Morning",7:"Morning",8:"Morning",9:"Morning",10:"Morning",11:"Morning",
                   12:"Afternoon",13:"Afternoon",14:"Afternoon",15:"Afternoon",16:"Afternoon",17:"Afternoon",
                   18:"Evening",19:"Evening",20:"Evening",21:"Evening",22:"Evening",23:"Evening"}
        tod_str = tod_map[hour]

        # Encode inputs
        src_enc = (le_src.transform([src_center])[0]
                   if src_center in le_src.classes_ else 0)
        dst_enc = (le_dst.transform([dst_center])[0]
                   if dst_center in le_dst.classes_ else 0)
        tod_enc = (le_tod.transform([tod_str])[0]
                   if tod_str in le_tod.classes_ else 0)

        # Look up corridor FTL ratio
        corr_ftl_ratio = (
            df[(df["source_center"]==src_center) &
               (df["destination_center"]==dst_center)]["route_type"]
            .eq("FTL").mean()
        )
        if np.isnan(corr_ftl_ratio):
            corr_ftl_ratio = 0.5

        # Build feature vector matching reg_features from notebook 03
        # Minimal base features (no G_rtype / node graph features available here)
        row_data = {
            "segment_osrm_time"             : osrm_time,
            "segment_osrm_distance"         : osrm_dist,
            "actual_distance_to_destination": osrm_dist * 10,  # approx
            "hour_of_day"                   : hour,
            "day_of_week"                   : day,
            "tod_bucket_enc"                : tod_enc,
            "source_enc"                    : src_enc,
            "destination_enc"               : dst_enc,
            "corridor_ftl_ratio"            : corr_ftl_ratio,
        }

        try:
            import pandas as pd
            row_df = pd.DataFrame([row_data])
            # Align to model's expected columns
            for col in ftl_reg.feature_names_in_:
                if col not in row_df.columns:
                    row_df[col] = 0
            row_df = row_df[ftl_reg.feature_names_in_]

            pred_ftl     = float(ftl_reg.predict(row_df)[0])
            pred_carting = float(carting_reg.predict(row_df)[0])

            time_saved       = pred_carting - pred_ftl
            delay_saved      = cost_per_min * max(time_saved, 0)
            ftl_premium_cost = ftl_premium  * osrm_dist
            net_benefit      = delay_saved  - ftl_premium_cost

            rec  = "FTL"     if (time_saved > 0 and net_benefit > 0) else "CARTING"
            rec_color = COLORS["secondary"] if rec == "FTL" else COLORS["accent"]

            r1, r2, r3 = st.columns(3)
            with r1:
                metric_card("Predicted Time (FTL)",     f"{pred_ftl:.0f} min",    color=COLORS["secondary"])
            with r2:
                metric_card("Predicted Time (Carting)", f"{pred_carting:.0f} min",color=COLORS["accent"])
            with r3:
                metric_card("Recommendation", rec, color=rec_color)

            st.markdown("<br>", unsafe_allow_html=True)
            e1, e2, e3 = st.columns(3)
            with e1:
                metric_card("Time Saved by FTL",    f"{time_saved:+.1f} min", color=COLORS["secondary"])
            with e2:
                metric_card("FTL Cost Premium",     f"₹{ftl_premium_cost:.0f}", color=COLORS["warn"])
            with e3:
                metric_card("Net Benefit of FTL",   f"₹{net_benefit:.0f}",
                            color=COLORS["good"] if net_benefit > 0 else COLORS["bad"])

            # Rationale
            if rec == "FTL":
                rationale = (f"FTL is predicted to be **{time_saved:.1f} min faster**. "
                             f"The delay cost avoided (₹{delay_saved:.0f}) exceeds the FTL "
                             f"cost premium (₹{ftl_premium_cost:.0f}), giving a net benefit of "
                             f"₹{net_benefit:.0f}. **Dispatch FTL.**")
            elif time_saved > 0:
                rationale = (f"FTL is faster by {time_saved:.1f} min, but the FTL premium "
                             f"(₹{ftl_premium_cost:.0f}) exceeds the delay cost saved "
                             f"(₹{delay_saved:.0f}). **Dispatch Carting.**")
            else:
                rationale = (f"Carting is predicted to be **{-time_saved:.1f} min faster** "
                             f"on this corridor. **Dispatch Carting.**")

            st.info(f"📋 **Rationale:** {rationale}")

        except Exception as e:
            st.error(f"Prediction failed: {e}. Ensure the model pkl files match the features "
                     "used in notebook 03.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SENSITIVITY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈  Sensitivity Analysis":

    st.markdown("## 📈 Sensitivity Analysis")
    st.markdown(
        "How does the FTL vs Carting recommendation change as your cost "
        "assumptions shift? Tune the sliders and the heatmap updates live."
    )

    # ── Inputs ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        cpm_min = st.slider("Min SLA penalty (₹/min)",  0.5, 5.0, 0.5, 0.5)
        cpm_max = st.slider("Max SLA penalty (₹/min)",  1.0, 10.0, 5.5, 0.5)
    with col2:
        fpp_min = st.slider("Min FTL premium (₹/km)",  1.0, 5.0,  2.0, 1.0)
        fpp_max = st.slider("Max FTL premium (₹/km)",  2.0, 20.0, 12.0, 1.0)

    cpm_range = np.arange(cpm_min, cpm_max + 0.01, 0.5)
    fpp_range = np.arange(fpp_min, fpp_max + 0.01, 2.0)

    # ── Recompute from corridor_stats ─────────────────────────────────────────
    # We need predicted time savings per corridor — approximate from historical medians:
    # Use FTL vs Carting historical median actual times per corridor
    ftl_hist = (df[df["route_type"]=="FTL"]
                .groupby(["source_center","destination_center"])["segment_actual_time"]
                .median().reset_index().rename(columns={"segment_actual_time":"ftl_time"}))
    cart_hist = (df[df["route_type"]=="Carting"]
                 .groupby(["source_center","destination_center"])["segment_actual_time"]
                 .median().reset_index().rename(columns={"segment_actual_time":"cart_time"}))

    both = ftl_hist.merge(cart_hist, on=["source_center","destination_center"])
    both = both.merge(
        cs[["source_center","destination_center","trip_count","median_osrm_distance"]],
        on=["source_center","destination_center"], how="left"
    )
    both["time_saved"] = both["cart_time"] - both["ftl_time"]

    # Compute sensitivity grid
    sensitivity = []
    for cpm in cpm_range:
        for fpp in fpp_range:
            delay_saved      = cpm * both["time_saved"].clip(lower=0)
            ftl_premium_cost = fpp * both["median_osrm_distance"].fillna(25)
            net_benefit      = delay_saved - ftl_premium_cost
            ftl_rec          = (both["time_saved"] > 0) & (net_benefit > 0)
            sensitivity.append({
                "SLA Penalty (₹/min)": round(cpm, 1),
                "FTL Premium (₹/km)" : round(fpp, 1),
                "% Corridors → FTL"  : round(ftl_rec.mean() * 100, 1),
            })

    sens_df = pd.DataFrame(sensitivity)
    pivot   = sens_df.pivot(
        index="SLA Penalty (₹/min)",
        columns="FTL Premium (₹/km)",
        values="% Corridors → FTL",
    )

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale="RdYlGn",
        zmin=0, zmax=100,
        text=[[f"{v:.0f}%" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        colorbar=dict(title="% Corridors<br>Recommended FTL"),
    ))
    fig_heat.update_layout(
        xaxis_title="FTL Cost Premium (₹/km)",
        yaxis_title="SLA Penalty Rate (₹/min)",
        height=500,
        margin=dict(t=20, b=60),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.caption(
        "Each cell shows what % of corridors with both FTL and Carting history "
        "would be recommended FTL under those cost assumptions. "
        "Green = more FTL justified; Red = Carting dominates."
    )

    # ── Reading guide ─────────────────────────────────────────────────────────
    st.markdown("### How to read this")
    st.markdown(
        """
        - **Move right** (higher FTL premium): fewer corridors justify FTL — the cost advantage shrinks.
        - **Move up** (higher SLA penalty): more corridors justify FTL — the cost of being late rises.
        - The cell matching your rate card gives the exact % of network corridors you should convert to FTL.
        - Note: this uses **historical median times** per corridor, not the ML model predictions.
          The ETA & Route Predictor page uses the trained models for individual trip decisions.
        """
    )

    # ── Breakeven line chart ───────────────────────────────────────────────────
    st.markdown("### Break-even: FTL Premium vs Time Saved")
    st.caption("For a given SLA penalty rate, how much time does FTL need to save to justify its premium?")

    breakeven_cpm = st.slider("SLA Penalty (₹/min) for break-even chart", 0.5, 10.0, 2.0, 0.5)
    dist_range    = np.linspace(5, 200, 100)
    for fpp_val in [4, 8, 12]:
        breakeven_mins = (fpp_val * dist_range) / breakeven_cpm
        fig_be = px.line() if "fig_be" not in dir() else fig_be
    
    fig_be2 = go.Figure()
    for fpp_val in [4, 8, 12]:
        breakeven_mins = (fpp_val * dist_range) / breakeven_cpm
        fig_be2.add_trace(go.Scatter(
            x=dist_range, y=breakeven_mins,
            mode="lines", name=f"₹{fpp_val}/km FTL premium",
        ))
    fig_be2.update_layout(
        xaxis_title="Corridor Distance (km)",
        yaxis_title="Minutes FTL Must Save to Break Even",
        height=380, plot_bgcolor="white",
        legend=dict(title="FTL Premium"),
        margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig_be2, use_container_width=True)
    st.caption(
        f"At ₹{breakeven_cpm}/min SLA penalty: on a 50 km corridor with ₹8/km FTL premium, "
        f"FTL must save at least **{(8*50/breakeven_cpm):.0f} min** to be cost-justified."
    )