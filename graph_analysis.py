"""Activity 4 - directed, weighted JoPACC payment-activity graphs.

The notebook's ACH January-December 2023 method is generalised here for the
Flask selectors.  Months are ranked by the selected indicator, divided into
four activity states, restored to time order, and linked by consecutive
directed transitions whose weights count repetition.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "jopacc_system_month_november_2022_to_june_2026.csv"

ACTIVITY_STATES = ["Low", "Medium", "High", "Very High"]
STATE_COLOURS = {
    "Low": "#bfdbfe",
    "Medium": "#93c5fd",
    "High": "#3b82f6",
    "Very High": "#1d4ed8",
}
INDICATORS = {
    "transactions_total": "Total transactions",
    "transaction_value_jod": "Transaction value (JOD)",
    "average_transaction_value": "Average transaction value (JOD)",
}


class GraphInputError(ValueError):
    """Raised when navigator selections cannot produce a meaningful graph."""


def load_payment_data(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load common, non-structural JoPACC indicators for graph analysis."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"JoPACC dataset was not found: {data_path}")
    data = pd.read_csv(data_path)
    required = {"report_month", "system", "transactions_total", "transaction_value_jod"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    data["report_month"] = pd.to_datetime(
        data["report_month"], format="%Y-%m", errors="coerce"
    )
    for column in ["transactions_total", "transaction_value_jod"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[["report_month", "system", "transactions_total", "transaction_value_jod"]].isna().any().any():
        raise ValueError("The graph source data contains missing or invalid values.")
    if data.duplicated(["system", "report_month"]).any():
        raise ValueError("The graph source data contains duplicate system-month rows.")
    if (data["transactions_total"] <= 0).any():
        raise ValueError("Average values require positive transaction totals.")

    data["average_transaction_value"] = (
        data["transaction_value_jod"] / data["transactions_total"]
    )
    return data.sort_values(["system", "report_month"]).reset_index(drop=True)


def payment_options(data_path: str | Path = DEFAULT_DATA_PATH) -> dict:
    """Return valid systems, coverage, and indicators for the Flask form."""
    data = load_payment_data(data_path)
    coverage = {}
    for system, group in data.groupby("system", sort=True):
        coverage[system] = {
            "start": group["report_month"].min().strftime("%Y-%m"),
            "end": group["report_month"].max().strftime("%Y-%m"),
        }
    return {
        "systems": sorted(data["system"].unique()),
        "coverage": coverage,
        "indicators": INDICATORS.copy(),
    }


def _parse_month(value: str, field_name: str) -> pd.Timestamp:
    try:
        parsed = pd.to_datetime(value, format="%Y-%m", errors="raise")
    except (TypeError, ValueError) as exc:
        raise GraphInputError(f"The {field_name.lower()} has to be written as YYYY-MM.") from exc
    return parsed


def _assign_activity_states(monthly: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """Rank values and split them into the notebook's four ordered groups."""
    ranked = monthly.sort_values([indicator, "report_month"]).reset_index(drop=True)
    ranked["activity"] = ""
    for state, positions in zip(
        ACTIVITY_STATES, np.array_split(np.arange(len(ranked)), len(ACTIVITY_STATES))
    ):
        ranked.loc[positions, "activity"] = state
    return ranked.sort_values("report_month").reset_index(drop=True)


def _build_weighted_graph(monthly: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(ACTIVITY_STATES)
    sequence = monthly["activity"].tolist()
    for source, target in zip(sequence, sequence[1:]):
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += 1
        else:
            graph.add_edge(source, target, weight=1)
    return graph


def _graph_measures(graph: nx.DiGraph) -> list[dict]:
    in_centrality = nx.in_degree_centrality(graph)
    out_centrality = nx.out_degree_centrality(graph)
    try:
        pagerank = nx.pagerank(graph, alpha=1.0, weight="weight", max_iter=500)
    except nx.PowerIterationFailedConvergence:
        pagerank = nx.pagerank(graph, alpha=0.85, weight="weight", max_iter=500)

    # Reversing the graph measures how easily a state can reach later states,
    # matching the outgoing-closeness calculation used in Activity_4.ipynb.
    outgoing_closeness = nx.closeness_centrality(graph.reverse())
    betweenness = nx.betweenness_centrality(graph, weight=None)

    rows = []
    for node in ACTIVITY_STATES:
        rows.append(
            {
                "node": node,
                "in_degree": int(graph.in_degree(node)),
                "out_degree": int(graph.out_degree(node)),
                "incoming_frequency": int(graph.in_degree(node, weight="weight")),
                "outgoing_frequency": int(graph.out_degree(node, weight="weight")),
                "in_degree_centrality": float(in_centrality[node]),
                "out_degree_centrality": float(out_centrality[node]),
                "pagerank": float(pagerank[node]),
                "closeness": float(outgoing_closeness[node]),
                "betweenness": float(betweenness[node]),
            }
        )
    return rows


def _draw_graph(graph: nx.DiGraph, title: str) -> str:
    figure, axis = plt.subplots(figsize=(8.5, 6.2))
    positions = nx.circular_layout(graph)
    weights = [graph[source][target]["weight"] for source, target in graph.edges()]

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=3100,
        node_color=[STATE_COLOURS[node] for node in graph.nodes()],
        edgecolors="#1e3a5f",
        linewidths=1.2,
        ax=axis,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=10,
        font_weight="bold",
        font_color="#0f172a",
        ax=axis,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        arrows=True,
        arrowsize=20,
        width=[1.2 + weight * 0.8 for weight in weights],
        edge_color="#64748b",
        connectionstyle="arc3,rad=0.14",
        min_source_margin=22,
        min_target_margin=22,
        ax=axis,
    )
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=9,
        label_pos=0.52,
        connectionstyle="arc3,rad=0.14",
        ax=axis,
    )
    axis.set_title(title, fontsize=13, pad=16)
    axis.axis("off")
    figure.tight_layout()

    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=135, bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _longest_state_run(states: list[str]) -> tuple[str, int]:
    best_state = states[0]
    best_length = 1
    current_state = states[0]
    current_length = 1
    for state in states[1:]:
        if state == current_state:
            current_length += 1
        else:
            current_state = state
            current_length = 1
        if current_length > best_length:
            best_state = current_state
            best_length = current_length
    return best_state, best_length


def analyze_payments(
    payment_system: str,
    start_month: str,
    end_month: str,
    indicator: str,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> dict:
    """Run the complete Activity 4 navigator analysis for user selections."""
    data = load_payment_data(data_path)
    valid_systems = set(data["system"])
    if payment_system not in valid_systems:
        raise GraphInputError("Please choose a payment system that is in the data.")
    if indicator not in INDICATORS:
        raise GraphInputError("Please choose one of the payment indicators we have.")

    start = _parse_month(start_month, "Start month")
    end = _parse_month(end_month, "End month")
    if start > end:
        raise GraphInputError("The start month cannot come after the end month.")

    selected = data[
        data["system"].eq(payment_system)
        & data["report_month"].between(start, end, inclusive="both")
    ][["report_month", indicator]].copy()

    expected_months = pd.date_range(start, end, freq="MS")
    actual_months = pd.DatetimeIndex(selected["report_month"])
    missing_months = expected_months.difference(actual_months)
    if len(missing_months):
        missing_labels = ", ".join(month.strftime("%Y-%m") for month in missing_months)
        raise GraphInputError(
            f"The payment system we chose has no record for: {missing_labels}. "
            "Please choose a period that is inside the months it actually covers."
        )
    if len(selected) < 4:
        raise GraphInputError(
            "Please choose at least four months in a row, because we need that many "
            "to create all four activity states."
        )

    monthly = _assign_activity_states(selected, indicator)
    graph = _build_weighted_graph(monthly)
    measures = _graph_measures(graph)
    edge_rows = [
        {"source": source, "target": target, "weight": values["weight"]}
        for source, target, values in graph.edges(data=True)
    ]
    edge_rows.sort(key=lambda row: (-row["weight"], row["source"], row["target"]))

    state_counts = monthly["activity"].value_counts()
    maximum_state_count = int(state_counts.max())
    leading_states = [
        state
        for state in ACTIVITY_STATES
        if int(state_counts.get(state, 0)) == maximum_state_count
    ]
    if len(leading_states) == len(ACTIVITY_STATES):
        most_frequent_state = f"All states ({maximum_state_count} months each)"
    elif len(leading_states) > 1:
        most_frequent_state = (
            ", ".join(leading_states) + f" (tie at {maximum_state_count} months)"
        )
    else:
        most_frequent_state = f"{leading_states[0]} ({maximum_state_count} months)"
    most_common_transition = edge_rows[0]
    changes = int((monthly["activity"] != monthly["activity"].shift()).sum() - 1)
    transition_count = len(monthly) - 1
    longest_state, longest_run = _longest_state_run(monthly["activity"].tolist())
    leading_pagerank = max(measures, key=lambda row: row["pagerank"])

    monthly["month_label"] = monthly["report_month"].dt.strftime("%B %Y")
    monthly["indicator_value"] = monthly[indicator]
    monthly_records = monthly[
        ["report_month", "month_label", "indicator_value", "activity"]
    ].to_dict("records")

    group_sizes = {
        state: int((monthly["activity"] == state).sum()) for state in ACTIVITY_STATES
    }
    return {
        "system": payment_system,
        "start_month": start.strftime("%Y-%m"),
        "end_month": end.strftime("%Y-%m"),
        "indicator": indicator,
        "indicator_label": INDICATORS[indicator],
        "monthly_data": monthly_records,
        "edges": edge_rows,
        "measures": measures,
        "graph_image": _draw_graph(
            graph,
            f"{payment_system}: directed weighted graph of the activity transitions",
        ),
        "group_sizes": group_sizes,
        "summary": {
            "most_frequent_state": most_frequent_state,
            "most_common_transition": (
                f"{most_common_transition['source']} to "
                f"{most_common_transition['target']} "
                f"({most_common_transition['weight']} time(s))"
            ),
            "state_changes": changes,
            "transition_count": transition_count,
            "change_rate": changes / transition_count * 100,
            "longest_run": f"{longest_state} for {longest_run} month(s)",
            "leading_pagerank": (
                f"{leading_pagerank['node']} ({leading_pagerank['pagerank']:.3f})"
            ),
        },
    }


if __name__ == "__main__":
    result = analyze_payments(
        "ACH", "2023-01", "2023-12", "transactions_total"
    )
    print(result["summary"])
    print(pd.DataFrame(result["edges"]).to_string(index=False))
