"""Flask App for Activities 2, 3, and 4."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from activity2_analysis import (
    create_pca_chart,
    create_trend_chart,
    find_similar_months,
    get_analysis,
)
from graph_analysis import GraphInputError, analyze_payments, payment_options
from rag_analysis import rag_query


BASE_DIR = Path(__file__).resolve().parent


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        DATA_PATH=str(
            BASE_DIR / "jopacc_system_month_november_2022_to_june_2026.csv"
        ),
    )
    if test_config:
        app.config.update(test_config)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/cluster", methods=["GET", "POST"])
    def cluster():
        analysis = get_analysis(str(Path(app.config["DATA_PATH"]).resolve()))
        data = analysis["data"]
        systems = sorted(data["system"].unique())
        years = sorted(data["report_year"].unique())
        patterns = analysis["profiles"].sort_values("cluster")["pattern"].tolist()

        filters = {
            "system": request.form.get("system", "all"),
            "year": request.form.get("year", "all"),
            "month": request.form.get("month", "all"),
            "start_month": request.form.get(
                "start_month", data["report_month"].min().strftime("%Y-%m")
            ),
            "end_month": request.form.get(
                "end_month", data["report_month"].max().strftime("%Y-%m")
            ),
            "pattern": request.form.get("pattern", "all"),
            "state": request.form.get("state", "all"),
        }

        selected = data.copy()
        errors = []
        if filters["system"] != "all":
            if filters["system"] not in systems:
                errors.append("Please choose a payment system that is in the data.")
            else:
                selected = selected[selected["system"].eq(filters["system"])]

        if filters["year"] != "all":
            try:
                chosen_year = int(filters["year"])
            except (TypeError, ValueError):
                chosen_year = -1
            if chosen_year not in years:
                errors.append("Please choose a year that is in the data.")
            else:
                selected = selected[selected["report_year"].eq(chosen_year)]

        if filters["month"] != "all":
            try:
                chosen_month = int(filters["month"])
            except (TypeError, ValueError):
                chosen_month = -1
            if chosen_month not in range(1, 13):
                errors.append("Please choose a month between 1 and 12.")
            else:
                selected = selected[
                    selected["report_month"].dt.month.eq(chosen_month)
                ]

        start = None
        end = None
        try:
            start = pd.to_datetime(filters["start_month"], format="%Y-%m")
            end = pd.to_datetime(filters["end_month"], format="%Y-%m")
        except (TypeError, ValueError):
            errors.append("The start and end months have to be written as YYYY-MM.")
        if start is not None and end is not None:
            if start > end:
                errors.append("The start month cannot come after the end month.")
            else:
                selected = selected[
                    selected["report_month"].between(start, end, inclusive="both")
                ]

        if filters["pattern"] != "all":
            if filters["pattern"] not in patterns:
                errors.append("Please choose one of the patterns we discovered.")
            else:
                selected = selected[selected["pattern"].eq(filters["pattern"])]

        if filters["state"] != "all":
            if filters["state"] not in {"normal", "unusual"}:
                errors.append("Please choose either the normal or the unusual state.")
            else:
                selected = selected[selected["state"].eq(filters["state"])]

        selected = selected.sort_values(
            ["report_month", "system"], ascending=[False, True]
        )
        detail = selected.iloc[0].to_dict() if len(selected) == 1 and not errors else None
        similar = []
        if detail is not None:
            similar = find_similar_months(
                analysis, selected.index[0], count=5
            ).to_dict("records")

        results = selected.to_dict("records") if not errors else []
        for row in results:
            row["report_month_label"] = row["report_month"].strftime("%B %Y")
        if detail is not None:
            detail["report_month_label"] = detail["report_month"].strftime("%B %Y")
        for row in similar:
            row["report_month_label"] = row["report_month"].strftime("%B %Y")

        trend_chart = create_trend_chart(selected) if not errors else None
        highlighted = selected.index.tolist() if request.method == "POST" else []
        pca_chart = create_pca_chart(data, highlighted)
        validation = analysis["validation"]

        return render_template(
            "cluster.html",
            systems=systems,
            years=years,
            months=range(1, 13),
            patterns=patterns,
            filters=filters,
            results=results,
            detail=detail,
            similar=similar,
            errors=errors,
            trend_chart=trend_chart,
            pca_chart=pca_chart,
            profiles=analysis["profiles"].sort_values("cluster").to_dict("records"),
            model_metrics=analysis["comparison"].to_dict("records"),
            explained_variance=float(
                analysis["pca_summary"]["explained_variance_ratio"].sum() * 100
            ),
            dataset_summary={
                "rows": validation["rows"],
                "systems": validation["systems"],
                "first_month": validation["first_month"].strftime("%Y-%m"),
                "last_month": validation["last_month"].strftime("%Y-%m"),
            },
        )

    @app.route("/text", methods=["GET", "POST"])
    def text():
        question = ""
        answer = ""
        docs = []
        error = ""
        if request.method == "POST":
            question = request.form.get("question", "")
            try:
                answer, docs = rag_query(
                    question,
                    top_k=5,
                )
            except (FileNotFoundError, ValueError) as exc:
                error = str(exc)
        return render_template(
            "text.html",
            question=question,
            answer=answer,
            docs=docs,
            error=error,
        )

    @app.route("/graph", methods=["GET", "POST"])
    def graph():
        options = payment_options(app.config["DATA_PATH"])
        filters = {
            "system": request.form.get("system", "ACH"),
            "start_month": request.form.get("start_month", "2023-01"),
            "end_month": request.form.get("end_month", "2023-12"),
            "indicator": request.form.get("indicator", "transactions_total"),
        }
        result = None
        error = ""
        if request.method == "POST":
            try:
                result = analyze_payments(
                    payment_system=filters["system"],
                    start_month=filters["start_month"],
                    end_month=filters["end_month"],
                    indicator=filters["indicator"],
                    data_path=app.config["DATA_PATH"],
                )
            except GraphInputError as exc:
                error = str(exc)
        return render_template(
            "graph.html",
            systems=options["systems"],
            coverage=options["coverage"],
            indicators=options["indicators"],
            filters=filters,
            result=result,
            error=error,
        )

    @app.template_filter("number")
    def format_number(value):
        return f"{float(value):,.0f}"

    @app.template_filter("decimal")
    def format_decimal(value):
        return f"{float(value):,.3f}"

    @app.template_filter("score")
    def format_score(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "N/A"
        return f"{float(value):.3f}"

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=False)
