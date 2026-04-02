from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from clis.data.preprocessing import preprocess_from_csv, filter_consumer_spending


# Category labels aligned to the figures shown in the paper
CATEGORY_MAPPING = {
    "Groceries": ["COOP LOCAL", "SAINSBURY"],
    "Eating": ["DELIVEROO", "JUSTEAT", "HARVESTER", "CHIQUITO", "ASK ITALIAN", "BILL'S"],
    "Shopping": [
        "NEXT", "TOPSHOP", "DOROTHY PERKINS", "MATALAN", "REEBOK", "RIVER ISLAND", "NIKE",
        "ADIDAS", "NEW LOOK", "JD SPORTS", "MILLETS", "NORTH FACE", "UMBRO", "TK MAXX",
        "HOBBY LOBBY", "HOBBYCRAFT", "ETSY", "A LIFE ON CANVAS", "FIVE SENSES ART",
        "CRAFTASTIC", "BRILLIANT BRUSHES", "CASS ART", "STITCH BY STITCH",
        "A YARN STORY", "SPORTS DIRECT", "AMAZON", "MOUNTAIN WAREHOUSE"
    ],
    "Entertainment": ["NETFLIX", "BLIZZARD", "XBOX", "MOJANG STUDIOS", "SQUAREONIX", "DISNEY", "GAME", "GAMESTATION", "CEX"],
    "Fitness": ["PUREGYM", "GRAND UNION BJJ", "FOOTBALLPITCH"],
    "Finance": ["HALIFAX", "LBG", "PREMIER FINANCE"],
    "Personal Care": ["BOOTS", "LLOYDS PHARMACY", "REMEDY PLUS CARE"],
    "Coffee Shops and Bars": [
        "COFFEE REPUBLIC", "FULL OF BEANS", "AMT COFFEE", "COFFEE #1", "COSTA COFFEE",
        "STARBUCKS", "THE ROYAL OAK", "RED LION", "KINGS ARMS", "ROSE & CROWN",
        "THE CROWN", "WHITE HART"
    ],
    "Bookstore": ["BLACKWELL'S", "WATERSTONES", "FOYLES", "DAUNT BOOKS", "THE WORKS"],
}

CATEGORY_ORDER = [
    "Shopping",
    "Groceries",
    "Entertainment",
    "Eating",
    "Fitness",
    "Bookstore",
    "Finance",
    "Coffee Shops and Bars",
    "Personal Care",
]


def ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def apply_plot_style() -> None:
    plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.figsize": (12, 7),
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })


def categorize_transaction(merchant: object) -> str:
    merchant = str(merchant).upper().strip()
    for category, keywords in CATEGORY_MAPPING.items():
        if any(keyword in merchant for keyword in keywords):
            return category
    return "General"


def add_spending_category(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["Spending Category"] = tmp["Third Party Name"].apply(categorize_transaction)
    return tmp


def _add_bar_labels(ax, offset_ratio: float = 0.01) -> None:
    heights = [bar.get_height() for bar in ax.patches]
    if not heights:
        return
    max_height = max(heights)
    offset = max_height * offset_ratio

    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )


def plot_total_spending_by_category(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = ensure_output_dir(output_dir)

    tmp = add_spending_category(df)
    category_spending = (
        tmp.groupby("Spending Category")["Amount"]
        .sum()
        .reindex(CATEGORY_ORDER + ["General"])
        .dropna()
    )

    plt.figure(figsize=(12, 8))
    ax = category_spending.plot(kind="bar", color="orange", edgecolor="black")
    plt.title("Total Spending by Category")
    plt.xlabel("Spending Category")
    plt.ylabel("Total Amount Spent (£)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    _add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(output_dir / "spending_by_category.png", dpi=300)
    plt.close()


def plot_transaction_frequency_by_category(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = ensure_output_dir(output_dir)

    tmp = add_spending_category(df)
    category_counts = (
        tmp["Spending Category"]
        .value_counts()
        .reindex(CATEGORY_ORDER + ["General"])
        .dropna()
    )

    plt.figure(figsize=(12, 8))
    ax = category_counts.plot(kind="bar", color="orange", edgecolor="black")
    plt.title("Transaction Frequency by Category")
    plt.xlabel("Spending Category")
    plt.ylabel("Number of Transactions")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    _add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(output_dir / "transaction_frequency_by_category.png", dpi=300)
    plt.close()


def plot_filtered_transaction_amounts(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = ensure_output_dir(output_dir)

    plt.figure(figsize=(12, 8))
    plt.hist(df["Amount"], bins=50, edgecolor="black", log=True)
    plt.xlabel("Transaction Amount (£)")
    plt.ylabel("Frequency (Log Scale)")
    plt.title("Distribution of Transaction Amounts (Filtered for Numeric Third Party Account No)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "filtered_transaction_amounts.png", dpi=300)
    plt.close()


def generate_paper_figures(input_csv: str | Path, output_dir: str | Path) -> pd.DataFrame:
    apply_plot_style()

    cleaned = preprocess_from_csv(input_csv)
    spending = filter_consumer_spending(cleaned)

    plot_total_spending_by_category(spending, output_dir)
    plot_transaction_frequency_by_category(spending, output_dir)
    plot_filtered_transaction_amounts(spending, output_dir)

    return spending


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    input_csv = project_root / "data" / "raw" / "simulated_fake_transactions_dataset_2.csv"
    output_dir = project_root / "outputs" / "figures" / "eda"

    spending_df = generate_paper_figures(input_csv, output_dir)
    print(spending_df.head())
    print(spending_df.shape)
