import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np


def load_and_visualize_excel(file_path):
    try:
        # Load the Excel file
        df = pd.read_csv(file_path)
        # Check if any data was loaded
        if df.empty:
            print("The Excel file is empty.")
            return

        # Display the first few rows in the console
        print("Excel Data Preview:")
        print(df.head())

        # Check available column names
        print(df.columns)

        num_accounts = df["Account No"].nunique()
        print(f"Total unique accounts: {num_accounts}")

        # Remove rows where 'Third Party Account No' is numeric
        df = df[df["Third Party Account No"].isna()]
        df = df[df["Amount"] < 0]
        df["Amount"] = df["Amount"].abs()

        num_accounts = df["Account No"].nunique()
        print(f"Total unique accounts: {num_accounts}")
        # Simple visualization: Plotting the first two numeric columns, if available
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:

            # Plot histogram with only negative transaction amounts
            plt.figure(figsize=(12, 12))
            plt.hist(df["Amount"], bins=50, edgecolor='black', log=True)
            plt.xlabel("Transaction Amount (£)")
            plt.ylabel("Frequency (Log Scale)")
            plt.title("Distribution of Transaction Amounts (Filtered for Numeric Third Party Account No)")
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            # Save and show the plot
            plt.savefig("filtered_transaction_amounts.png", dpi=300, bbox_inches='tight')
            plt.show()

            # Define threshold for small payments
            small_payment_threshold = 250

            # Count transactions below and above the threshold
            small_payments = df[df["Amount"] < 250].shape[0]
            large_payments = df[df["Amount"] >= 250].shape[0]

            # Prepare data for pie chart
            sizes = [small_payments, large_payments]
            labels = ["Small Payments (< £250)", "Large Payments (≥ £250)"]
            colors = ["lightblue", "orange"]

            # Plot pie chart
            plt.figure(figsize=(7, 7))
            plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140,
                    wedgeprops={'edgecolor': 'black'})
            plt.title("Share of Small Payments in Transaction Amounts")
            plt.savefig("Share of Small Payments in Transaction Amounts.png", dpi=300,
                        bbox_inches='tight')  # Save as PNG
            plt.show()

            # Define bins for transaction amounts (each £250)
            bins = list(range(0, int(df["Amount"].max()) + 250, 250))

            # Generate labels
            labels = [f"£{bins[i]} - £{bins[i + 1]}" for i in range(len(bins) - 1)]

            # Categorize transactions into bins
            df["amount_category"] = pd.cut(df["Amount"], bins=bins, labels=labels, right=False)

            # Count transactions in each category
            category_counts = df["amount_category"].value_counts().sort_index()

            # Calculate percentages for legend
            percentages = category_counts / category_counts.sum() * 100
            legend_labels = [f"{label}: {percent:.1f}%" for label, percent in zip(category_counts.index, percentages)]

            # # Plot pie chart
            plt.figure(figsize=(12, 8))
            wedges, texts = plt.pie(category_counts, startangle=140, wedgeprops={'edgecolor': 'black'})

            # Add legend with percentages
            plt.legend(wedges, legend_labels, title="Transaction Ranges", loc="center left", bbox_to_anchor=(1, 0.5))

            # Add title
            plt.title("Transaction Amount Distribution by £250 Intervals")

            plt.savefig("Transaction Amount Distribution by £250 Intervals.png", dpi=300,
                        bbox_inches='tight')  # Save as PNG

            # Show plot
            plt.show()

            ################################################################

            # Plot: Top 10 Most Frequent Transaction Recipients ################################################################
            top_merchants = df["Third Party Name"].value_counts().head(10)

            plt.figure(figsize=(14, 20))
            top_merchants.plot(kind="bar", color='blue', edgecolor='black')
            plt.xlabel("Merchant")
            plt.ylabel("Transaction Count")
            plt.title("Top 10 Most Frequent Transaction Recipients")
            plt.xticks(rotation=45, ha="right")
            plt.savefig("top_10_frequent_recipients.png", dpi=300, bbox_inches='tight')  # Save as PNG
            plt.show()

            # Extract unique transaction recipients
            unique_recipients = df["Third Party Name"].unique()

            # Function to check if a string is purely numeric
            def is_numeric(value):
                return bool(re.fullmatch(r"\d+", str(value).strip()))  # Strictly match full numeric strings

            # Filter out numeric values
            filtered_recipients = [recipient for recipient in unique_recipients if not is_numeric(recipient)]

            # Display the first 20 unique filtered recipients (for preview)
            print(filtered_recipients[:20])  # Adjust the number as needed

            # Save to a text file for further analysis (optional)
            with open("filtered_unique_recipients.txt", "w") as f:
                for recipient in filtered_recipients:
                    f.write(f"{recipient}\n")

            print(f"Total unique transaction recipients (excluding numbers): {len(filtered_recipients)}")

            # Define category mappings
            category_mapping = {
                "Groceries": ["COOP LOCAL", "SAINSBURY"],
                "Eating": ["DELIVEROO", "JUSTEAT", "HARVESTER", "CHIQUITO", "ASK ITALIAN", "BILL'S"],
                "Shopping": ["NEXT", "TOPSHOP", "DOROTHY PERKINS", "MATALAN", "REEBOK", "RIVER ISLAND", "NIKE",
                             "ADIDAS", "NEW LOOK"],
                "Entertainment": ["NETFLIX", "BLIZZARD", "XBOX", "MOJANG STUDIOS", "SQUAREONIX", "DISNEY", "GAME",
                                  "GAMESTATION", "CEX"],
                "Fitness": ["PUREGYM", "GRAND UNION BJJ", "FOOTBALLPITCH"],
                "Finances": ["HALIFAX", "LBG", "PREMIER FINANCE"],
                "Personal Care": ["BOOTS", "LLOYDS PHARMACY", "REMEDY PLUS CARE"],
                "Coffee Shops": ["COFFEE REPUBLIC", "FULL OF BEANS", "AMT COFFEE", "COFFEE #1", "COSTA COFFEE",
                                 "STARBUCKS", "THE ROYAL OAK"],
                "Bookstore": ["BLACKWELL'S", "WATERSTONES", "FOYLES", "DAUNT BOOKS", "THE WORKS"]
            }

            # Function to assign categories based on keywords in the merchant name
            def categorize_transaction(merchant):
                merchant = str(merchant).upper()  # Convert to uppercase for matching
                for category, keywords in category_mapping.items():
                    if any(keyword in merchant for keyword in keywords):
                        return category
                # return "Unknown"  # Default category
                return None

            # Apply categorization
            df["category"] = df["Third Party Name"].apply(categorize_transaction)

            # Sum transaction amounts by category
            category_spending = df.groupby("category")["Amount"].sum().sort_values(ascending=False)

            # Plot bar chart
            plt.figure(figsize=(14, 14))
            bars = category_spending.plot(kind="bar", color="orange", edgecolor="black")

            # category_spending.plot(kind="bar", color="skyblue", edgecolor="black",log=False)
            plt.xlabel("Spending Category", fontsize=12)
            plt.ylabel("Total Amount Spent (£)", fontsize=12)
            plt.title("Total Spending by Category", fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=12)
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            # Add number labels on bars
            for bar in bars.patches:
                plt.text(bar.get_x() + bar.get_width() / 2,  # X position (center of bar)
                         bar.get_height() + 900,  # Y position (above the bar)
                         f"{int(bar.get_height()):,}",  # Format number with commas
                         ha="center", va="center", fontsize=12, fontweight="bold")

            # Save chart as an image (optional)
            plt.savefig("spending_by_category.png", dpi=300, bbox_inches='tight')
            # Show the plot
            plt.show()

            # Count the number of transactions per category
            category_counts = df["category"].value_counts().sort_values(ascending=False)

            # Plot bar chart
            plt.figure(figsize=(14, 14))
            bars = category_counts.plot(kind="bar", color="orange", edgecolor="black")

            # category_counts.plot(kind="bar", color="orange", edgecolor="black")
            plt.xlabel("Spending Category")
            plt.ylabel("Number of Transactions")
            plt.title("Transaction Frequency by Category")
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            # Add number labels on bars
            for bar in bars.patches:
                plt.text(bar.get_x() + bar.get_width() / 2,  # X position (center of bar)
                         bar.get_height() + 600,  # Y position (above the bar)
                         f"{int(bar.get_height()):,}",  # Format number with commas
                         ha="center", va="center", fontsize=10, fontweight="bold")

            # Save chart as an image (optional)
            plt.savefig("transaction_frequency_by_category.png", dpi=300, bbox_inches='tight')

            # Show the plot
            plt.show()
            ################################################################

            # Plot: Transaction Volume Over Time################################################################
            # transactions_over_time = df.groupby("Date").size()
            #
            # plt.figure(figsize=(12, 5))
            # transactions_over_time.plot(color='green', marker='o', linestyle='-')
            # plt.xlabel("Date")
            # plt.ylabel("Number of Transactions")
            # plt.title("Transaction Volume Over Time")
            # plt.xticks(rotation=45)
            # plt.savefig("transaction_volume_over_time.png", dpi=300, bbox_inches='tight')  # Save as PNG
            # plt.show()
            ################################################################

        else:
            print("Not enough numeric data available for visualization.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # load_and_visualize_excel("dataset/fake_transactional_dataset_1.csv")
    load_and_visualize_excel("dataset/simulated_fake_transactions_dataset_2.csv")
