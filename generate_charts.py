import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("static/charts", exist_ok=True)

data = pd.read_csv("dataset/indian_roads_dataset.csv")


# ---------------- FUNCTION ----------------
def plot_chart(column, title, filename, bar_color, text_color, title_color):
    counts = data[column].value_counts()
    total = len(data)

    plt.figure(figsize=(7,5))

    bars = plt.bar(
        counts.index,
        counts.values,
        color=bar_color,
        width=0.65
    )

    # 🔥 INSIDE BAR TEXT (IMPROVED VISIBILITY)
    for bar in bars:
        height = bar.get_height()
        percent = (height / total) * 100

        plt.text(
            bar.get_x() + bar.get_width()/2,
            height - (height * 0.10),   # inside bar safely
            f'{height}\n({percent:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=9,
            color=text_color,
            fontweight='bold'
        )

    # 🔥 BOLD COLORED TITLE
    plt.title(
        title,
        fontsize=16,
        fontweight='bold',
        color=title_color
    )

    plt.ylabel("Count", fontsize=11, fontweight='bold', color="#444")

    plt.xticks(rotation=15, fontsize=10)
    plt.tight_layout()

    plt.savefig(f"static/charts/{filename}")
    plt.close()


# ---------------- COOL MEDIUM COLORS ----------------

plot_chart(
    "accident_severity",
    "Accident Severity Distribution",
    "severity_chart.png",
    "#4C6FA6",   # medium blue
    "#FFFFFF",   # white text
    "#2C3E50"    # dark title
)

plot_chart(
    "weather",
    "Weather Conditions Analysis",
    "weather_chart.png",
    "#C97C5D",   # warm brown-orange
    "#FFFFFF",
    "#2C3E50"
)

plot_chart(
    "road_type",
    "Road Type Analysis",
    "roadtype_chart.png",
    "#5C9E6F",   # medium green
    "#FFFFFF",
    "#2C3E50"
)

plot_chart(
    "traffic_density",
    "Traffic Density Analysis",
    "traffic_chart.png",
    "#B05C5C",   # soft red
    "#FFFFFF",
    "#2C3E50"
)

print("Charts Updated Successfully ✔")