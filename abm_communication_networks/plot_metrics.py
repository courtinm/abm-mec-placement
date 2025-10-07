import os
import csv
import matplotlib.pyplot as plt


def load_csv(file_path):
    steps, values = [], []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            steps.append(int(row['Step']))
            values.append(float(row['Value']))
    return steps, values


def plot_metric(steps, values, title, ylabel, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(steps, values, label=title, color='tab:blue')
    plt.xlabel("Simulation Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("metrics", exist_ok=True)
    plt.savefig(f"metrics/{filename}.png")
    plt.close()


def plot_learning_logs():
    log_folder = "logs"
    files = [f for f in os.listdir(log_folder) if f.endswith("_learning.csv")]

    if not files:
        print("No learning logs found.")
        return

    for file in files:
        steps, q_values, epsilons = [], [], []
        with open(os.path.join(log_folder, file)) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                steps.append(int(row["Step"]))
                q_values.append(float(row["Max_Q"]))
                epsilons.append(float(row["Epsilon"]))

        rn_id = file.split("_")[0].upper()
        plot_metric(steps, q_values, f"{rn_id} Q-Value Over Time", "Q-Value", f"{rn_id}_q")
        plot_metric(steps, epsilons, f"{rn_id} Epsilon Over Time", "Epsilon", f"{rn_id}_epsilon")


def plot_network_metrics():
    metrics_folder = "logs"
    tracked_metrics = [
        ("latency_avg.csv", "Average Latency per Step", "Latency (ms)", "latency_avg"),
        ("latency_max.csv", "Max Latency per Step", "Latency (ms)", "latency_max"),
        ("failed_connections.csv", "Failed Connections", "Count", "failed_connections"),
        ("nlos_connections.csv", "NLoS Connections", "Count", "nlos_connections"),
        ("handoffs.csv", "User Handoffs", "Count", "handoffs"),
        ("optimal_connections.csv", "Optimal Connection %", "Percentage", "optimal_pct")
    ]

    for filename, title, ylabel, save_as in tracked_metrics:
        path = os.path.join(metrics_folder, filename)
        if os.path.exists(path):
            steps, values = load_csv(path)
            plot_metric(steps, values, title, ylabel, save_as)
        else:
            print(f"Metric log not found: {filename}")


def main():
    print("Plotting relay learning metrics...")
    plot_learning_logs()
    print("Plotting network performance metrics...")
    plot_network_metrics()
    print("All plots saved to 'metrics/' folder.")


if __name__ == "__main__":
    main()
