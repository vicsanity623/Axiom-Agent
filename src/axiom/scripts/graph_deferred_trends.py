from __future__ import annotations

import json
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

# These imports will require 'pip install matplotlib'
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from axiom.config import DEFAULT_STATE_FILE

if TYPE_CHECKING:
    from pathlib import Path

# --- Configuration ---
STATE_FILE_PATH: Path = DEFAULT_STATE_FILE
REFRESH_INTERVAL_SECONDS: int = 200  # How often to update the graph (in seconds)


def read_high_priority_task_count(state_file: Path) -> int | None:
    """
    Safely reads the state file and returns the number of high-priority tasks.
    Returns None if the file cannot be read or parsed.
    """
    if not state_file.exists():
        print(f"State file not found at {state_file}, waiting...")
        return None

    try:
        with state_file.open("r", encoding="utf-8") as f:
            content = f.read()
            if not content:
                print("State file is empty, waiting...")
                return None
            data: dict[str, Any] = json.loads(content)

        high_priority_tasks = data.get("high_priority_tasks", [])
        return len(high_priority_tasks)

    except (OSError, json.JSONDecodeError) as e:
        print(f"Could not read or parse state file (is it being written?): {e}")
        return None


def main() -> None:
    """
    Main function to run the live-updating plot.
    """
    print("--- Axiom High-Priority Task Monitor ---")
    print(f"Monitoring state file: {STATE_FILE_PATH}")
    print(f"Updating every {REFRESH_INTERVAL_SECONDS} seconds. Press Ctrl+C to stop.")

    # --- Matplotlib Setup ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    timestamps: list[datetime] = []
    task_counts: list[int] = []

    max_datapoints = 1440  # 24 hours of 1-minute ticks

    try:
        while True:
            count = read_high_priority_task_count(STATE_FILE_PATH)

            if count is not None:
                now = datetime.now()
                timestamps.append(now)
                task_counts.append(count)

                if len(timestamps) > max_datapoints:
                    timestamps.pop(0)
                    task_counts.pop(0)

                numeric_dates = mdates.date2num(timestamps)

                # 2. Update the plot
                ax.clear()
                ax.plot(
                    numeric_dates,
                    task_counts,
                    marker="o",
                    linestyle="-",
                    color="b",
                    label="High-Priority Tasks",
                )

                if len(task_counts) >= 5:
                    moving_avg = [
                        sum(task_counts[i - 5 : i]) / 5
                        for i in range(5, len(task_counts) + 1)
                    ]
                    ax.plot(
                        numeric_dates[4:],
                        moving_avg,
                        linestyle="--",
                        color="r",
                        label="Trend (5-min SMA)",
                    )

                # 3. Formatting
                ax.set_title("High-Priority Task Trend Over Time (Live)")
                ax.set_xlabel("Time")
                ax.set_ylabel("Number of High-Priority Tasks")
                ax.grid(True, which="both", linestyle="--", linewidth=0.5)
                ax.legend()

                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                fig.autofmt_xdate(rotation=45)

                plt.tight_layout()
                plt.pause(0.1)

                print(f"[{now.strftime('%H:%M:%S')}] High-Priority Tasks: {count}")

            time.sleep(REFRESH_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nStopping monitor. Final graph will remain open.")
        plt.ioff()
        plt.show()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
