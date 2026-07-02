import time
import json
from collections import defaultdict

class SlicerProfiler:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.enabled = False
        # Timings: dict of key -> list of float durations (for overall summary statistics)
        self.timings = defaultdict(list)
        # Nested pass timings: dict of pass_name -> dict of category -> list of float durations
        self.pass_timings = defaultdict(lambda: defaultdict(list))
        # Start times for active trackers
        self.start_times = {}

    def enable(self):
        self.enabled = True

    def start(self, name: str):
        if not self.enabled:
            return
        self.start_times[name] = time.perf_counter()

    def stop(self, name: str):
        if not self.enabled:
            return
        if name in self.start_times:
            duration = time.perf_counter() - self.start_times[name]
            self.timings[name].append(duration)
            del self.start_times[name]
            return duration
        return 0.0

    def record_pass_time(self, pass_name: str, category: str, duration: float):
        if not self.enabled:
            return
        self.pass_timings[pass_name][category].append(duration)

    def save_report(self, filename_base: str):
        if not self.enabled:
            return

        # Prepare summary statistics
        summary = {}
        for name, durations in self.timings.items():
            summary[name] = {
                "total_seconds": sum(durations),
                "count": len(durations),
                "avg_seconds": sum(durations) / len(durations) if durations else 0,
                "min_seconds": min(durations) if durations else 0,
                "max_seconds": max(durations) if durations else 0
            }

        # Prepare pass timings statistics
        pass_data = {}
        for pass_name, cats in self.pass_timings.items():
            pass_data[pass_name] = {}
            for cat, durations in cats.items():
                pass_data[pass_name][cat] = {
                    "total_seconds": sum(durations),
                    "count": len(durations),
                    "avg_seconds": sum(durations) / len(durations) if durations else 0
                }

        report_data = {
            "summary": summary,
            "pass_breakdown": pass_data
        }

        # Save JSON
        json_filename = f"{filename_base}.json"
        with open(json_filename, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"✓ Performance diagnostics saved to {json_filename}")

        # Save Markdown Report
        md_filename = f"{filename_base}.md"
        with open(md_filename, "w") as f:
            f.write("# Slicer Performance Diagnostics Report\n\n")
            f.write("## Summary of Categories\n\n")
            f.write("| Category | Total Time (s) | Call Count | Avg Time (s) | Min Time (s) | Max Time (s) |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            for name, stats in sorted(summary.items(), key=lambda x: x[1]['total_seconds'], reverse=True):
                f.write(
                    f"| {name} | {stats['total_seconds']:.4f} | {stats['count']} | "
                    f"{stats['avg_seconds']:.4f} | {stats['min_seconds']:.4f} | {stats['max_seconds']:.4f} |\n"
                )
            f.write("\n")

            # Pass breakdown
            f.write("## Slicing Breakdown per Layer/Pass (Wall Layers & Infill)\n\n")
            if pass_data:
                # Get all unique categories across all passes
                categories = sorted(list(set(cat for p in pass_data.values() for cat in p.keys())))
                headers = ["Pass / Layer Name"] + [f"{cat} Total (s)" for cat in categories]
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                
                # Order: Wall 1, Wall 2, ... Infill, or alphabetically if named differently
                sorted_passes = sorted(pass_data.keys(), key=lambda x: (not x.startswith("Wall"), x))
                for pass_name in sorted_passes:
                    row = [pass_name]
                    for cat in categories:
                        row_data = pass_data[pass_name].get(cat, {})
                        val = row_data.get("total_seconds", 0.0)
                        row.append(f"{val:.4f}")
                    f.write("| " + " | ".join(row) + " |\n")
            else:
                f.write("No per-pass breakdown recorded.\n")
        print(f"✓ Performance diagnostics report saved to {md_filename}")
