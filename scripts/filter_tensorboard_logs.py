"""Create a compact TensorBoard snapshot; never modify source event files."""

import argparse
from pathlib import Path
import re


def filter_logs(source, output, pattern):
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError(f"Source log directory does not exist: {source}")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must not contain each other")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new snapshot directory: {output}")
    expression = re.compile(pattern)
    files = sorted(source.rglob("events.out.tfevents.*"))
    if not files:
        raise ValueError(f"No TensorBoard event files found under {source}")
    groups = {}
    for path in files:
        groups.setdefault(path.parent, []).append(path)
    stats = []
    for run, paths in groups.items():
        tags, kept_tags, count = set(), set(), 0
        destination = output / run.relative_to(source)
        writer = EventFileWriter(str(destination))
        try:
            for path in paths:
                for event in LegacyEventFileLoader(str(path)).Load():
                    if event.HasField("summary"):
                        tags.update(value.tag for value in event.summary.value)
                        values = [value for value in event.summary.value if expression.fullmatch(value.tag)]
                        if not values:
                            continue
                        filtered = Event()
                        filtered.CopyFrom(event)
                        filtered.summary.ClearField("value")
                        filtered.summary.value.extend(values)
                        kept_tags.update(value.tag for value in values)
                        count += len(values)
                        writer.add_event(filtered)
                    else:
                        # Preserve wall times/steps, restart markers and run metadata.
                        writer.add_event(event)
        finally:
            writer.close()
        stats.append({"run": str(run.relative_to(source)), "original_tags": len(tags),
                      "retained_tags": sorted(kept_tags), "retained_values": count})
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--filter-file", required=True, type=Path)
    args = parser.parse_args()
    stats = filter_logs(args.source, args.output, args.filter_file.read_text().strip())
    for row in stats:
        print(f"{row['run']}: {row['original_tags']} -> {len(row['retained_tags'])} tags, "
              f"{row['retained_values']} retained values")


if __name__ == "__main__":
    main()
