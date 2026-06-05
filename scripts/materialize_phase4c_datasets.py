from pathlib import Path
import shutil

import pandas as pd


def copy_episode_group(
    csv_path: Path, source_root: Path, destination_root: Path, manifest_output_path: Path
) -> tuple[int, list[int]]:
    frame = pd.read_csv(csv_path)
    copied = 0
    missing_ids: list[int] = []
    copied_ids: list[int] = []

    destination_root.mkdir(parents=True, exist_ok=True)

    for _, row in frame.iterrows():
        episode_id = int(row["episode_id"])
        source_dir = source_root / f"episode_{episode_id}"
        destination_dir = destination_root / source_dir.name

        if not source_dir.exists():
            missing_ids.append(episode_id)
            continue

        if destination_dir.exists():
            shutil.rmtree(destination_dir)

        shutil.copytree(source_dir, destination_dir)
        copied_ids.append(episode_id)
        copied += 1

    manifest_frame = frame[frame["episode_id"].astype(int).isin(copied_ids)].copy()
    manifest_frame["episode_id"] = manifest_frame["episode_id"].astype(int)
    if "episode_steps" in manifest_frame.columns:
        manifest_frame["episode_steps"] = manifest_frame["episode_steps"].astype(int)
    manifest_frame.to_csv(manifest_output_path, index=False)

    return copied, missing_ids


def main() -> None:
    source_root = Path("testdata/1/episodes")
    output_root = Path("testdata/1/selected_phase4c_reward_distance")
    top_csv = Path("outputs/dataset_selection/phase4c_reward_distance_top_combined.csv")
    bottom_csv = Path("outputs/dataset_selection/phase4c_reward_distance_bottom_combined.csv")

    top_output = output_root / "top_combined"
    bottom_output = output_root / "bottom_combined"
    top_manifest = output_root / "top_combined_manifest.csv"
    bottom_manifest = output_root / "bottom_combined_manifest.csv"

    top_copied, top_missing_ids = copy_episode_group(top_csv, source_root, top_output, top_manifest)
    bottom_copied, bottom_missing_ids = copy_episode_group(
        bottom_csv, source_root, bottom_output, bottom_manifest
    )

    summary_path = output_root / "copy_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"source_root={source_root}",
                f"top_csv={top_csv}",
                f"bottom_csv={bottom_csv}",
                f"top_output={top_output}",
                f"bottom_output={bottom_output}",
                f"top_manifest={top_manifest}",
                f"bottom_manifest={bottom_manifest}",
                f"top_copied={top_copied}",
                f"top_missing={len(top_missing_ids)}",
                f"top_missing_ids={','.join(str(episode_id) for episode_id in top_missing_ids) or 'none'}",
                f"bottom_copied={bottom_copied}",
                f"bottom_missing={len(bottom_missing_ids)}",
                f"bottom_missing_ids={','.join(str(episode_id) for episode_id in bottom_missing_ids) or 'none'}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Copied {top_copied} top episodes to {top_output}")
    print(f"Copied {bottom_copied} bottom episodes to {bottom_output}")
    print(f"Wrote {top_manifest}")
    print(f"Wrote {bottom_manifest}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()