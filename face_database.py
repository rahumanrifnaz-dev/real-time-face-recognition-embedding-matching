"""Build, save, load, and query the local face-embedding database."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from face_encoder import get_embeddings_from_folder

DEFAULT_ENROLLMENT_DIR = Path("data/enrolled")
DEFAULT_DATABASE_PATH = Path("embeddings/face_embeddings.npz")
DEFAULT_THRESHOLD = 0.60
UNKNOWN_LABEL = "Unknown"


def euclidean_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return L2 distance; lower values indicate more similar embeddings."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError(f"Embedding shapes differ: {first.shape} and {second.shape}")
    return float(np.linalg.norm(first - second))


def match_embedding(
    query: np.ndarray,
    database: dict[str, list[np.ndarray]],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[str, float, str | None]:
    """Return final label, best distance, and closest enrolled identity.

    Multiple references are retained per identity. The best distance across all
    consented enrollment samples is used, then the threshold decides Known/Unknown.
    """
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")
    best_name: str | None = None
    best_distance = float("inf")
    for name, references in database.items():
        for reference in references:
            distance = euclidean_distance(query, reference)
            if distance < best_distance:
                best_name, best_distance = name, distance
    label = best_name if best_name is not None and best_distance <= threshold else UNKNOWN_LABEL
    return label, best_distance, best_name


def save_database(database: dict[str, list[np.ndarray]], path: str | Path = DEFAULT_DATABASE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    vectors: list[np.ndarray] = []
    for name, references in sorted(database.items()):
        for reference in references:
            names.append(name)
            vectors.append(np.asarray(reference, dtype=np.float32))
    matrix = np.stack(vectors) if vectors else np.empty((0, 128), dtype=np.float32)
    np.savez_compressed(path, names=np.asarray(names), embeddings=matrix, version=np.asarray([1]))


def load_database(path: str | Path = DEFAULT_DATABASE_PATH) -> dict[str, list[np.ndarray]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding database not found: {path}. Run face_database.py --build.")
    database: dict[str, list[np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as stored:
        names = stored["names"]
        vectors = stored["embeddings"]
    if len(names) != len(vectors):
        raise ValueError("Invalid database: names and embeddings have different lengths")
    for name, vector in zip(names, vectors):
        database.setdefault(str(name), []).append(np.asarray(vector, dtype=np.float32))
    return database


def build_database(enrollment_dir: str | Path = DEFAULT_ENROLLMENT_DIR) -> dict[str, list[np.ndarray]]:
    root = Path(enrollment_dir)
    root.mkdir(parents=True, exist_ok=True)
    database: dict[str, list[np.ndarray]] = {}
    for identity_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        embeddings, _ = get_embeddings_from_folder(identity_dir)
        if embeddings:
            database[identity_dir.name] = embeddings
        else:
            print(f"No valid enrollment faces found for {identity_dir.name}")
    return database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or inspect the local embedding database.")
    parser.add_argument("--build", action="store_true", help="build embeddings from data/enrolled")
    parser.add_argument("--list", action="store_true", help="list identities in the saved database")
    parser.add_argument("--enrollment-dir", type=Path, default=DEFAULT_ENROLLMENT_DIR)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.build and not args.list:
        raise SystemExit("Choose --build and/or --list. See --help.")
    if args.build:
        database = build_database(args.enrollment_dir)
        sample_count = sum(len(items) for items in database.values())
        identity_dirs = sorted(path for path in args.enrollment_dir.iterdir() if path.is_dir())
        if sample_count == 0:
            print("\nDATA COLLECTION REQUIRED: no valid enrollment images were found.")
            print("Add 5-10 permitted or clearly synthetic face images to each folder:")
            if identity_dirs:
                for identity_dir in identity_dirs:
                    print(f"  {identity_dir}/image_01.jpg")
            else:
                print(f"  {args.enrollment_dir}/Person_A/image_01.jpg")
            print("The existing embedding database was not changed.")
            raise SystemExit(1)

        save_database(database, args.database)
        print("\nEnrollment complete\n")
        for identity_dir in identity_dirs:
            count = len(database.get(identity_dir.name, []))
            print(f"{identity_dir.name}: {count} valid image(s)")
        print(f"\nTotal enrolled identities: {len(database)}")
        print(f"Total embeddings: {sample_count}")
        print(f"\nSaved to:\n{args.database}")
    if args.list:
        database = load_database(args.database)
        for name, references in sorted(database.items()):
            print(f"{name}: {len(references)} reference embedding(s)")


if __name__ == "__main__":
    main()
