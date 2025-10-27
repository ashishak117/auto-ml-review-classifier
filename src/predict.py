import argparse
import sys
from utils import load_model


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment for a review.")
    parser.add_argument("--model", type=str, required=True, help="Path to joblib model")
    parser.add_argument(
        "--text",
        type=str,
        help='Text to classify. If omitted, reads from STDIN (end with Ctrl+D / Ctrl+Z).',
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.text:
        texts = [args.text]
    else:
        print("Enter text (one or more lines). Press Ctrl+D (macOS/Linux) or Ctrl+Z (Windows) when done:")
        raw = sys.stdin.read().strip()
        texts = [t for t in raw.splitlines() if t.strip()]

    preds = model.predict(texts)
    for t, p in zip(texts, preds):
        print(f"\nTEXT: {t}\nPREDICTION: {p}")


if __name__ == "__main__":
    main()
