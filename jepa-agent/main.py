"""JEPA Coding Agent — CLI entry point."""

import argparse
import sys
import os

# Ensure DEEPSEEK_API_KEY is set
if not os.environ.get("DEEPSEEK_API_KEY"):
    print("Warning: DEEPSEEK_API_KEY env var not set.")
    print("Set it or add to config.py before running.")

from agent import JEPAAgent


def main():
    parser = argparse.ArgumentParser(
        description="JEPA-style coding agent: DeepSeek + CodeT5+"
    )
    parser.add_argument(
        "--task", "-t",
        required=True,
        help="Description of the fix or change to make",
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the file to modify",
    )
    parser.add_argument(
        "--candidates", "-k",
        type=int,
        default=5,
        help="Number of candidate patches per step (default: 5)",
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=3,
        help="Maximum JEPA loop steps (default: 3)",
    )
    parser.add_argument(
        "--loss",
        choices=["cosine", "l2"],
        default="cosine",
        help="JEPA loss type (default: cosine)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  JEPA Coding Agent — DeepSeek + CodeT5+")
    print("=" * 60)
    print(f"  Task:      {args.task}")
    print(f"  File:      {args.file}")
    print(f"  Candidates: {args.candidates}")
    print(f"  Max steps: {args.steps}")
    print(f"  Loss:      {args.loss}")
    print("=" * 60)

    agent = JEPAAgent()

    result = agent.step(
        task=args.task,
        file_path=args.file,
        k=args.candidates,
    )

    print("\n" + "=" * 60)
    print("  Result")
    print("=" * 60)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
    else:
        print(f"  Selected candidate: {result['best_idx']}")
        print(f"  Description:       {result['best_description']}")
        print(f"  JEPA loss:         {result['jepa_loss']:.4f}")
        print(f"  Applied:           {'✓' if result['success'] else '✗'}")
        if result.get("all_losses"):
            print(f"  All losses:         {[f'{s:.4f}' for s in result['all_losses']]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
