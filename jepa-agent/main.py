"""JEPA Coding Agent — CLI entry point."""

import argparse
import asyncio
import sys
import os

# Ensure DEEPSEEK_API_KEY is set
if not os.environ.get("DEEPSEEK_API_KEY"):
    print("Warning: DEEPSEEK_API_KEY env var not set.")
    print("Set it or add to config.py before running.")

from agent import run_agent, JEPAAgent


async def _run_onboard(file_path: str, install_parsers: bool = False):
    """Run onboarding standalone and print profile."""
    agent = JEPAAgent()
    try:
        result = await agent.onboard(file_path, install_parsers=install_parsers)
        profile = result.get("profile", {})
        if profile:
            print("\n" + "=" * 60)
            print("  Project Profile")
            print("=" * 60)
            print(f"  Project:       {profile.get('project_name', '?')}")
            print(f"  Root:          {profile.get('project_root', '?')}")
            print(f"  Primary lang:  {profile.get('primary_language', '?')}")
            print(f"  Extensions:    {', '.join(profile.get('all_extensions', []))}")
            print(f"  Languages:")
            for lang in profile.get("languages", []):
                print(f"    - {lang['language']:12s} (conf: {lang.get('confidence', 0)*100:.0f}%, {lang['detected_by']})")
            missing = result.get("missing_parsers", [])
            if missing:
                print(f"\n  Missing parsers: {', '.join(missing)}")
                if not install_parsers:
                    print("  (use --install-parsers to auto-install)")
            print("=" * 60)
        else:
            print(f"  Onboard result: {result.get('status', '?')}")
        return result
    finally:
        await agent.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="JEPA-style coding agent (MCP-orchestrated): DeepSeek + CodeT5+"
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Run project onboarding only (detect languages, check parsers), then exit",
    )
    parser.add_argument(
        "--install-parsers",
        action="store_true",
        help="When used with --init, auto-install missing tree-sitter parsers",
    )
    parser.add_argument(
        "--task", "-t",
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

    # Standalone onboarding mode
    if args.init:
        print("=" * 60)
        print("  JEPA Project Onboarding")
        print("=" * 60)
        print(f"  File: {args.file}")
        if args.install_parsers:
            print("  Install parsers: yes")
        print("=" * 60)
        asyncio.run(_run_onboard(args.file, install_parsers=args.install_parsers))
        return

    if not args.task:
        parser.error("--task is required unless --init is specified")

    print("=" * 60)
    print("  JEPA Coding Agent — MCP Orchestrated")
    print("  DeepSeek + CodeT5+ via 6 MCP Servers")
    print("=" * 60)
    print(f"  Task:      {args.task}")
    print(f"  File:      {args.file}")
    print(f"  Candidates: {args.candidates}")
    print(f"  Max steps: {args.steps}")
    print(f"  Loss:      {args.loss}")
    print("=" * 60)

    results = run_agent(
        task=args.task,
        file_path=args.file,
        k=args.candidates,
        steps=args.steps,
    )

    for step_result in results:
        print("\n" + "=" * 60)
        print(f"  Step {step_result.get('step', '?')} Result")
        print("=" * 60)
        if "error" in step_result:
            print(f"  ERROR: {step_result['error']}")
        else:
            print(f"  Selected candidate: {step_result.get('best_idx', '?')}")
            print(f"  Description:       {step_result.get('best_description', '?')}")
            print(f"  JEPA loss:         {step_result.get('jepa_loss', 1.0):.4f}")
            print(f"  Applied:           {'✓' if step_result.get('success') else '✗'}")
            if step_result.get("all_losses"):
                print(f"  All losses:         {[f'{s:.4f}' for s in step_result['all_losses']]}")
        print("=" * 60)


if __name__ == "__main__":
    main()
