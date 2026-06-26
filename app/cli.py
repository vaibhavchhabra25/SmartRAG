"""Quick CLI for testing the pipeline: python -m app.cli "your question here"."""
import sys

from app.graph import ask


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m app.cli "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = ask(question)

    print(f"\nQ: {question}\n")
    print(result.get("final_answer", "<no answer>"))

    citations = result.get("citations")
    if citations:
        print("\nCitations:")
        for c in citations:
            print(f"  - {c['source_doc']} §{c['section']}")

    safety = result.get("safety", {})
    print(f"\n[safety: {safety}]")


if __name__ == "__main__":
    main()
