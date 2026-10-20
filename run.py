"""Discover cutoff-safe trends with local embeddings, sampled Jev decisions and optional hosted drafting."""
import argparse
import sys
from pathlib import Path

from trend_hunter.config import ConfigurationError, resolve_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--versions")
    parser.add_argument("--review-out")
    parser.add_argument("--cache", default=".cache/models.sqlite3")
    parser.add_argument("--jev-cache", default=".cache/decisions.sqlite3")
    parser.add_argument("--ollama-url", help="Override TREND_OLLAMA_URL")
    parser.add_argument("--embedding-model", help="Override TREND_EMBEDDING_MODEL")
    parser.add_argument("--chat-provider", choices=("ollama", "openai"), help="Override TREND_CHAT_PROVIDER")
    parser.add_argument("--chat-model", help="Override the selected provider's model setting")
    parser.add_argument("--jev-model", help="Override TREND_JEV_MODEL")
    parser.add_argument("--max-reviews", "--max-audits", type=int, default=None,
                        help="Optional review-attempt budget; unset reviews until enough outputs or exhaustion")
    parser.add_argument("--max-units", type=int, default=0, help="Development input sample; zero uses all eligible units")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cluster-size", type=int, default=80)
    parser.add_argument("--embedding-num-gpu", type=int)
    parser.add_argument("--embedding-residency-inputs", type=int, default=2048,
                        help="Release BGE between input windows; zero keeps it loaded through ingestion")
    parser.add_argument("--chat-context", type=int, default=16384)
    parser.add_argument("--chat-batch-size", type=int, default=64)
    parser.add_argument("--review-batch", type=int, default=10)
    parser.add_argument("--exploration-ratio", type=float, default=.5)
    parser.add_argument("--max-refinements", type=int, default=1)
    parser.add_argument("--neighbor-reviews", type=int, default=16,
                        help="Additional sampled neighbor reviews before final drafting; zero disables reconciliation")
    parser.add_argument("--draft-retries", type=int, default=1)
    parser.add_argument("--accept-ratio", type=float, default=.8)
    parser.add_argument("--provisional-ratio", type=float, default=.5)
    parser.add_argument("--jev-token-target", type=int, default=20_000_000)
    parser.add_argument("--jev-token-max", type=int, default=50_000_000)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Build numerical candidates without generation or Jev; OpenAI provider requires cached embeddings")
    args = parser.parse_args()
    try:
        models = resolve_models(embedding_model=args.embedding_model, chat_provider=args.chat_provider,
                                chat_model=args.chat_model, jev_model=args.jev_model,
                                ollama_url=args.ollama_url, prepare_only=args.prepare_only)
    except ConfigurationError as exc:
        parser.error(str(exc))
    if (args.max_reviews is not None and args.max_reviews < 1) or min(args.max_units, args.embedding_residency_inputs) < 0:
        parser.error("Review budget must be positive when set; max-units must be nonnegative")
    if min(args.batch_size, args.review_batch, args.chat_context, args.chat_batch_size) < 1 or args.cluster_size < 2:
        parser.error("Batch/context sizes must be positive; cluster-size must be at least two")
    if min(args.max_refinements, args.draft_retries, args.neighbor_reviews) < 0 or not 0 <= args.exploration_ratio <= 1:
        parser.error("Invalid refinement, retry or exploration setting")
    if not 0 <= args.provisional_ratio <= args.accept_ratio <= 1:
        parser.error("Require 0 <= provisional-ratio <= accept-ratio <= 1")
    try:
        from trend_hunter.pipeline import run, write_json
        result, review = run(args.input, args.as_of,
            releases_path=args.versions or str(Path(args.input).with_name("versions.csv")),
            cache_path=args.cache, decision_cache_path=args.jev_cache,
            base_url=models.ollama_url, embedding_model=models.embedding_model, chat_model=models.chat_model,
            chat_provider=models.chat_provider,
            max_reviews=args.max_reviews, max_units=args.max_units, batch_size=args.batch_size,
            cluster_size=args.cluster_size, embedding_num_gpu=args.embedding_num_gpu,
            embedding_residency_inputs=args.embedding_residency_inputs,
            chat_context=args.chat_context, chat_batch_size=args.chat_batch_size,
            review_batch=args.review_batch, exploration_ratio=args.exploration_ratio,
            max_refinements=args.max_refinements, draft_retries=args.draft_retries,
            neighbor_reviews=args.neighbor_reviews,
            accept_ratio=args.accept_ratio, provisional_ratio=args.provisional_ratio,
            jev_model=models.jev_model, jev_token_target=args.jev_token_target, jev_token_max=args.jev_token_max,
            prepare_only=args.prepare_only)
        write_json(args.out, result)
        if args.review_out:
            write_json(args.review_out, review)
        print(f"Wrote {len(result['trends'])} trends to {args.out}; stop={review.get('stop_reason')}", file=sys.stderr)
        return 2 if (review.get("stop_reason") in {"decision_service_error", "local_model_error", "generation_service_error", "embedding_cache_error"}
                     or review.get("local_cleanup_errors")) else 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Trend Hunter failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

