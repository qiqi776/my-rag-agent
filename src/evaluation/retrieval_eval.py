"""Retrieval regression runner."""

from __future__ import annotations

from src.evaluation.models import (
    RetrievalEvalCase,
    RetrievalEvalCaseResult,
    RetrievalEvalReport,
    SearchServiceLike,
)


class RetrievalEvalRunner:
    """Run retrieval regression cases against SearchService."""

    def __init__(self, search_service: SearchServiceLike) -> None:
        self.search_service = search_service

    def run(self, cases: list[RetrievalEvalCase]) -> RetrievalEvalReport:
        results: list[RetrievalEvalCaseResult] = []

        for case in cases:
            if (
                not case.expected_doc_ids
                and not case.expected_chunk_ids
                and not case.expected_source_paths
            ):
                raise ValueError(
                    f"Retrieval eval case '{case.name}' must define expected_doc_ids, "
                    "expected_chunk_ids, or expected_source_paths"
                )
            output = self.search_service.search(
                case.query,
                collection=case.collection,
                top_k=case.top_k,
                mode=case.mode,
            )
            returned_doc_ids = [item.doc_id for item in output.results]
            returned_chunk_ids = [item.chunk_id for item in output.results]
            returned_source_paths = [item.source_path for item in output.results]
            matched_doc_ids = sorted(set(returned_doc_ids) & set(case.expected_doc_ids))
            matched_chunk_ids = sorted(set(returned_chunk_ids) & set(case.expected_chunk_ids))
            matched_source_paths = sorted(
                expected
                for expected in case.expected_source_paths
                if any(path.endswith(expected) for path in returned_source_paths)
            )

            expected_total = (
                len(case.expected_doc_ids)
                + len(case.expected_chunk_ids)
                + len(case.expected_source_paths)
            )
            matched_total = (
                len(matched_doc_ids) + len(matched_chunk_ids) + len(matched_source_paths)
            )
            hit_at_k = matched_total > 0
            recall_at_k = matched_total / expected_total

            results.append(
                RetrievalEvalCaseResult(
                    name=case.name,
                    passed=recall_at_k == 1.0,
                    hit_at_k=hit_at_k,
                    recall_at_k=round(recall_at_k, 2),
                    returned_doc_ids=returned_doc_ids,
                    returned_chunk_ids=returned_chunk_ids,
                    returned_source_paths=returned_source_paths,
                    matched_doc_ids=matched_doc_ids,
                    matched_chunk_ids=matched_chunk_ids,
                    matched_source_paths=matched_source_paths,
                )
            )

        total_cases = len(results)
        return RetrievalEvalReport(
            total_cases=total_cases,
            passed_cases=sum(1 for result in results if result.passed),
            average_hit_at_k=round(
                sum(1.0 if result.hit_at_k else 0.0 for result in results) / total_cases,
                2,
            )
            if total_cases
            else 0.0,
            average_recall_at_k=round(
                sum(result.recall_at_k for result in results) / total_cases,
                2,
            )
            if total_cases
            else 0.0,
            cases=results,
        )
