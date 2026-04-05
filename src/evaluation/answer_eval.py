"""Answer regression runner."""

from __future__ import annotations

from src.evaluation.models import AnswerEvalCase, AnswerEvalCaseResult, AnswerEvalReport


class AnswerEvalRunner:
    """Run answer regression cases against AnswerService."""

    def __init__(self, answer_service: object) -> None:
        self.answer_service = answer_service

    def run(self, cases: list[AnswerEvalCase]) -> AnswerEvalReport:
        results: list[AnswerEvalCaseResult] = []

        for case in cases:
            output = self.answer_service.answer(
                case.query,
                collection=case.collection,
                top_k=case.top_k,
                mode=case.mode,
            )
            answer_text = output.answer.lower()
            matched_keywords = [
                keyword for keyword in case.expected_keywords if keyword.lower() in answer_text
            ]
            missing_keywords = [
                keyword for keyword in case.expected_keywords if keyword not in matched_keywords
            ]
            matched_source_paths = [
                expected
                for expected in case.expected_source_paths
                if any(citation.source_path.endswith(expected) for citation in output.citations)
            ]
            missing_source_paths = [
                expected
                for expected in case.expected_source_paths
                if expected not in matched_source_paths
            ]

            keyword_coverage = (
                len(matched_keywords) / len(case.expected_keywords)
                if case.expected_keywords
                else 1.0
            )
            source_coverage = (
                len(matched_source_paths) / len(case.expected_source_paths)
                if case.expected_source_paths
                else 1.0
            )
            answer_nonempty = bool(output.answer.strip())

            results.append(
                AnswerEvalCaseResult(
                    name=case.name,
                    passed=answer_nonempty and keyword_coverage == 1.0 and source_coverage == 1.0,
                    answer_nonempty=answer_nonempty,
                    keyword_coverage=round(keyword_coverage, 2),
                    source_coverage=round(source_coverage, 2),
                    citation_count=len(output.citations),
                    matched_keywords=matched_keywords,
                    missing_keywords=missing_keywords,
                    matched_source_paths=matched_source_paths,
                    missing_source_paths=missing_source_paths,
                )
            )

        total_cases = len(results)
        return AnswerEvalReport(
            total_cases=total_cases,
            passed_cases=sum(1 for result in results if result.passed),
            average_keyword_coverage=round(
                sum(result.keyword_coverage for result in results) / total_cases,
                2,
            )
            if total_cases
            else 0.0,
            average_source_coverage=round(
                sum(result.source_coverage for result in results) / total_cases,
                2,
            )
            if total_cases
            else 0.0,
            average_citation_count=round(
                sum(result.citation_count for result in results) / total_cases,
                2,
            )
            if total_cases
            else 0.0,
            cases=results,
        )
