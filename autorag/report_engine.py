"""
Report Engine Subsystem for AutoRAG Platform.
Generates and exports evaluation summaries in JSON, Markdown, CSV, and interactive HTML.
"""

import json
import csv
import io
import math
from typing import List, Dict, Any, Optional
from datetime import datetime

from .workspace import WorkspaceManager


def _get_entry_composite_score(entry: Dict[str, Any]) -> Optional[float]:
    """Canonical score extractor for report exports."""
    score = entry.get("composite_score")
    if score is None and isinstance(entry.get("results"), dict):
        score = entry.get("results", {}).get("composite_score")
    if isinstance(score, (int, float)) and math.isfinite(score):
        return float(score)
    return None


class ReportEngine:
    """Exports evaluation reports and optimization comparison summaries."""

    def __init__(self, workspace: WorkspaceManager):
        self.workspace = workspace

    def export_json(self, data: Dict[str, Any], filename: str) -> str:
        content = json.dumps(data, indent=2, default=str)
        return self.workspace.save_report(filename, content, extension="json")

    def export_markdown(self, leaderboard: List[Dict[str, Any]], filename: str) -> str:
        lines = [
            "# 🚀 AutoRAG Pipeline Optimization Report",
            f"**Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## 🏆 Leaderboard Summary\n",
            "| Rank | Trial ID | Composite Score | Hit Rate | Answer Correctness | Completeness | Relevance | Chunk Size | Strategy | Top K |",
            "|:----:|:--------:|:---------------:|:--------:|:------------------:|:------------:|:---------:|:----------:|:--------:|:-----:|"
        ]

        for rank, entry in enumerate(leaderboard, 1):
            c = entry.get("config") or {}
            chunk_cfg = c.get("chunking_config", {})
            ret_cfg = c.get("retriever_config", {})
            score = _get_entry_composite_score(entry)
            score_str = f"{score:.4f}" if score is not None else "N/A"
            ans_m = entry.get("results", {}).get("answer_metrics", {})
            ans_corr = ans_m.get("answer_correctness", ans_m.get("accuracy", ans_m.get("faithfulness", 0.0)))
            
            lines.append(
                f"| {rank} | `{entry.get('experiment_id', 'N/A')}` | "
                f"**{score_str}** | "
                f"{entry.get('results', {}).get('retrieval_metrics', {}).get('hit_rate', 0.0):.4f} | "
                f"{ans_corr:.4f} | "
                f"{ans_m.get('completeness', 0.0):.4f} | "
                f"{ans_m.get('relevance', 0.0):.4f} | "
                f"{chunk_cfg.get('strategy', '')} ({chunk_cfg.get('chunk_size', '')}/{chunk_cfg.get('chunk_overlap', '')}) | "
                f"{ret_cfg.get('strategy', 'hybrid')} (Dist: {ret_cfg.get('distance_metric', '')}, K: {ret_cfg.get('top_k', '')}) |"
            )

        if leaderboard:
            best = leaderboard[0]
            best_c = best.get("config") or {}
            env = best.get("environment") or {}
            res = best.get("results") or {}
            best_score = _get_entry_composite_score(best)
            best_score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
            
            lines.extend([
                "\n## 💡 Recommended Configuration\n",
                "### Pipeline Summary",
                f"- **Pipeline ID:** `{best.get('pipeline_id', 'N/A')}`",
                f"- **LLM Provider:** `{best_c.get('llm_config', {}).get('provider')}` - `{best_c.get('llm_config', {}).get('model_name')}`",
                f"- **Embedding:** `{best_c.get('embedding_config', {}).get('model_name')}`",
                f"- **Chunking Strategy:** `{best_c.get('chunking_config', {}).get('strategy')}`",
                f"- **Chunk Size/Overlap:** `{best_c.get('chunking_config', {}).get('chunk_size')}` / `{best_c.get('chunking_config', {}).get('chunk_overlap')}`",
                f"- **Retriever Strategy:** `{best_c.get('retriever_config', {}).get('strategy')}`",
                f"- **Distance Metric:** `{best_c.get('retriever_config', {}).get('distance_metric')}`",
                f"- **Top-K Chunks:** `{best_c.get('retriever_config', {}).get('top_k')}`",
                
                "\n### Dataset Summary",
                f"- **Dataset ID:** `{best.get('dataset_id', 'N/A')}`",
                
                "\n### Experiment Metadata",
                f"- **Experiment ID:** `{best.get('experiment_id', 'N/A')}`",
                f"- **Timestamp:** `{best.get('timestamp', 'N/A')}`",
                f"- **Runtime:** `{best.get('runtime', 0.0)} seconds`",
                
                "\n### Environment Snapshot",
                f"- **Python Version:** `{env.get('python_version', 'N/A')}`",
                f"- **OS:** `{env.get('os', 'N/A')}`",
                f"- **Architecture:** `{env.get('architecture', 'N/A')}`",
                
                "\n### Results",
                f"- **Overall Composite Score:** `{best_score_str}`",
                "**Retrieval Metrics:**",
                f"  - Hit Rate: `{res.get('retrieval_metrics', {}).get('hit_rate', 0.0):.4f}`",
                f"  - Precision: `{res.get('retrieval_metrics', {}).get('precision', 0.0):.4f}`",
                f"  - MRR: `{res.get('retrieval_metrics', {}).get('mrr', 0.0):.4f}`",
                "**Answer Metrics:**",
                f"  - Accuracy: `{res.get('answer_metrics', {}).get('accuracy', 0.0):.4f}`",
                f"  - Completeness: `{res.get('answer_metrics', {}).get('completeness', 0.0):.4f}`",
                f"  - Relevance: `{res.get('answer_metrics', {}).get('relevance', 0.0):.4f}`",
                
                "\n### Artifacts",
            ])
            artifacts = best.get("artifacts") or []
            if artifacts:
                for a in artifacts:
                    lines.append(f"- `{a.get('type')}`: {a.get('hash')} ({a.get('path')})")
            else:
                lines.append("- (No artifacts explicitly saved in this trial record)")

        content = "\n".join(lines)
        return self.workspace.save_report(filename, content, extension="md")

    def export_csv(self, leaderboard: List[Dict[str, Any]], filename: str) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            "Rank", "ExperimentID", "CompositeScore", "HitRate", "Precision",
            "AnswerCorrectness", "Completeness", "Relevance", "LLMProvider", "LLMModel",
            "EmbeddingModel", "ChunkStrategy", "ChunkSize", "ChunkOverlap",
            "RetrieverStrategy", "DistanceMetric", "TopK"
        ])

        for rank, entry in enumerate(leaderboard, 1):
            c = entry.get("config") or {}
            res = entry.get("results") or {}
            ret_m = res.get("retrieval_metrics", {})
            ans_m = res.get("answer_metrics", {})
            chunk_cfg = c.get("chunking_config", {})
            ret_cfg = c.get("retriever_config", {})
            score = _get_entry_composite_score(entry)
            ans_corr = ans_m.get("answer_correctness", ans_m.get("accuracy", ans_m.get("faithfulness", 0.0)))

            writer.writerow([
                rank,
                entry.get("experiment_id", ""),
                score if score is not None else "",
                ret_m.get("hit_rate", 0.0),
                ret_m.get("precision", 0.0),
                ans_corr,
                ans_m.get("completeness", 0.0),
                ans_m.get("relevance", 0.0),
                c.get("llm_config", {}).get("provider", ""),
                c.get("llm_config", {}).get("model_name", ""),
                c.get("embedding_config", {}).get("model_name", ""),
                chunk_cfg.get("strategy", ""),
                chunk_cfg.get("chunk_size", 0),
                chunk_cfg.get("chunk_overlap", 0),
                ret_cfg.get("strategy", ""),
                ret_cfg.get("distance_metric", ""),
                ret_cfg.get("top_k", 0)
            ])

        return self.workspace.save_report(filename, output.getvalue(), extension="csv")

    def export_html(self, leaderboard: List[Dict[str, Any]], filename: str) -> str:
        best = leaderboard[0] if leaderboard else {}
        best_score = _get_entry_composite_score(best)
        best_score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
        
        rows_html = ""
        for rank, entry in enumerate(leaderboard, 1):
            c = entry.get("config") or {}
            chunk_cfg = c.get("chunking_config", {})
            ret_cfg = c.get("retriever_config", {})
            score = _get_entry_composite_score(entry)
            score_str = f"{score:.4f}" if score is not None else "N/A"
            ans_m = entry.get("results", {}).get("answer_metrics", {})
            ans_corr = ans_m.get("answer_correctness", ans_m.get("accuracy", ans_m.get("faithfulness", 0.0)))
            rows_html += f"""
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 12px; font-weight: bold;">#{rank}</td>
                <td style="padding: 12px; font-family: monospace;">{entry.get('experiment_id', '')}</td>
                <td style="padding: 12px; font-weight: bold; color: #2563eb;">{score_str}</td>
                <td style="padding: 12px;">{entry.get('results', {}).get('retrieval_metrics', {}).get('hit_rate', 0.0):.4f}</td>
                <td style="padding: 12px;">{ans_corr:.4f}</td>
                <td style="padding: 12px;">{chunk_cfg.get('chunk_size', 512)}</td>
                <td style="padding: 12px;"><span style="background: #eff6ff; color: #1d4ed8; padding: 2px 8px; border-radius: 4px;">{ret_cfg.get('strategy', 'hybrid')}</span></td>
                <td style="padding: 12px;">{ret_cfg.get('top_k', 4)}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AutoRAG Optimization Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 40px; }}
        .card {{ background: #ffffff; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }}
        h1 {{ margin-top: 0; color: #1e293b; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th {{ background: #f1f5f9; padding: 12px; font-size: 14px; color: #475569; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🚀 AutoRAG Pipeline Optimization Executive Summary</h1>
        <p style="color: #64748b;">Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <div style="background: #f0fdf4; border-left: 4px solid #16a34a; padding: 16px; border-radius: 4px; margin-top: 16px;">
            <h3 style="margin: 0 0 8px 0; color: #15803d;">Top Recommended Pipeline</h3>
            <p style="margin: 0; font-size: 15px;">
                Highest Composite Score: <strong>{best_score_str}</strong> using Chunk Size 
                <strong>{best.get('config', {}).get('chunking_config', {}).get('chunk_size', 512)}</strong> & 
                Strategy <strong>{best.get('config', {}).get('retriever_config', {}).get('strategy', 'hybrid')}</strong>.
            </p>
        </div>
    </div>
    <div class="card">
        <h2>📊 Experiment Leaderboard</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th><th>Trial ID</th><th>Composite Score</th><th>Hit Rate</th><th>Answer Correctness</th><th>Chunk Size</th><th>Retriever</th><th>Top K</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        return self.workspace.save_report(filename, html_content, extension="html")
