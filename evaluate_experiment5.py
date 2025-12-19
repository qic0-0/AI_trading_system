"""
Experiment 5: LLM-Based Evaluation of Market-Sense Agent Responses
===================================================================

Reads experiment 5 results and uses LLM to evaluate each response
according to the 5 evaluation criteria.

Usage:
    python evaluate_experiment5.py --results results/experiment5_results
    python evaluate_experiment5.py --results results/experiment5_results --output results/experiment5_evaluation
"""

import argparse
import json
import os
from glob import glob
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import logging

# Use existing LLM client
from llm.llm_client import LLMClient, Message
from config.config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============== EVALUATION PROMPT ==============

EVALUATION_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of AI-generated responses about financial markets and economics.

You will be given a question and an AI agent's response. Evaluate the response on 5 criteria, each scored 1-5:

## Scoring Criteria

1. **Factual Accuracy (1-5)**: Are the facts stated correct?
   - 1: Major factual errors
   - 3: Some minor inaccuracies
   - 5: Fully accurate

2. **Logical Reasoning (1-5)**: Is the reasoning coherent and logical?
   - 1: Incoherent or illogical
   - 3: Basic logic, some gaps
   - 5: Clear, well-structured reasoning

3. **Relevance (1-5)**: Is the response relevant to the question?
   - 1: Off-topic
   - 3: Partially relevant
   - 5: Directly addresses the question

4. **Completeness (1-5)**: Does it cover the key points?
   - 1: Missing most key points
   - 3: Covers basics
   - 5: Comprehensive coverage

5. **Trading Applicability (1-5)**: Would this help a trading decision?
   - 1: Not actionable at all
   - 3: Somewhat useful for trading
   - 5: Directly actionable trading insights

## Response Format

You MUST respond in this exact JSON format:
{
    "factual_accuracy": <score 1-5>,
    "logical_reasoning": <score 1-5>,
    "relevance": <score 1-5>,
    "completeness": <score 1-5>,
    "trading_applicability": <score 1-5>,
    "total": <sum of all scores>,
    "notes": "<brief explanation of scores, 1-2 sentences>"
}

Only output the JSON, nothing else."""

EVALUATION_USER_TEMPLATE = """## Question
{question}

## AI Agent's Response
{answer}

Please evaluate this response according to the 5 criteria and provide scores in JSON format."""


# ============== EVALUATOR CLASS ==============

class Experiment5Evaluator:
    """Evaluates Market-Sense Agent responses using LLM."""
    
    def __init__(self, results_dir: str, output_dir: str = None):
        self.results_dir = results_dir
        self.output_dir = output_dir or results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.evaluations = []
        
        # Initialize LLM client using existing config
        logger.info("Initializing LLM client...")
        self.llm_client = LLMClient(config.llm)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_results(self) -> List[Dict]:
        """Load experiment 5 results from JSON."""
        json_files = glob(os.path.join(self.results_dir, "experiment5_responses_*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No experiment5 results found in {self.results_dir}")
        
        latest_file = max(json_files, key=os.path.getctime)
        print(f"📂 Loading: {latest_file}")
        
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        print(f"   Found {len(results)} Q&A pairs")
        return results
    
    def evaluate_single(self, question: str, answer: str, question_id: str) -> Dict[str, Any]:
        """Evaluate a single Q&A pair using LLM."""
        print(f"   Evaluating {question_id}...")
        
        if not answer or answer.strip() == "":
            return {
                "question_id": question_id,
                "factual_accuracy": 0,
                "logical_reasoning": 0,
                "relevance": 0,
                "completeness": 0,
                "trading_applicability": 0,
                "total": 0,
                "notes": "No answer provided",
                "success": False
            }
        
        try:
            # Build messages for LLM
            messages = [
                Message(role="system", content=EVALUATION_SYSTEM_PROMPT),
                Message(role="user", content=EVALUATION_USER_TEMPLATE.format(
                    question=question,
                    answer=answer
                ))
            ]
            
            # Call LLM using existing client
            response = self.llm_client.chat(messages, temperature=0.3)
            
            # Parse response
            response_text = response.content.strip()
            
            # Try to parse JSON
            # Handle potential markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                response_text = response_text[start_idx:end_idx]
            
            scores = json.loads(response_text)
            
            # Validate and extract scores
            evaluation = {
                "question_id": question_id,
                "factual_accuracy": int(scores.get("factual_accuracy", 0)),
                "logical_reasoning": int(scores.get("logical_reasoning", 0)),
                "relevance": int(scores.get("relevance", 0)),
                "completeness": int(scores.get("completeness", 0)),
                "trading_applicability": int(scores.get("trading_applicability", 0)),
                "total": int(scores.get("total", 0)),
                "notes": scores.get("notes", ""),
                "success": True
            }
            
            # Recalculate total to ensure accuracy
            evaluation["total"] = (
                evaluation["factual_accuracy"] +
                evaluation["logical_reasoning"] +
                evaluation["relevance"] +
                evaluation["completeness"] +
                evaluation["trading_applicability"]
            )
            
            print(f"      ✓ Score: {evaluation['total']}/25")
            return evaluation
            
        except json.JSONDecodeError as e:
            print(f"      ✗ JSON parse error: {e}")
            return {
                "question_id": question_id,
                "factual_accuracy": 0,
                "logical_reasoning": 0,
                "relevance": 0,
                "completeness": 0,
                "trading_applicability": 0,
                "total": 0,
                "notes": f"JSON parse error: {str(e)}",
                "success": False
            }
        except Exception as e:
            print(f"      ✗ Error: {e}")
            return {
                "question_id": question_id,
                "factual_accuracy": 0,
                "logical_reasoning": 0,
                "relevance": 0,
                "completeness": 0,
                "trading_applicability": 0,
                "total": 0,
                "notes": f"Error: {str(e)}",
                "success": False
            }
    
    def save_incremental(self, evaluation: Dict[str, Any]):
        """Save evaluation immediately."""
        json_path = os.path.join(self.output_dir, f"experiment5_evaluation_{self.timestamp}.json")
        
        existing = []
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    existing = json.load(f)
            except:
                existing = []
        
        existing.append(evaluation)
        with open(json_path, 'w') as f:
            json.dump(existing, f, indent=2)
        
        print(f"      💾 Saved ({len(existing)} evaluations)")
    
    def run(self):
        """Run evaluation on all Q&A pairs."""
        results = self.load_results()
        
        print("\n" + "=" * 70)
        print("EXPERIMENT 5: LLM-Based Evaluation")
        print("=" * 70)
        print(f"Evaluating {len(results)} responses...")
        print("=" * 70)
        
        for idx, r in enumerate(results):
            question_id = r.get("question_id", f"Q{idx+1}")
            question = r.get("question", "")
            answer = r.get("answer", "")
            
            print(f"\n[{idx+1}/{len(results)}] {question_id}")
            
            evaluation = self.evaluate_single(question, answer, question_id)
            evaluation["category"] = r.get("category", "")
            evaluation["category_name"] = r.get("category_name", "")
            evaluation["question"] = question
            
            self.evaluations.append(evaluation)
            self.save_incremental(evaluation)
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
        
        self.save_final()
        self.print_summary()
    
    def save_final(self):
        """Save final results in multiple formats."""
        # JSON (already saved incrementally)
        json_path = os.path.join(self.output_dir, f"experiment5_evaluation_{self.timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.evaluations, f, indent=2)
        print(f"📄 JSON saved: {json_path}")
        
        # CSV
        csv_path = os.path.join(self.output_dir, f"experiment5_evaluation_{self.timestamp}.csv")
        df = pd.DataFrame(self.evaluations)
        df.to_csv(csv_path, index=False)
        print(f"📄 CSV saved: {csv_path}")
        
        # Markdown report
        md_path = os.path.join(self.output_dir, f"experiment5_evaluation_{self.timestamp}.md")
        with open(md_path, 'w') as f:
            f.write(self._generate_markdown_report())
        print(f"📄 Markdown saved: {md_path}")
    
    def _generate_markdown_report(self) -> str:
        """Generate comprehensive markdown evaluation report."""
        lines = []
        lines.append("# Experiment 5: Market-Sense Agent Evaluation Results")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Scoring criteria reference
        lines.append("## Scoring Criteria")
        lines.append("")
        lines.append("| Criterion | Score (1-5) | Description |")
        lines.append("|-----------|-------------|-------------|")
        lines.append("| Factual Accuracy | 1-5 | Are the facts stated correct? |")
        lines.append("| Logical Reasoning | 1-5 | Is the reasoning coherent and logical? |")
        lines.append("| Relevance | 1-5 | Is the response relevant to the question? |")
        lines.append("| Completeness | 1-5 | Does it cover the key points? |")
        lines.append("| Trading Applicability | 1-5 | Would this help a trading decision? |")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Main evaluation table
        lines.append("## Evaluation Results")
        lines.append("")
        lines.append("| Question ID | Factual | Logical | Relevance | Completeness | Applicability | Total | Notes |")
        lines.append("|-------------|---------|---------|-----------|--------------|---------------|-------|-------|")
        
        for e in self.evaluations:
            notes = e.get("notes", "")[:50] + "..." if len(e.get("notes", "")) > 50 else e.get("notes", "")
            lines.append(
                f"| {e['question_id']} | {e['factual_accuracy']} | {e['logical_reasoning']} | "
                f"{e['relevance']} | {e['completeness']} | {e['trading_applicability']} | "
                f"**{e['total']}/25** | {notes} |"
            )
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Category summaries
        lines.append("## Summary by Category")
        lines.append("")
        
        categories = {}
        for e in self.evaluations:
            cat = e.get("category", "Unknown")
            cat_name = e.get("category_name", cat)
            if cat not in categories:
                categories[cat] = {"name": cat_name, "scores": [], "evaluations": []}
            categories[cat]["scores"].append(e["total"])
            categories[cat]["evaluations"].append(e)
        
        lines.append("| Category | Avg Score | Min | Max | Questions |")
        lines.append("|----------|-----------|-----|-----|-----------|")
        
        for cat_key in sorted(categories.keys()):
            cat = categories[cat_key]
            scores = cat["scores"]
            avg = sum(scores) / len(scores) if scores else 0
            min_score = min(scores) if scores else 0
            max_score = max(scores) if scores else 0
            lines.append(f"| {cat_key}: {cat['name']} | **{avg:.1f}/25** | {min_score} | {max_score} | {len(scores)} |")
        
        # Overall
        all_scores = [e["total"] for e in self.evaluations]
        overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
        lines.append(f"| **OVERALL** | **{overall_avg:.1f}/25** | {min(all_scores) if all_scores else 0} | {max(all_scores) if all_scores else 0} | {len(all_scores)} |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Criteria breakdown
        lines.append("## Average Scores by Criterion")
        lines.append("")
        
        criteria = ["factual_accuracy", "logical_reasoning", "relevance", "completeness", "trading_applicability"]
        criteria_names = ["Factual Accuracy", "Logical Reasoning", "Relevance", "Completeness", "Trading Applicability"]
        
        lines.append("| Criterion | Average Score |")
        lines.append("|-----------|---------------|")
        
        for criterion, name in zip(criteria, criteria_names):
            scores = [e[criterion] for e in self.evaluations if e.get("success")]
            avg = sum(scores) / len(scores) if scores else 0
            lines.append(f"| {name} | **{avg:.2f}/5** |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Detailed notes
        lines.append("## Detailed Notes")
        lines.append("")
        
        for e in self.evaluations:
            lines.append(f"### {e['question_id']}: {e.get('question', '')[:80]}...")
            lines.append("")
            lines.append(f"**Score:** {e['total']}/25")
            lines.append("")
            lines.append(f"**Notes:** {e.get('notes', 'No notes')}")
            lines.append("")
        
        return "\n".join(lines)
    
    def print_summary(self):
        """Print summary to console."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        all_scores = [e["total"] for e in self.evaluations if e.get("success")]
        
        if all_scores:
            avg = sum(all_scores) / len(all_scores)
            print(f"Total Evaluations: {len(self.evaluations)}")
            print(f"Successful: {len(all_scores)}")
            print(f"Average Score: {avg:.1f}/25 ({avg/25*100:.1f}%)")
            print(f"Min Score: {min(all_scores)}/25")
            print(f"Max Score: {max(all_scores)}/25")
        else:
            print("No successful evaluations")
        
        print("=" * 70)


# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(description="Evaluate Experiment 5 responses using LLM")
    
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to experiment 5 results directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (defaults to results directory)"
    )
    
    args = parser.parse_args()
    
    evaluator = Experiment5Evaluator(
        results_dir=args.results,
        output_dir=args.output
    )
    evaluator.run()


if __name__ == "__main__":
    main()
