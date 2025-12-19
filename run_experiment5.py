"""
Experiment 5: Market-Sense Agent Evaluation
=============================================

Collects responses from Market-Sense Agent for 20 economic/market questions.
Results are saved for later evaluation.

Usage:
    python run_experiment5.py --output results/experiment5_results
"""

import argparse
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any
from llm.llm_client import LLMClient

# Import Market-Sense Agent
from agents.market_sense_agent import MarketSenseAgent
from config.config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== QUESTIONS ==============

QUESTIONS = {
    "A": {
        "name": "Macroeconomic Understanding",
        "questions": [
            {
                "id": "A1",
                "question": "How do rising interest rates typically affect growth stocks vs value stocks?"
            },
            {
                "id": "A2",
                "question": "What is the relationship between inflation and stock market performance?"
            },
            {
                "id": "A3",
                "question": "How does a strong US dollar impact multinational companies?"
            },
            {
                "id": "A4",
                "question": "What economic indicators should investors watch for recession signals?"
            },
            {
                "id": "A5",
                "question": "How do Federal Reserve policy decisions affect different market sectors?"
            }
        ]
    },
    "B": {
        "name": "Sector Analysis",
        "questions": [
            {
                "id": "B1",
                "question": "What factors drive semiconductor stock prices?"
            },
            {
                "id": "B2",
                "question": "How do oil price changes affect airline stocks?"
            },
            {
                "id": "B3",
                "question": "What are the key metrics for evaluating tech company valuations?"
            },
            {
                "id": "B4",
                "question": "How does consumer confidence affect retail sector stocks?"
            },
            {
                "id": "B5",
                "question": "What makes healthcare stocks defensive investments?"
            }
        ]
    },
    "C": {
        "name": "Historical Events",
        "questions": [
            {
                "id": "C1",
                "question": "What caused the 2022 tech stock crash?"
            },
            {
                "id": "C2",
                "question": "How did COVID-19 initially impact different market sectors?"
            },
            {
                "id": "C3",
                "question": "What happened during the 2023 banking crisis (SVB collapse)?"
            },
            {
                "id": "C4",
                "question": "How did the market react to the 2024 Fed rate decisions?"
            },
            {
                "id": "C5",
                "question": "What drove NVIDIA's stock price increase in 2023-2024?"
            }
        ]
    },
    "D": {
        "name": "Company-Specific Analysis",
        "questions": [
            {
                "id": "D1",
                "question": "What are the main growth drivers for Apple?"
            },
            {
                "id": "D2",
                "question": "What risks does Tesla face as a company?"
            },
            {
                "id": "D3",
                "question": "Why is NVIDIA considered a leader in AI?"
            },
            {
                "id": "D4",
                "question": "What challenges does Intel face in the semiconductor market?"
            },
            {
                "id": "D5",
                "question": "How does Amazon's cloud business (AWS) affect its stock?"
            }
        ]
    }
}


# ============== EXPERIMENT RUNNER ==============

class Experiment5Runner:
    """Collects Market-Sense Agent responses for evaluation."""
    
    def __init__(self, output_dir: str = "results/experiment5"):
        self.output_dir = output_dir
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize LLM client and Market-Sense Agent
        logger.info("Initializing Market-Sense Agent...")
        self.llm_client = LLMClient(config.llm)
        self.agent = MarketSenseAgent(self.llm_client, config)
        
        logger.info(f"Results will be saved to: {output_dir}")
    
    def query_agent(self, question: str, question_id: str) -> Dict[str, Any]:
        """
        Query the Market-Sense Agent with a question.
        
        Args:
            question: The question to ask
            question_id: Question identifier (e.g., "A1")
        
        Returns:
            Dict with question, response, and metadata
        """
        logger.info(f"  Querying: {question_id}")
        
        try:
            # Create context for the agent
            # We use a generic context since these are general knowledge questions
            context = {
                "ticker": "GENERAL",  # Not stock-specific
                "query": question,
                "mode": "analysis",  # Request analysis mode
                "include_news": True,
                "include_knowledge": True
            }
            
            # Call the agent
            response = self.agent.run(context)
            
            # Extract the response content
            if isinstance(response, dict):
                answer = response.get("analysis", response.get("response", str(response)))
                raw_response = response
            else:
                answer = str(response)
                raw_response = {"raw": answer}
            
            result = {
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "raw_response": raw_response,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log first 100 chars of response
            preview = answer[:100] + "..." if len(answer) > 100 else answer
            logger.info(f"    ✓ Response: {preview}")
            
            return result
            
        except Exception as e:
            logger.error(f"    ✗ Error: {str(e)}")
            return {
                "question_id": question_id,
                "question": question,
                "answer": None,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def save_incremental(self, result: Dict[str, Any]):
        """Save result immediately after each query."""
        json_path = os.path.join(self.output_dir, f"experiment5_responses_{self.timestamp}.json")
        
        # Load existing results
        existing = []
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    existing = json.load(f)
            except:
                existing = []
        
        # Append and save
        existing.append(result)
        with open(json_path, 'w') as f:
            json.dump(existing, f, indent=2, default=str)
        
        logger.info(f"    💾 Saved ({len(existing)} responses)")
    
    def run_all(self):
        """Run all questions through the Market-Sense Agent."""
        # Count total questions
        total_questions = sum(len(cat["questions"]) for cat in QUESTIONS.values())
        
        logger.info("=" * 70)
        logger.info("EXPERIMENT 5: Market-Sense Agent Evaluation")
        logger.info("=" * 70)
        logger.info(f"Total Questions: {total_questions}")
        logger.info("=" * 70)
        
        question_count = 0
        
        for category_key, category in QUESTIONS.items():
            category_name = category["name"]
            questions = category["questions"]
            
            logger.info(f"\n--- Category {category_key}: {category_name} ---")
            
            for q in questions:
                question_count += 1
                question_id = q["id"]
                question_text = q["question"]
                
                logger.info(f"\n[{question_count}/{total_questions}] {question_id}: {question_text[:50]}...")
                
                result = self.query_agent(question_text, question_id)
                result["category"] = category_key
                result["category_name"] = category_name
                
                self.results.append(result)
                self.save_incremental(result)
        
        logger.info("\n" + "=" * 70)
        logger.info("ALL QUESTIONS COMPLETE")
        logger.info("=" * 70)
        
        # Save final outputs
        self.save_final()
    
    def save_final(self):
        """Save final results in multiple formats."""
        # Save JSON (already done incrementally, but update final)
        json_path = os.path.join(self.output_dir, f"experiment5_responses_{self.timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"JSON saved: {json_path}")
        
        # Save Markdown for easy reading
        md_path = os.path.join(self.output_dir, f"experiment5_responses_{self.timestamp}.md")
        with open(md_path, 'w') as f:
            f.write(self._generate_markdown())
        logger.info(f"Markdown saved: {md_path}")
        
        # Save evaluation template
        template_path = os.path.join(self.output_dir, f"experiment5_evaluation_template_{self.timestamp}.md")
        with open(template_path, 'w') as f:
            f.write(self._generate_evaluation_template())
        logger.info(f"Evaluation template saved: {template_path}")
    
    def _generate_markdown(self) -> str:
        """Generate markdown with all Q&A pairs."""
        lines = []
        lines.append("# Experiment 5: Market-Sense Agent Responses")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        current_category = None
        
        for r in self.results:
            # Category header
            if r.get("category") != current_category:
                current_category = r.get("category")
                category_name = r.get("category_name", current_category)
                lines.append(f"## Category {current_category}: {category_name}")
                lines.append("")
            
            # Question and Answer
            lines.append(f"### {r['question_id']}: {r['question']}")
            lines.append("")
            
            if r.get("success"):
                lines.append("**Answer:**")
                lines.append("")
                lines.append(r.get("answer", "No answer"))
                lines.append("")
            else:
                lines.append(f"**Error:** {r.get('error', 'Unknown error')}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_evaluation_template(self) -> str:
        """Generate evaluation template for manual scoring."""
        lines = []
        lines.append("# Experiment 5: Evaluation Form")
        lines.append("")
        lines.append("## Scoring Criteria (1-5)")
        lines.append("")
        lines.append("| Criterion | 1 (Poor) | 3 (Average) | 5 (Excellent) |")
        lines.append("|-----------|----------|-------------|---------------|")
        lines.append("| Factual Accuracy | Major errors | Some inaccuracies | Fully accurate |")
        lines.append("| Logical Reasoning | Incoherent | Basic logic | Clear, well-structured |")
        lines.append("| Relevance | Off-topic | Partially relevant | Directly addresses question |")
        lines.append("| Completeness | Missing key points | Covers basics | Comprehensive |")
        lines.append("| Trading Applicability | Not actionable | Somewhat useful | Directly actionable |")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Evaluation Table")
        lines.append("")
        lines.append("| Question ID | Factual | Logical | Relevance | Completeness | Applicability | Total (/25) | Notes |")
        lines.append("|-------------|---------|---------|-----------|--------------|---------------|-------------|-------|")
        
        for r in self.results:
            qid = r["question_id"]
            lines.append(f"| {qid} | | | | | | | |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Summary Statistics")
        lines.append("")
        lines.append("| Category | Avg Score | Best Question | Worst Question |")
        lines.append("|----------|-----------|---------------|----------------|")
        lines.append("| A: Macroeconomic | /25 | | |")
        lines.append("| B: Sector Analysis | /25 | | |")
        lines.append("| C: Historical Events | /25 | | |")
        lines.append("| D: Company-Specific | /25 | | |")
        lines.append("| **Overall** | **/25** | | |")
        lines.append("")
        
        return "\n".join(lines)


# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(description="Run Experiment 5: Market-Sense Agent Evaluation")
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/experiment5",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    runner = Experiment5Runner(output_dir=args.output)
    runner.run_all()


if __name__ == "__main__":
    main()
