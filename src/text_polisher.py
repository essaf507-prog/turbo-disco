"""
Module 4: Text Polisher
职责：应用学到的模板，将用户的原始文章润色为学术论文风格
"""

import json
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import uuid

from src.utils import logger, OpenAIClient, save_json, load_json, ensure_dir_exists


class TextPolisher:
    """Polish user text using learned academic templates"""
    
    def __init__(self, output_dir: str = "data/output"):
        """
        Initialize text polisher
        
        Args:
            output_dir: Directory to save polished text and reports
        """
        self.client = OpenAIClient()
        self.output_dir = output_dir
        ensure_dir_exists(output_dir)
        logger.info("TextPolisher initialized")
    
    def polish_text(
        self,
        user_text: str,
        template: Dict[str, Any],
        target_formality: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Polish user text using template
        
        Args:
            user_text: User's original article
            template: Template generated from papers
            target_formality: Target formality level (0-1)
        
        Returns:
            Dictionary with polished text and metadata
        """
        logger.info("Starting text polishing process...")
        
        target_formality = target_formality or template.get("metadata", {}).get("formality_level", 0.8)
        
        # Extract template components
        phrase_library = template.get("phrase_library", {})
        common_patterns = template.get("common_patterns", {})
        
        # Polish the text
        polished_text = self._apply_template(
            user_text,
            phrase_library,
            common_patterns,
            target_formality,
        )
        
        # Generate comparison
        changes = self._identify_changes(user_text, polished_text)
        
        # Generate report
        report = self._generate_polishing_report(user_text, polished_text, changes)
        
        result = {
            "original_text": user_text,
            "polished_text": polished_text,
            "changes": changes,
            "statistics": report,
            "template_used": template.get("template_id"),
        }
        
        logger.info(f"Polishing complete: {len(changes)} changes made")
        return result
    
    def _apply_template(
        self,
        text: str,
        phrase_library: Dict[str, List[str]],
        common_patterns: Dict[str, Any],
        target_formality: float,
    ) -> str:
        """Apply template to polish text using AI"""
        
        system_prompt = """You are an expert academic editor. Your task is to polish user text 
to match the style and tone of high-quality academic papers. Maintain the original meaning 
while improving formality, clarity, and academic tone."""
        
        transitions = ", ".join(phrase_library.get("transitions", [])[:10])
        evidence = ", ".join(phrase_library.get("evidence_markers", [])[:5])
        key_vocab = ", ".join(phrase_library.get("key_vocabulary", [])[:10])
        
        user_prompt = f"""Polish this text to make it sound like an academic paper. 
Use a formality level of {target_formality} (where 1.0 is maximum formality).

STYLE GUIDELINES:
- Recommended transition words: {transitions}
- Evidence markers: {evidence}
- Academic vocabulary: {key_vocab}
- Hedging phrases: {", ".join(phrase_library.get("hedging_phrases", [])[:5])}
- Emphasis phrases: {", ".join(phrase_library.get("emphasis_phrases", [])[:5])}

ORIGINAL TEXT:
{text}

Please provide ONLY the polished text without any explanation or commentary."""
        
        try:
            polished = self.client.call_api(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.6,
                max_tokens=4000,
            )
            
            return polished.strip()
        
        except Exception as e:
            logger.error(f"Failed to polish text: {e}")
            raise
    
    def _identify_changes(self, original: str, polished: str) -> List[Dict[str, str]]:
        """
        Identify specific changes made during polishing
        
        Args:
            original: Original text
            polished: Polished text
        
        Returns:
            List of changes
        """
        logger.info("Identifying changes...")
        
        system_prompt = """Compare the original and polished text, and identify the key changes made.
Return changes as a JSON array."""
        
        user_prompt = f"""Compare these two versions and list the main changes:

ORIGINAL:
{original[:2000]}

POLISHED:
{polished[:2000]}

Return as JSON array of changes with format:
[
  {{"original": "...", "polished": "...", "reason": "..."}},
  ...
]

List only 5-10 most significant changes."""
        
        try:
            response = self.client.call_api(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=1500,
            )
            
            changes = json.loads(response)
            return changes if isinstance(changes, list) else []
        
        except Exception as e:
            logger.warning(f"Failed to identify changes: {e}")
            return []
    
    def _generate_polishing_report(
        self,
        original: str,
        polished: str,
        changes: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Generate detailed polishing report"""
        
        report = {
            "original_length": len(original),
            "polished_length": len(polished),
            "length_change_percent": round(
                ((len(polished) - len(original)) / len(original) * 100) if len(original) > 0 else 0,
                2
            ),
            "changes_count": len(changes),
            "original_words": len(original.split()),
            "polished_words": len(polished.split()),
            "improvement_score": self._calculate_improvement_score(original, polished),
        }
        
        return report
    
    def _calculate_improvement_score(self, original: str, polished: str) -> float:
        """Calculate improvement score (0-1)"""
        # Simple heuristic: based on vocabulary and formality changes
        # In practice, would use more sophisticated metrics
        
        original_words = set(original.lower().split())
        polished_words = set(polished.lower().split())
        
        # Count new academic words
        academic_indicators = [
            "academic", "research", "study", "analysis", "evidence", 
            "methodology", "significantly", "notably", "propose", "demonstrate"
        ]
        
        new_academic = sum(1 for word in polished_words if any(ind in word for ind in academic_indicators))
        original_academic = sum(1 for word in original_words if any(ind in word for ind in academic_indicators))
        
        improvement = min(1.0, 0.5 + (new_academic - original_academic) * 0.02)
        return round(improvement, 2)
    
    def enhance_academic_tone(self, text: str) -> str:
        """Enhance the academic tone of text"""
        
        system_prompt = """You are an expert academic writer. Enhance the academic tone 
while preserving the original meaning and structure."""
        
        user_prompt = f"""Enhance the academic tone of this text:

{text}

Provide the enhanced version without any explanation."""
        
        try:
            enhanced = self.client.call_api(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                max_tokens=3000,
            )
            
            return enhanced.strip()
        
        except Exception as e:
            logger.error(f"Failed to enhance tone: {e}")
            raise
    
    def restructure_content(
        self,
        text: str,
        target_structure: Optional[List[str]] = None,
    ) -> str:
        """
        Restructure content to match paper structure
        
        Args:
            text: Text to restructure
            target_structure: Target structure (e.g., ['intro', 'method', 'results', 'conclusion'])
        
        Returns:
            Restructured text
        """
        if target_structure is None:
            target_structure = ["introduction", "methodology", "results", "conclusion"]
        
        logger.info(f"Restructuring content to sections: {target_structure}")
        
        system_prompt = f"""You are an expert academic writer. 
Restructure the given text into the following sections: {', '.join(target_structure)}.
Each section should have a clear heading."""
        
        user_prompt = f"""Restructure this text into the following academic sections:
{', '.join(target_structure)}

ORIGINAL TEXT:
{text}

Provide the restructured text with clear section headings."""
        
        try:
            restructured = self.client.call_api(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.6,
                max_tokens=4000,
            )
            
            return restructured.strip()
        
        except Exception as e:
            logger.error(f"Failed to restructure content: {e}")
            raise
    
    def save_polishing_result(
        self,
        result: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Union[str, Path]:
        """Save polishing result to file"""
        
        if output_path is None:
            output_path = Path(self.output_dir) / f"polished_{uuid.uuid4().hex[:8]}.json"
        
        output_path = Path(output_path)
        ensure_dir_exists(output_path.parent)
        
        save_json(result, output_path)
        logger.info(f"Saved polishing result to {output_path}")
        
        return output_path
    
    def export_polished_text(
        self,
        polished_text: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Union[str, Path]:
        """Export polished text to plain text file"""
        
        if output_path is None:
            output_path = Path(self.output_dir) / f"polished_{uuid.uuid4().hex[:8]}.txt"
        
        output_path = Path(output_path)
        ensure_dir_exists(output_path.parent)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(polished_text)
        
        logger.info(f"Exported polished text to {output_path}")
        return output_path
    
    def generate_comparison_report(
        self,
        result: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate a detailed comparison report"""
        
        report = []
        report.append("=" * 80)
        report.append("ACADEMIC TEXT POLISHING REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Statistics
        stats = result.get("statistics", {})
        report.append("STATISTICS:")
        report.append(f"  Original length: {stats.get('original_length', 0)} characters")
        report.append(f"  Polished length: {stats.get('polished_length', 0)} characters")
        report.append(f"  Length change: {stats.get('length_change_percent', 0)}%")
        report.append(f"  Original words: {stats.get('original_words', 0)}")
        report.append(f"  Polished words: {stats.get('polished_words', 0)}")
        report.append(f"  Total changes: {stats.get('changes_count', 0)}")
        report.append(f"  Improvement score: {stats.get('improvement_score', 0)}/1.0")
        report.append("")
        
        # Changes
        changes = result.get("changes", [])
        if changes:
            report.append("KEY CHANGES:")
            for i, change in enumerate(changes[:10], 1):
                report.append(f"\n  {i}. {change.get('reason', 'Improvement')}")
                report.append(f"     Original: {change.get('original', 'N/A')}")
                report.append(f"     Polished: {change.get('polished', 'N/A')}")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_path:
            output_path = Path(output_path)
            ensure_dir_exists(output_path.parent)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Saved comparison report to {output_path}")
        
        return report_text
