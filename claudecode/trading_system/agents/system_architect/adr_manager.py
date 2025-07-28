"""
Architecture Decision Records (ADR) Manager

This agent manages Architecture Decision Records for the trading system,
providing automated creation, tracking, and management of architectural decisions.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import yaml
import git

from ...core.base.agent import BaseAgent


@dataclass
class ADRRecord:
    """Architecture Decision Record structure"""
    number: int
    title: str
    status: str  # 'proposed', 'accepted', 'deprecated', 'superseded'
    date: str
    context: str
    decision: str
    consequences: str
    tags: List[str]
    related_adrs: List[int]
    superseded_by: Optional[int]
    supersedes: Optional[int]
    author: str
    reviewers: List[str]
    implementation_status: str  # 'not_started', 'in_progress', 'completed'


@dataclass
class ADRTemplate:
    """Template for creating ADRs"""
    name: str
    sections: List[str]
    prompts: Dict[str, str]
    tags: List[str]


@dataclass
class ADRAnalysis:
    """Analysis of ADR repository"""
    total_records: int
    by_status: Dict[str, int]
    by_tag: Dict[str, int]
    recent_decisions: List[ADRRecord]
    pending_reviews: List[ADRRecord]
    implementation_status: Dict[str, int]
    decision_trends: Dict[str, List[str]]


class ADRManager(BaseAgent):
    """
    Architecture Decision Records Manager
    
    Manages ADRs for the trading system including:
    - Creating new ADRs from templates
    - Tracking decision status and implementation
    - Analyzing decision patterns and trends
    - Maintaining relationships between decisions
    - Generating ADR reports and summaries
    - Automating ADR workflow processes
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ADRManager", config.get('adr_manager', {}))
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.adr_dir = Path(config.get('adr_directory', 'docs/architecture/decisions'))
        self.template_dir = Path(config.get('template_directory', 'docs/architecture/templates'))
        self.auto_number = config.get('auto_number', True)
        self.require_reviews = config.get('require_reviews', True)
        self.default_format = config.get('format', 'markdown')  # markdown, yaml, json
        
        # Initialize directories
        self.adr_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load templates
        self.templates = self._load_templates()
        
        # Initialize git if available
        self.git_repo = self._init_git_repo()
        
        # ADR metadata
        self.adr_index_file = self.adr_dir / 'index.json'
        self.adr_index = self._load_adr_index()
    
    async def create_adr(self, title: str, context: str, decision: str, 
                        consequences: str, template_name: str = 'default',
                        tags: Optional[List[str]] = None,
                        related_adrs: Optional[List[int]] = None) -> str:
        """
        Create a new Architecture Decision Record
        
        Args:
            title: Title of the decision
            context: Context and background
            decision: The decision made
            consequences: Consequences and trade-offs
            template_name: Template to use
            tags: Associated tags
            related_adrs: Related ADR numbers
            
        Returns:
            Path to created ADR file
        """
        self.logger.info(f"Creating new ADR: {title}")
        
        # Get next ADR number
        adr_number = self._get_next_adr_number()
        
        # Create ADR record
        adr = ADRRecord(
            number=adr_number,
            title=title,
            status='proposed',
            date=datetime.now().strftime('%Y-%m-%d'),
            context=context,
            decision=decision,
            consequences=consequences,
            tags=tags or [],
            related_adrs=related_adrs or [],
            superseded_by=None,
            supersedes=None,
            author=self._get_current_user(),
            reviewers=[],
            implementation_status='not_started'
        )
        
        # Generate ADR file
        file_path = await self._write_adr_file(adr, template_name)
        
        # Update index
        self._update_adr_index(adr)
        
        # Commit to git if available
        if self.git_repo:
            self._commit_adr(file_path, f"Add ADR-{adr_number:04d}: {title}")
        
        self.logger.info(f"Created ADR-{adr_number:04d} at {file_path}")
        return str(file_path)
    
    async def update_adr_status(self, adr_number: int, status: str, 
                               reviewer: Optional[str] = None) -> bool:
        """
        Update ADR status
        
        Args:
            adr_number: ADR number to update
            status: New status
            reviewer: Reviewer name (for accepted status)
            
        Returns:
            Success status
        """
        self.logger.info(f"Updating ADR-{adr_number:04d} status to {status}")
        
        adr = self._find_adr_by_number(adr_number)
        if not adr:
            self.logger.error(f"ADR-{adr_number:04d} not found")
            return False
        
        # Validate status transition
        if not self._is_valid_status_transition(adr.status, status):
            self.logger.error(f"Invalid status transition from {adr.status} to {status}")
            return False
        
        # Update status
        adr.status = status
        if reviewer and status == 'accepted':
            if reviewer not in adr.reviewers:
                adr.reviewers.append(reviewer)
        
        # Update file
        file_path = self._get_adr_file_path(adr_number)
        await self._update_adr_file(file_path, adr)
        
        # Update index
        self._update_adr_index(adr)
        
        # Commit changes
        if self.git_repo:
            self._commit_adr(file_path, f"Update ADR-{adr_number:04d} status to {status}")
        
        return True
    
    async def supersede_adr(self, old_adr_number: int, new_adr_number: int) -> bool:
        """
        Mark an ADR as superseded by another
        
        Args:
            old_adr_number: ADR being superseded
            new_adr_number: ADR that supersedes
            
        Returns:
            Success status
        """
        self.logger.info(f"Superseding ADR-{old_adr_number:04d} with ADR-{new_adr_number:04d}")
        
        old_adr = self._find_adr_by_number(old_adr_number)
        new_adr = self._find_adr_by_number(new_adr_number)
        
        if not old_adr or not new_adr:
            self.logger.error("One or both ADRs not found")
            return False
        
        # Update relationships
        old_adr.status = 'superseded'
        old_adr.superseded_by = new_adr_number
        new_adr.supersedes = old_adr_number
        
        # Update files
        old_file_path = self._get_adr_file_path(old_adr_number)
        new_file_path = self._get_adr_file_path(new_adr_number)
        
        await self._update_adr_file(old_file_path, old_adr)
        await self._update_adr_file(new_file_path, new_adr)
        
        # Update index
        self._update_adr_index(old_adr)
        self._update_adr_index(new_adr)
        
        return True
    
    async def analyze_adrs(self) -> Dict[str, Any]:
        """
        Analyze ADR repository for insights
        
        Returns:
            Analysis results
        """
        self.logger.info("Analyzing ADR repository")
        
        all_adrs = self._load_all_adrs()
        
        analysis = ADRAnalysis(
            total_records=len(all_adrs),
            by_status=self._count_by_status(all_adrs),
            by_tag=self._count_by_tag(all_adrs),
            recent_decisions=self._get_recent_adrs(all_adrs, 5),
            pending_reviews=self._get_pending_reviews(all_adrs),
            implementation_status=self._count_by_implementation_status(all_adrs),
            decision_trends=self._analyze_decision_trends(all_adrs)
        )
        
        return {
            'total_records': analysis.total_records,
            'status_distribution': analysis.by_status,
            'tag_distribution': analysis.by_tag,
            'recent_decisions': [self._adr_to_dict(adr) for adr in analysis.recent_decisions],
            'pending_reviews': [self._adr_to_dict(adr) for adr in analysis.pending_reviews],
            'implementation_status': analysis.implementation_status,
            'decision_trends': analysis.decision_trends,
            'recommendations': self._generate_adr_recommendations(analysis)
        }
    
    async def generate_adr_report(self, format: str = 'markdown') -> str:
        """
        Generate comprehensive ADR report
        
        Args:
            format: Output format (markdown, html, json)
            
        Returns:
            Path to generated report
        """
        self.logger.info(f"Generating ADR report in {format} format")
        
        analysis = await self.analyze_adrs()
        
        if format == 'markdown':
            report_content = self._generate_markdown_report(analysis)
            report_path = self.adr_dir / 'adr_report.md'
        elif format == 'html':
            report_content = self._generate_html_report(analysis)
            report_path = self.adr_dir / 'adr_report.html'
        elif format == 'json':
            report_content = json.dumps(analysis, indent=2, default=str)
            report_path = self.adr_dir / 'adr_report.json'
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"ADR report generated: {report_path}")
        return str(report_path)
    
    async def suggest_adr_from_code_changes(self, changed_files: List[str]) -> Optional[Dict[str, Any]]:
        """
        Suggest creating an ADR based on code changes
        
        Args:
            changed_files: List of changed file paths
            
        Returns:
            ADR suggestion or None
        """
        self.logger.info("Analyzing code changes for ADR suggestions")
        
        # Analyze changes for architectural significance
        significant_changes = self._analyze_architectural_changes(changed_files)
        
        if not significant_changes:
            return None
        
        # Generate ADR suggestion
        suggestion = {
            'suggested_title': self._suggest_adr_title(significant_changes),
            'context': self._suggest_context(significant_changes),
            'affected_components': significant_changes['components'],
            'change_type': significant_changes['type'],
            'urgency': significant_changes['urgency'],
            'suggested_tags': significant_changes['tags'],
            'template': self._suggest_template(significant_changes)
        }
        
        return suggestion
    
    async def validate_adr_compliance(self) -> Dict[str, Any]:
        """
        Validate ADR compliance across the repository
        
        Returns:
            Compliance report
        """
        self.logger.info("Validating ADR compliance")
        
        all_adrs = self._load_all_adrs()
        violations = []
        
        for adr in all_adrs:
            adr_violations = self._check_adr_compliance(adr)
            if adr_violations:
                violations.extend(adr_violations)
        
        compliance_score = max(0, 100 - len(violations) * 5)  # 5 points per violation
        
        return {
            'compliance_score': compliance_score,
            'total_violations': len(violations),
            'violations': violations,
            'recommendations': self._generate_compliance_recommendations(violations)
        }
    
    def _load_templates(self) -> Dict[str, ADRTemplate]:
        """Load ADR templates"""
        templates = {}
        
        # Default template
        templates['default'] = ADRTemplate(
            name='default',
            sections=['Title', 'Status', 'Context', 'Decision', 'Consequences'],
            prompts={
                'Context': 'What is the issue that we are seeing that is motivating this decision or change?',
                'Decision': 'What is the change that we are proposing and/or doing?',
                'Consequences': 'What becomes easier or more difficult to do because of this change?'
            },
            tags=['general']
        )
        
        # Technical architecture template
        templates['technical'] = ADRTemplate(
            name='technical',
            sections=['Title', 'Status', 'Context', 'Decision', 'Consequences', 'Alternatives', 'Implementation'],
            prompts={
                'Context': 'What technical challenge or requirement is driving this decision?',
                'Decision': 'What technical solution are we adopting?',
                'Consequences': 'What are the technical implications and trade-offs?',
                'Alternatives': 'What other options were considered?',
                'Implementation': 'How will this be implemented and by when?'
            },
            tags=['technical', 'architecture']
        )
        
        # Security template
        templates['security'] = ADRTemplate(
            name='security',
            sections=['Title', 'Status', 'Context', 'Decision', 'Consequences', 'Security_Implications', 'Compliance'],
            prompts={
                'Context': 'What security concern or requirement is being addressed?',
                'Decision': 'What security measures or changes are being implemented?',
                'Consequences': 'How does this impact security posture and operations?',
                'Security_Implications': 'What are the security implications and risks?',
                'Compliance': 'How does this affect regulatory compliance?'
            },
            tags=['security', 'compliance']
        )
        
        # Load custom templates from files
        if self.template_dir.exists():
            for template_file in self.template_dir.glob('*.yaml'):
                try:
                    with open(template_file, 'r') as f:
                        template_data = yaml.safe_load(f)
                        template = ADRTemplate(**template_data)
                        templates[template.name] = template
                except Exception as e:
                    self.logger.warning(f"Could not load template {template_file}: {e}")
        
        return templates
    
    def _init_git_repo(self) -> Optional[git.Repo]:
        """Initialize git repository if available"""
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo
        except (git.exc.InvalidGitRepositoryError, git.exc.GitCommandError):
            self.logger.warning("Git repository not found or not accessible")
            return None
    
    def _load_adr_index(self) -> Dict[int, Dict[str, Any]]:
        """Load ADR index"""
        if self.adr_index_file.exists():
            try:
                with open(self.adr_index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load ADR index: {e}")
        
        return {}
    
    def _save_adr_index(self) -> None:
        """Save ADR index"""
        with open(self.adr_index_file, 'w') as f:
            json.dump(self.adr_index, f, indent=2)
    
    def _get_next_adr_number(self) -> int:
        """Get next available ADR number"""
        if not self.auto_number:
            return len(self.adr_index) + 1
        
        existing_numbers = set(self.adr_index.keys())
        if isinstance(list(existing_numbers)[0] if existing_numbers else 0, str):
            existing_numbers = {int(k) for k in existing_numbers}
        
        return max(existing_numbers, default=0) + 1
    
    def _get_current_user(self) -> str:
        """Get current user for authorship"""
        if self.git_repo:
            try:
                config = self.git_repo.config_reader()
                return f"{config.get_value('user', 'name')} <{config.get_value('user', 'email')}>"
            except Exception:
                pass
        
        return "System Architect Agent"
    
    async def _write_adr_file(self, adr: ADRRecord, template_name: str) -> Path:
        """Write ADR to file"""
        template = self.templates.get(template_name, self.templates['default'])
        
        if self.default_format == 'markdown':
            content = self._format_adr_markdown(adr, template)
            filename = f"ADR-{adr.number:04d}-{self._slugify(adr.title)}.md"
        elif self.default_format == 'yaml':
            content = self._format_adr_yaml(adr)
            filename = f"ADR-{adr.number:04d}-{self._slugify(adr.title)}.yaml"
        elif self.default_format == 'json':
            content = self._format_adr_json(adr)
            filename = f"ADR-{adr.number:04d}-{self._slugify(adr.title)}.json"
        else:
            raise ValueError(f"Unsupported format: {self.default_format}")
        
        file_path = self.adr_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return file_path
    
    def _format_adr_markdown(self, adr: ADRRecord, template: ADRTemplate) -> str:
        """Format ADR as Markdown"""
        content = f"""# ADR-{adr.number:04d}: {adr.title}

**Status:** {adr.status}  
**Date:** {adr.date}  
**Author:** {adr.author}  
**Tags:** {', '.join(adr.tags)}  
**Implementation Status:** {adr.implementation_status}

"""
        
        if adr.supersedes:
            content += f"**Supersedes:** ADR-{adr.supersedes:04d}  \n"
        
        if adr.superseded_by:
            content += f"**Superseded by:** ADR-{adr.superseded_by:04d}  \n"
        
        if adr.related_adrs:
            content += f"**Related ADRs:** {', '.join(f'ADR-{num:04d}' for num in adr.related_adrs)}  \n"
        
        content += "\n"
        
        # Add sections based on template
        if 'Context' in template.sections:
            content += f"""## Context

{adr.context}

"""
        
        if 'Decision' in template.sections:
            content += f"""## Decision

{adr.decision}

"""
        
        if 'Consequences' in template.sections:
            content += f"""## Consequences

{adr.consequences}

"""
        
        # Add reviewers if any
        if adr.reviewers:
            content += f"""## Reviewers

{chr(10).join(f"- {reviewer}" for reviewer in adr.reviewers)}

"""
        
        content += "---\n*This ADR was managed by the ADR Manager Agent*\n"
        
        return content
    
    def _format_adr_yaml(self, adr: ADRRecord) -> str:
        """Format ADR as YAML"""
        adr_dict = self._adr_to_dict(adr)
        return yaml.dump(adr_dict, default_flow_style=False)
    
    def _format_adr_json(self, adr: ADRRecord) -> str:
        """Format ADR as JSON"""
        adr_dict = self._adr_to_dict(adr)
        return json.dumps(adr_dict, indent=2)
    
    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug"""
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug.strip('-')
    
    def _update_adr_index(self, adr: ADRRecord) -> None:
        """Update ADR index with record"""
        self.adr_index[str(adr.number)] = {
            'title': adr.title,
            'status': adr.status,
            'date': adr.date,
            'author': adr.author,
            'tags': adr.tags,
            'implementation_status': adr.implementation_status
        }
        self._save_adr_index()
    
    def _commit_adr(self, file_path: Path, message: str) -> None:
        """Commit ADR to git"""
        try:
            self.git_repo.index.add([str(file_path), str(self.adr_index_file)])
            self.git_repo.index.commit(message)
            self.logger.info(f"Committed ADR to git: {message}")
        except Exception as e:
            self.logger.warning(f"Could not commit to git: {e}")
    
    def _find_adr_by_number(self, number: int) -> Optional[ADRRecord]:
        """Find ADR by number"""
        index_entry = self.adr_index.get(str(number))
        if not index_entry:
            return None
        
        # Load full ADR from file
        file_path = self._get_adr_file_path(number)
        if not file_path.exists():
            return None
        
        return self._load_adr_from_file(file_path)
    
    def _get_adr_file_path(self, number: int) -> Path:
        """Get file path for ADR number"""
        # Find file with ADR number prefix
        pattern = f"ADR-{number:04d}-*.{self.default_format.replace('markdown', 'md')}"
        matches = list(self.adr_dir.glob(pattern))
        
        if matches:
            return matches[0]
        
        # Fallback to expected filename
        title_slug = self.adr_index.get(str(number), {}).get('title', 'unknown')
        filename = f"ADR-{number:04d}-{self._slugify(title_slug)}.{self.default_format.replace('markdown', 'md')}"
        return self.adr_dir / filename
    
    def _load_adr_from_file(self, file_path: Path) -> Optional[ADRRecord]:
        """Load ADR from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_path.suffix == '.md':
                return self._parse_markdown_adr(content)
            elif file_path.suffix == '.yaml':
                data = yaml.safe_load(content)
                return ADRRecord(**data)
            elif file_path.suffix == '.json':
                data = json.loads(content)
                return ADRRecord(**data)
        except Exception as e:
            self.logger.error(f"Could not load ADR from {file_path}: {e}")
        
        return None
    
    def _parse_markdown_adr(self, content: str) -> Optional[ADRRecord]:
        """Parse ADR from Markdown content"""
        # Extract metadata from header
        lines = content.split('\n')
        metadata = {}
        
        for line in lines:
            if line.startswith('**') and ':**' in line:
                key, value = line.split(':**', 1)
                key = key.strip('*').lower().replace(' ', '_')
                value = value.strip()
                metadata[key] = value
        
        # Extract number from title
        title_line = next((line for line in lines if line.startswith('# ADR-')), '')
        if title_line:
            match = re.match(r'# ADR-(\d+): (.+)', title_line)
            if match:
                number = int(match.group(1))
                title = match.group(2)
            else:
                return None
        else:
            return None
        
        # Extract sections
        sections = {}
        current_section = None
        section_content = []
        
        for line in lines:
            if line.startswith('## '):
                if current_section:
                    sections[current_section] = '\n'.join(section_content).strip()
                current_section = line[3:].strip().lower()
                section_content = []
            elif current_section:
                section_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(section_content).strip()
        
        # Create ADR record
        return ADRRecord(
            number=number,
            title=title,
            status=metadata.get('status', 'unknown'),
            date=metadata.get('date', ''),
            context=sections.get('context', ''),
            decision=sections.get('decision', ''),
            consequences=sections.get('consequences', ''),
            tags=metadata.get('tags', '').split(', ') if metadata.get('tags') else [],
            related_adrs=[],  # Would need more complex parsing
            superseded_by=None,
            supersedes=None,
            author=metadata.get('author', ''),
            reviewers=[],  # Would need parsing from reviewers section
            implementation_status=metadata.get('implementation_status', 'unknown')
        )
    
    async def _update_adr_file(self, file_path: Path, adr: ADRRecord) -> None:
        """Update ADR file with new data"""
        template = self.templates.get('default')
        
        if file_path.suffix == '.md':
            content = self._format_adr_markdown(adr, template)
        elif file_path.suffix == '.yaml':
            content = self._format_adr_yaml(adr)
        elif file_path.suffix == '.json':
            content = self._format_adr_json(adr)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _is_valid_status_transition(self, current: str, new: str) -> bool:
        """Check if status transition is valid"""
        valid_transitions = {
            'proposed': ['accepted', 'rejected', 'deprecated'],
            'accepted': ['deprecated', 'superseded'],
            'rejected': ['proposed'],  # Can be reconsidered
            'deprecated': [],  # Terminal state
            'superseded': []   # Terminal state
        }
        
        return new in valid_transitions.get(current, [])
    
    def _load_all_adrs(self) -> List[ADRRecord]:
        """Load all ADRs from files"""
        adrs = []
        
        for adr_file in self.adr_dir.glob('ADR-*.md'):
            adr = self._load_adr_from_file(adr_file)
            if adr:
                adrs.append(adr)
        
        return sorted(adrs, key=lambda x: x.number)
    
    def _count_by_status(self, adrs: List[ADRRecord]) -> Dict[str, int]:
        """Count ADRs by status"""
        counts = {}
        for adr in adrs:
            counts[adr.status] = counts.get(adr.status, 0) + 1
        return counts
    
    def _count_by_tag(self, adrs: List[ADRRecord]) -> Dict[str, int]:
        """Count ADRs by tag"""
        counts = {}
        for adr in adrs:
            for tag in adr.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return counts
    
    def _count_by_implementation_status(self, adrs: List[ADRRecord]) -> Dict[str, int]:
        """Count ADRs by implementation status"""
        counts = {}
        for adr in adrs:
            status = adr.implementation_status
            counts[status] = counts.get(status, 0) + 1
        return counts
    
    def _get_recent_adrs(self, adrs: List[ADRRecord], limit: int) -> List[ADRRecord]:
        """Get most recent ADRs"""
        return sorted(adrs, key=lambda x: x.date, reverse=True)[:limit]
    
    def _get_pending_reviews(self, adrs: List[ADRRecord]) -> List[ADRRecord]:
        """Get ADRs pending review"""
        return [adr for adr in adrs if adr.status == 'proposed']
    
    def _analyze_decision_trends(self, adrs: List[ADRRecord]) -> Dict[str, List[str]]:
        """Analyze decision trends over time"""
        trends = {
            'monthly_decisions': [],
            'popular_tags': [],
            'status_trends': []
        }
        
        # Monthly decision count
        monthly_counts = {}
        for adr in adrs:
            month = adr.date[:7]  # YYYY-MM
            monthly_counts[month] = monthly_counts.get(month, 0) + 1
        
        trends['monthly_decisions'] = [f"{month}: {count}" for month, count in monthly_counts.items()]
        
        # Popular tags
        tag_counts = self._count_by_tag(adrs)
        popular_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        trends['popular_tags'] = [f"{tag}: {count}" for tag, count in popular_tags]
        
        return trends
    
    def _generate_adr_recommendations(self, analysis: ADRAnalysis) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if analysis.pending_reviews:
            recommendations.append(f"Review {len(analysis.pending_reviews)} pending ADR(s)")
        
        not_started = analysis.implementation_status.get('not_started', 0)
        if not_started > 0:
            recommendations.append(f"Plan implementation for {not_started} accepted ADR(s)")
        
        if analysis.total_records < 5:
            recommendations.append("Consider documenting more architectural decisions")
        
        return recommendations
    
    def _generate_markdown_report(self, analysis: Dict[str, Any]) -> str:
        """Generate Markdown ADR report"""
        return f"""# Architecture Decision Records Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total ADRs:** {analysis['total_records']}
- **Status Distribution:** {', '.join(f"{k}: {v}" for k, v in analysis['status_distribution'].items())}
- **Pending Reviews:** {len(analysis['pending_reviews'])}

## Recent Decisions

{chr(10).join(f"- **ADR-{adr['number']:04d}**: {adr['title']} ({adr['status']})" for adr in analysis['recent_decisions'])}

## Tag Distribution

{chr(10).join(f"- **{tag}**: {count}" for tag, count in analysis['tag_distribution'].items())}

## Implementation Status

{chr(10).join(f"- **{status.replace('_', ' ').title()}**: {count}" for status, count in analysis['implementation_status'].items())}

## Recommendations

{chr(10).join(f"- {rec}" for rec in analysis['recommendations'])}

---
*Report generated by ADR Manager Agent*
"""
    
    def _generate_html_report(self, analysis: Dict[str, Any]) -> str:
        """Generate HTML ADR report"""
        # Simple HTML report (could be enhanced with templates)
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>ADR Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Architecture Decision Records Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Summary</h2>
    <ul>
        <li><strong>Total ADRs:</strong> {analysis['total_records']}</li>
        <li><strong>Pending Reviews:</strong> {len(analysis['pending_reviews'])}</li>
    </ul>
    
    <h2>Status Distribution</h2>
    <table>
        <tr><th>Status</th><th>Count</th></tr>
        {chr(10).join(f"<tr><td>{status}</td><td>{count}</td></tr>" for status, count in analysis['status_distribution'].items())}
    </table>
    
    <h2>Recent Decisions</h2>
    <ul>
        {chr(10).join(f"<li><strong>ADR-{adr['number']:04d}:</strong> {adr['title']} ({adr['status']})</li>" for adr in analysis['recent_decisions'])}
    </ul>
</body>
</html>"""
    
    def _analyze_architectural_changes(self, changed_files: List[str]) -> Optional[Dict[str, Any]]:
        """Analyze code changes for architectural significance"""
        # Patterns that indicate architectural changes
        architectural_patterns = {
            'new_service': r'.*/(services?|apps?)/[^/]+/__init__\.py$',
            'api_change': r'.*/api/.*\.py$',
            'database_schema': r'.*/migrations/.*\.py$|.*models\.py$',
            'config_change': r'.*config.*\.py$|.*settings.*\.py$',
            'dockerfile': r'Dockerfile|docker-compose\.ya?ml$',
            'dependencies': r'requirements\.txt|pyproject\.toml|setup\.py$'
        }
        
        significant_changes = {
            'type': 'general',
            'components': [],
            'urgency': 'medium',
            'tags': []
        }
        
        for file_path in changed_files:
            for change_type, pattern in architectural_patterns.items():
                if re.search(pattern, file_path):
                    significant_changes['type'] = change_type
                    significant_changes['components'].append(file_path)
                    
                    if change_type in ['api_change', 'database_schema']:
                        significant_changes['urgency'] = 'high'
                        significant_changes['tags'].append('breaking_change')
                    elif change_type in ['new_service', 'dockerfile']:
                        significant_changes['urgency'] = 'high'
                        significant_changes['tags'].append('infrastructure')
        
        # Return None if no significant changes detected
        if not significant_changes['components']:
            return None
        
        return significant_changes
    
    def _suggest_adr_title(self, changes: Dict[str, Any]) -> str:
        """Suggest ADR title based on changes"""
        change_type = changes['type']
        components = changes['components']
        
        if change_type == 'new_service':
            return f"Add new service: {Path(components[0]).parts[-2]}"
        elif change_type == 'api_change':
            return f"API changes in {Path(components[0]).stem}"
        elif change_type == 'database_schema':
            return "Database schema changes"
        elif change_type == 'config_change':
            return "Configuration changes"
        elif change_type == 'dockerfile':
            return "Container and deployment changes"
        elif change_type == 'dependencies':
            return "Dependency updates"
        else:
            return "Architectural changes"
    
    def _suggest_context(self, changes: Dict[str, Any]) -> str:
        """Suggest context for ADR based on changes"""
        return f"Code changes detected in {len(changes['components'])} file(s) related to {changes['type']}. These changes may have architectural implications that should be documented."
    
    def _suggest_template(self, changes: Dict[str, Any]) -> str:
        """Suggest appropriate template based on changes"""
        if 'security' in changes.get('tags', []):
            return 'security'
        elif changes['type'] in ['new_service', 'api_change', 'dockerfile']:
            return 'technical'
        else:
            return 'default'
    
    def _check_adr_compliance(self, adr: ADRRecord) -> List[Dict[str, str]]:
        """Check ADR compliance rules"""
        violations = []
        
        # Check required fields
        if not adr.context.strip():
            violations.append({
                'adr': f"ADR-{adr.number:04d}",
                'type': 'missing_content',
                'description': 'Context section is empty'
            })
        
        if not adr.decision.strip():
            violations.append({
                'adr': f"ADR-{adr.number:04d}",
                'type': 'missing_content',
                'description': 'Decision section is empty'
            })
        
        if not adr.consequences.strip():
            violations.append({
                'adr': f"ADR-{adr.number:04d}",
                'type': 'missing_content',
                'description': 'Consequences section is empty'
            })
        
        # Check if accepted ADRs have reviewers
        if adr.status == 'accepted' and not adr.reviewers:
            violations.append({
                'adr': f"ADR-{adr.number:04d}",
                'type': 'missing_review',
                'description': 'Accepted ADR has no recorded reviewers'
            })
        
        # Check if ADR has been reviewed in reasonable time
        if adr.status == 'proposed':
            date_obj = datetime.strptime(adr.date, '%Y-%m-%d')
            days_old = (datetime.now() - date_obj).days
            if days_old > 30:
                violations.append({
                    'adr': f"ADR-{adr.number:04d}",
                    'type': 'stale_proposal',
                    'description': f'Proposed ADR is {days_old} days old without review'
                })
        
        return violations
    
    def _generate_compliance_recommendations(self, violations: List[Dict[str, str]]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        violation_types = [v['type'] for v in violations]
        
        if 'missing_content' in violation_types:
            recommendations.append("Complete missing content in ADR sections")
        
        if 'missing_review' in violation_types:
            recommendations.append("Add reviewer information to accepted ADRs")
        
        if 'stale_proposal' in violation_types:
            recommendations.append("Review old proposed ADRs and update their status")
        
        return recommendations
    
    def _adr_to_dict(self, adr: ADRRecord) -> Dict[str, Any]:
        """Convert ADR record to dictionary"""
        return {
            'number': adr.number,
            'title': adr.title,
            'status': adr.status,
            'date': adr.date,
            'context': adr.context,
            'decision': adr.decision,
            'consequences': adr.consequences,
            'tags': adr.tags,
            'related_adrs': adr.related_adrs,
            'superseded_by': adr.superseded_by,
            'supersedes': adr.supersedes,
            'author': adr.author,
            'reviewers': adr.reviewers,
            'implementation_status': adr.implementation_status
        }