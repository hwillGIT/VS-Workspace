#!/usr/bin/env python3
"""
Architecture Intelligence Platform CLI
Command-line interface for deep architecture framework expertise
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.intelligence_engine import ArchitectureIntelligenceEngine, ArchitectureContext, FrameworkType
from frameworks.base_framework import AnalysisDepth


class ArchitectureCLI:
    """CLI interface for Architecture Intelligence Platform"""
    
    def __init__(self):
        self.console = Console()
        self.engine = None
        
    async def initialize_engine(self, config_path: Optional[Path] = None):
        """Initialize the architecture intelligence engine"""
        if not self.engine:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Initializing Architecture Intelligence Engine...", total=None)
                self.engine = ArchitectureIntelligenceEngine(config_path)
                progress.update(task, description="✓ Engine initialized")
    
    def display_banner(self):
        """Display application banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║            Architecture Intelligence Platform                ║
║          Deep Framework Expertise with AI Intelligence       ║
║                                                              ║
║  🏗️  TOGAF • DDD • C4 • Zachman • ArchiMate • +15 More     ║
║  🧠 Intelligent Cross-Framework Pattern Mining              ║
║  🚀 Pragmatic Implementation Accelerators                   ║
╚══════════════════════════════════════════════════════════════╝
        """
        self.console.print(Panel(banner, style="bold blue"))


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """Architecture Intelligence Platform - Deep framework expertise with intelligent pragmatism"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = Path(config) if config else None
    ctx.obj['cli'] = ArchitectureCLI()


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize the architecture intelligence platform"""
    cli_obj = ctx.obj['cli']
    cli_obj.display_banner()
    
    async def _init():
        await cli_obj.initialize_engine(ctx.obj['config'])
        cli_obj.console.print("✓ Architecture Intelligence Platform initialized successfully!", style="bold green")
        
        # Display available frameworks
        frameworks = [f.value for f in FrameworkType]
        table = Table(title="Available Architecture Frameworks")
        table.add_column("Framework", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Expertise Level", style="yellow")
        
        for framework in frameworks[:10]:  # Show first 10
            table.add_row(framework.upper(), "✓ Ready", "Expert")
        
        if len(frameworks) > 10:
            table.add_row("...", f"+ {len(frameworks) - 10} more", "Expert")
        
        cli_obj.console.print(table)
    
    asyncio.run(_init())


@cli.command()
@click.option('--project', '-p', required=True, help='Project name')
@click.option('--domain', '-d', required=True, help='Business domain')
@click.option('--org-size', type=click.Choice(['startup', 'small', 'medium', 'large', 'enterprise']), 
              help='Organization size')
@click.option('--frameworks', '-f', multiple=True, help='Specific frameworks to use')
@click.option('--depth', type=click.Choice(['basic', 'intermediate', 'expert']), default='expert',
              help='Analysis depth level')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', 'output_format', type=click.Choice(['json', 'markdown', 'html']), 
              default='markdown', help='Output format')
@click.pass_context
def analyze(ctx, project, domain, org_size, frameworks, depth, output, output_format):
    """Perform comprehensive architecture analysis"""
    cli_obj = ctx.obj['cli']
    
    async def _analyze():
        await cli_obj.initialize_engine(ctx.obj['config'])
        
        # Create architecture context
        context = ArchitectureContext(
            project_name=project,
            domain=domain,
            organization_size=org_size,
            goals=["modernization", "scalability", "maintainability"],  # Default goals
            quality_attributes=["performance", "security", "usability"]
        )
        
        # Convert framework strings to FrameworkType enum
        selected_frameworks = None
        if frameworks:
            try:
                selected_frameworks = [FrameworkType(f.lower()) for f in frameworks]
            except ValueError as e:
                cli_obj.console.print(f"❌ Invalid framework: {e}", style="bold red")
                return
        
        # Convert depth to enum
        analysis_depth = AnalysisDepth(depth.upper())
        
        cli_obj.console.print(f"🔍 Starting {depth} analysis for {project}...", style="bold blue")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=cli_obj.console
        ) as progress:
            task = progress.add_task("Analyzing architecture...", total=None)
            
            try:
                results = await cli_obj.engine.analyze_architecture(
                    context=context,
                    frameworks=selected_frameworks,
                    depth=depth
                )
                
                progress.update(task, description="✓ Analysis complete")
                
                # Display results summary
                cli_obj._display_analysis_summary(results)
                
                # Export results if requested
                if output:
                    exported_content = await cli_obj.engine.export_analysis(results, output_format)
                    with open(output, 'w') as f:
                        f.write(exported_content)
                    cli_obj.console.print(f"✓ Results exported to {output}", style="bold green")
                
            except Exception as e:
                progress.update(task, description="❌ Analysis failed")
                cli_obj.console.print(f"❌ Analysis failed: {e}", style="bold red")
                raise
    
    asyncio.run(_analyze())


@cli.command()
@click.option('--framework', '-f', type=click.Choice([f.value for f in FrameworkType]), 
              required=True, help='Framework to query')
@click.pass_context
def framework_info(ctx, framework):
    """Get detailed information about a specific framework"""
    cli_obj = ctx.obj['cli']
    
    async def _framework_info():
        await cli_obj.initialize_engine(ctx.obj['config'])
        
        framework_type = FrameworkType(framework.lower())
        capabilities = await cli_obj.engine.get_framework_capabilities(framework_type)
        
        cli_obj.console.print(f"\n📋 {framework.upper()} Framework Information", style="bold blue")
        
        # Create information panel
        info_text = f"""
**Framework:** {capabilities.get('framework', framework)}
**Capabilities:** {len(capabilities.get('capabilities', []))} core capabilities
**Depth Levels:** {', '.join(capabilities.get('depth_levels', []))}
**Artifacts:** {len(capabilities.get('artifacts', []))} standard artifacts
**Patterns:** {len(capabilities.get('patterns', []))} recognized patterns
        """
        
        cli_obj.console.print(Panel(info_text, title=f"{framework.upper()} Overview"))
        
        # Display capabilities
        if capabilities.get('capabilities'):
            table = Table(title="Framework Capabilities")
            table.add_column("Capability", style="cyan")
            table.add_column("Description", style="white")
            
            for capability in capabilities['capabilities'][:10]:  # Show first 10
                # Convert snake_case to title case
                display_name = capability.replace('_', ' ').title()
                table.add_row(display_name, f"Expert-level {display_name.lower()} support")
            
            cli_obj.console.print(table)
    
    asyncio.run(_framework_info())


@cli.command()
@click.option('--context', '-c', type=click.Path(exists=True), help='Context file (YAML/JSON)')
@click.option('--current-patterns', multiple=True, help='Currently implemented patterns')
@click.option('--goals', '-g', multiple=True, help='Architecture goals')
@click.pass_context
def recommend_patterns(ctx, context, current_patterns, goals):
    """Get intelligent pattern recommendations"""
    cli_obj = ctx.obj['cli']
    
    async def _recommend():
        await cli_obj.initialize_engine(ctx.obj['config'])
        
        # Load context from file if provided
        context_data = {}
        if context:
            with open(context, 'r') as f:
                if context.endswith('.yaml') or context.endswith('.yml'):
                    context_data = yaml.safe_load(f)
                else:
                    context_data = json.load(f)
        
        # Use provided goals or defaults
        goal_list = list(goals) if goals else ["scalability", "maintainability", "performance"]
        
        cli_obj.console.print("🧠 Generating intelligent pattern recommendations...", style="bold blue")
        
        try:
            recommendations = await cli_obj.engine.pattern_miner.get_pattern_recommendations(
                context=context_data,
                current_patterns=list(current_patterns),
                goals=goal_list
            )
            
            if recommendations:
                cli_obj._display_pattern_recommendations(recommendations)
            else:
                cli_obj.console.print("ℹ️  No additional patterns recommended for current context", style="yellow")
                
        except Exception as e:
            cli_obj.console.print(f"❌ Recommendation failed: {e}", style="bold red")
    
    asyncio.run(_recommend())


@cli.command()
@click.option('--template', '-t', type=click.Choice(['microservices', 'event-driven', 'ddd', 'clean']),
              help='Quick start template')
@click.option('--project', '-p', required=True, help='Project name')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.pass_context
def quick_start(ctx, template, project, output_dir):
    """Generate quick start architecture template"""
    cli_obj = ctx.obj['cli']
    
    async def _quick_start():
        await cli_obj.initialize_engine(ctx.obj['config'])
        
        cli_obj.console.print(f"🚀 Generating {template} quick start for {project}...", style="bold blue")
        
        # Create output directory
        output_path = Path(output_dir) if output_dir else Path(f"./{project}-{template}-quickstart")
        output_path.mkdir(exist_ok=True)
        
        # Generate template files based on selection
        template_files = cli_obj._generate_quick_start_template(template, project)
        
        for filename, content in template_files.items():
            file_path = output_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)
        
        cli_obj.console.print(f"✓ Quick start template generated in {output_path}", style="bold green")
        
        # Display next steps
        next_steps = cli_obj._get_template_next_steps(template)
        cli_obj.console.print(Panel(next_steps, title="Next Steps", style="yellow"))
    
    asyncio.run(_quick_start())


@cli.command()
@click.option('--library-path', '-l', type=click.Path(exists=True), required=True,
              help='Path to directory containing PDF documents')
@click.option('--context', '-c', type=click.Path(exists=True), 
              help='Context file for relevance assessment')
@click.option('--extract', '-e', is_flag=True, 
              help='Execute knowledge extraction for top documents')
@click.option('--max-documents', '-m', type=int, default=5,
              help='Maximum documents to extract from')
@click.pass_context
def analyze_library(ctx, library_path, context, extract, max_documents):
    """Analyze architecture document library for relevant knowledge"""
    cli_obj = ctx.obj['cli']
    
    async def _analyze_library():
        await cli_obj.initialize_engine(ctx.obj['config'])
        
        # Load context if provided
        context_data = None
        if context:
            with open(context, 'r') as f:
                if context.endswith('.yaml') or context.endswith('.yml'):
                    context_dict = yaml.safe_load(f)
                else:
                    context_dict = json.load(f)
                    
                context_data = ArchitectureContext(
                    project_name=context_dict.get('project_name', 'Unknown'),
                    domain=context_dict.get('domain', 'general'),
                    goals=context_dict.get('goals', []),
                    technical_stack=context_dict.get('technical_stack', [])
                )
        
        cli_obj.console.print(f"📚 Analyzing document library at {library_path}...", style="bold blue")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=cli_obj.console
        ) as progress:
            task = progress.add_task("Analyzing documents...", total=None)
            
            try:
                # Analyze library
                library_analysis = await cli_obj.engine.analyze_architecture_library(
                    Path(library_path),
                    context_data
                )
                
                progress.update(task, description="✓ Library analysis complete")
                
                # Display results
                cli_obj._display_library_analysis(library_analysis)
                
                # Extract knowledge if requested
                if extract and library_analysis.get('extraction_plans'):
                    progress.update(task, description="Extracting knowledge...")
                    
                    extracted = await cli_obj.engine.extract_knowledge_from_documents(
                        library_analysis['extraction_plans'],
                        max_documents
                    )
                    
                    progress.update(task, description="✓ Knowledge extraction complete")
                    cli_obj._display_extracted_knowledge(extracted)
                
            except Exception as e:
                progress.update(task, description="❌ Analysis failed")
                cli_obj.console.print(f"❌ Library analysis failed: {e}", style="bold red")
                raise
    
    asyncio.run(_analyze_library())


@cli.command()
@click.pass_context
def list_frameworks(ctx):
    """List all available architecture frameworks"""
    cli_obj = ctx.obj['cli']
    
    # Group frameworks by category
    framework_categories = {
        "Enterprise Architecture": ["togaf", "zachman", "feaf", "dodaf"],
        "Domain Design": ["ddd"],
        "System Modeling": ["c4_model", "uml", "archimate"],
        "Process Modeling": ["bpmn"],
        "Architecture Styles": ["microservices", "serverless", "reactive", "hexagonal", "clean"],
        "Integration": ["event_driven"],
        "Security": ["stride"],
        "Documentation": ["arc42", "four_plus_one"]
    }
    
    cli_obj.console.print("\n📚 Available Architecture Frameworks\n", style="bold blue")
    
    for category, frameworks in framework_categories.items():
        table = Table(title=category)
        table.add_column("Framework", style="cyan")
        table.add_column("Expertise Level", style="green")
        table.add_column("Industry Adoption", style="yellow")
        
        for framework in frameworks:
            if framework in [f.value for f in FrameworkType]:
                table.add_row(framework.upper(), "Expert", "Widespread")
        
        cli_obj.console.print(table)


@cli.command()
@click.option('--config-file', '-c', type=click.Path(), help='Configuration file to create')
@click.pass_context
def configure(ctx, config_file):
    """Create or update configuration file"""
    cli_obj = ctx.obj['cli']
    
    config_path = Path(config_file) if config_file else Path("./architecture_intelligence_config.yaml")
    
    # Default configuration
    default_config = {
        "intelligence": {
            "min_confidence_threshold": 0.7,
            "max_recommendations": 10,
            "pattern_mining_depth": 3,
            "cross_framework_analysis": True
        },
        "frameworks": {
            "load_all": True,
            "preferred_frameworks": ["togaf", "ddd", "c4"],
            "depth_level": "expert"
        },
        "pragmatic": {
            "prioritize_actionable": True,
            "include_quick_wins": True,
            "generate_implementation_plans": True
        },
        "learning": {
            "enabled": True,
            "feedback_weight": 0.3,
            "pattern_discovery": True
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    cli_obj.console.print(f"✓ Configuration file created: {config_path}", style="bold green")
    cli_obj.console.print(f"Edit this file to customize the platform behavior", style="yellow")


# Helper methods for CLI class
def _display_analysis_summary(self, results: Dict[str, Any]):
    """Display analysis results summary"""
    summary = results.get('summary', {})
    insights = results.get('insights', [])
    
    # Executive summary panel
    summary_text = f"""
**Total Insights:** {summary.get('total_insights', 0)}
**High Priority Items:** {summary.get('high_priority_items', 0)}
**Quick Wins:** {summary.get('quick_wins', 0)}
**Frameworks Used:** {', '.join(results.get('frameworks_used', []))}
    """
    
    self.console.print(Panel(summary_text, title="📊 Analysis Summary", style="blue"))
    
    # Top recommendations table
    if summary.get('top_recommendations'):
        table = Table(title="🎯 Top Recommendations")
        table.add_column("Recommendation", style="cyan")
        table.add_column("Impact", style="red")
        table.add_column("Effort", style="yellow")
        table.add_column("Frameworks", style="green")
        
        for rec in summary['top_recommendations']:
            table.add_row(
                rec.get('title', 'Unknown'),
                rec.get('impact', 'Unknown'),
                rec.get('effort', 'Unknown'),
                ', '.join(rec.get('frameworks', []))
            )
            
        self.console.print(table)
    
    # Next steps
    next_steps = results.get('next_steps', [])
    if next_steps:
        steps_text = "\n".join([f"{i+1}. {step.get('step', 'Unknown step')}" 
                               for i, step in enumerate(next_steps[:5])])
        self.console.print(Panel(steps_text, title="🚀 Next Steps", style="green"))


def _display_pattern_recommendations(self, recommendations: List[Dict[str, Any]]):
    """Display pattern recommendations"""
    self.console.print("\n🎯 Pattern Recommendations\n", style="bold blue")
    
    for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
        panel_content = f"""
**Relevance Score:** {rec['relevance_score']:.2f}

**Description:** {rec['description']}

**Key Benefits:**
{chr(10).join(f"• {benefit}" for benefit in rec['benefits'][:3])}

**Implementation Effort:** {'🔴 High' if rec['implementation_effort'] > 0.7 else '🟡 Medium' if rec['implementation_effort'] > 0.4 else '🟢 Low'}

**Maturity:** {rec['maturity'].title()}
        """
        
        self.console.print(Panel(
            panel_content, 
            title=f"{i}. {rec['pattern_name']}", 
            style="cyan"
        ))


def _generate_quick_start_template(self, template: str, project: str) -> Dict[str, str]:
    """Generate quick start template files"""
    templates = {
        "microservices": {
            "README.md": f"""# {project} - Microservices Architecture

## Overview
This project implements a microservices architecture using domain-driven design principles.

## Architecture Components
- API Gateway
- Service Discovery
- Configuration Management
- Distributed Tracing
- Circuit Breakers

## Getting Started
1. Review the architecture documentation
2. Set up the development environment
3. Deploy the infrastructure components
4. Implement your first microservice

## Architecture Patterns Used
- Microservices Architecture
- API Gateway Pattern
- Database per Service
- Saga Pattern for Distributed Transactions
""",
            "architecture/architecture-vision.md": f"""# Architecture Vision - {project}

## Business Context
Microservices architecture to enable rapid scaling and independent team development.

## Architecture Principles
1. **Service Independence**: Each service owns its data and deployment
2. **API-First Design**: All communication through well-defined APIs
3. **Fault Tolerance**: Services designed to handle failures gracefully
4. **Observability**: Comprehensive monitoring and tracing

## Success Metrics
- Deployment frequency: Daily per service
- Mean time to recovery: < 30 minutes
- Service availability: 99.9%
""",
            "docker-compose.yml": """version: '3.8'
services:
  api-gateway:
    build: ./api-gateway
    ports:
      - "8080:8080"
    depends_on:
      - service-discovery
      
  service-discovery:
    image: consul:latest
    ports:
      - "8500:8500"
      
  user-service:
    build: ./services/user-service
    depends_on:
      - service-discovery
      
  order-service:
    build: ./services/order-service
    depends_on:
      - service-discovery
""",
        },
        
        "event-driven": {
            "README.md": f"""# {project} - Event-Driven Architecture

## Overview
Event-driven architecture enabling real-time processing and loose coupling.

## Key Components
- Event Bus
- Event Store
- Event Processors
- Saga Orchestrator

## Architecture Patterns
- Event-Driven Architecture
- Event Sourcing
- CQRS
- Saga Pattern
""",
            "architecture/event-model.md": f"""# Event Model - {project}

## Domain Events
- UserRegistered
- OrderPlaced  
- PaymentProcessed
- InventoryUpdated

## Event Flow
1. Commands generate domain events
2. Events published to event bus
3. Event processors update read models
4. Sagas coordinate cross-domain workflows
""",
        }
    }
    
    return templates.get(template, {"README.md": f"# {project} Quick Start\n\nTemplate not found."})


def _get_template_next_steps(self, template: str) -> str:
    """Get next steps for template"""
    steps = {
        "microservices": """
1. Review the generated architecture documentation
2. Set up your development environment with Docker
3. Implement your first microservice following DDD principles
4. Configure service discovery and API gateway
5. Add monitoring and observability tools
6. Implement CI/CD pipelines for each service
        """,
        "event-driven": """
1. Define your domain events and event schemas
2. Set up event streaming infrastructure (Kafka/RabbitMQ)
3. Implement event sourcing for your aggregates
4. Create read model projections
5. Implement saga patterns for distributed workflows
6. Add event monitoring and debugging tools
        """
    }
    
    return steps.get(template, "1. Review generated files\n2. Customize for your needs")


def _display_library_analysis(self, analysis: Dict[str, Any]):
    """Display library analysis results"""
    summary = analysis.get('summary', {})
    
    # Summary panel
    summary_text = f"""
**Total Documents:** {analysis.get('total_documents', 0)}
**Essential Documents:** {summary.get('essential_documents', 0)}
**Highly Relevant:** {summary.get('highly_relevant_documents', 0)}
**Relevant:** {summary.get('relevant_documents', 0)}
    """
    
    self.console.print(Panel(summary_text, title="📊 Library Analysis Summary", style="blue"))
    
    # Top documents table
    docs = analysis.get('document_analysis', [])
    if docs:
        table = Table(title="📚 Top Architecture Documents")
        table.add_column("Document", style="cyan", width=50)
        table.add_column("Relevance", style="yellow")
        table.add_column("Score", style="green")
        table.add_column("Frameworks", style="blue")
        
        for doc in docs[:10]:  # Top 10
            relevance_color = {
                "essential": "bold red",
                "highly_relevant": "red",
                "relevant": "yellow",
                "supplementary": "blue",
                "not_relevant": "dim"
            }.get(doc['relevance'], "white")
            
            table.add_row(
                doc['filename'][:50],
                Text(doc['relevance'].upper(), style=relevance_color),
                f"{doc['relevance_score']:.2f}",
                ', '.join(doc['frameworks'][:2]) + ('...' if len(doc['frameworks']) > 2 else '')
            )
        
        self.console.print(table)
    
    # Book recommendations
    recommendations = analysis.get('recommendations', [])
    if recommendations:
        self.console.print("\n📖 Recommended Books\n", style="bold blue")
        
        for i, rec in enumerate(recommendations[:5], 1):
            panel_content = f"""
**Reason:** {rec['reason']}

**Topics:** {', '.join(rec['topics'])}

**Priority:** {rec['priority'].upper()}
            """
            
            self.console.print(Panel(
                panel_content,
                title=f"{i}. {rec['title']}",
                style="cyan"
            ))


def _display_extracted_knowledge(self, extracted: Dict[str, Any]):
    """Display extracted knowledge"""
    self.console.print("\n📚 Extracted Knowledge\n", style="bold blue")
    
    # Summary
    summary = extracted.get('extraction_summary', {})
    if summary:
        table = Table(title="Extraction Summary")
        table.add_column("Document", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Patterns", style="yellow")
        table.add_column("Principles", style="yellow")
        
        for doc, info in summary.items():
            table.add_row(
                doc[:40] + "..." if len(doc) > 40 else doc,
                info.get('status', 'unknown'),
                str(info.get('patterns_extracted', 0)),
                str(info.get('principles_extracted', 0))
            )
        
        self.console.print(table)
    
    # Extracted patterns
    patterns = extracted.get('patterns', [])
    if patterns:
        self.console.print(f"\n🎯 Extracted Patterns ({len(patterns)})\n", style="bold yellow")
        for pattern in patterns[:3]:  # Show first 3
            self.console.print(f"• {pattern.get('name', 'Unknown')}: {pattern.get('context', 'N/A')}")


# Add helper methods to CLI class
ArchitectureCLI._display_analysis_summary = _display_analysis_summary
ArchitectureCLI._display_pattern_recommendations = _display_pattern_recommendations
ArchitectureCLI._generate_quick_start_template = _generate_quick_start_template  
ArchitectureCLI._get_template_next_steps = _get_template_next_steps
ArchitectureCLI._display_library_analysis = _display_library_analysis
ArchitectureCLI._display_extracted_knowledge = _display_extracted_knowledge


if __name__ == '__main__':
    cli()