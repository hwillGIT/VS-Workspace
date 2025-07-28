#!/usr/bin/env python3
"""
Plan Generator Tool
CLI tool for generating planning configurations and executing parallel planning workflows.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from core.parallel_planner import (
    ParallelPlanner, 
    PlanningContext,
    create_software_development_planner,
    create_trading_system_planner
)


class PlanGenerator:
    """Main plan generation utility."""
    
    def __init__(self):
        self.templates_dir = Path(__file__).parent.parent / "templates" / "project_templates"
        self.specifications_dir = Path(__file__).parent.parent / "specifications"
        self.examples_dir = Path(__file__).parent.parent / "examples"
    
    def list_available_templates(self) -> List[str]:
        """List all available project templates."""
        templates = []
        for template_file in self.templates_dir.glob("*.json"):
            templates.append(template_file.stem)
        return templates
    
    def list_available_specifications(self) -> List[str]:
        """List all available planning specifications."""
        specs = []
        for spec_file in self.specifications_dir.glob("*_spec.md"):
            specs.append(spec_file.stem.replace("_spec", ""))
        return specs
    
    def generate_config_from_template(self, template_name: str, output_path: Path, customizations: Dict[str, Any] = None) -> bool:
        """Generate a planning configuration from a template."""
        template_path = self.templates_dir / f"{template_name}.json"
        
        if not template_path.exists():
            print(f"❌ Template not found: {template_name}")
            print(f"Available templates: {', '.join(self.list_available_templates())}")
            return False
        
        try:
            with open(template_path, 'r') as f:
                template_config = json.load(f)
            
            # Apply customizations if provided
            if customizations:
                template_config = self._apply_customizations(template_config, customizations)
            
            # Generate timestamp and metadata
            generated_config = {
                "generated_at": datetime.now().isoformat(),
                "template_source": template_name,
                "generator_version": "1.0",
                **template_config
            }
            
            # Write to output file
            with open(output_path, 'w') as f:
                json.dump(generated_config, f, indent=2)
            
            print(f"✅ Configuration generated: {output_path}")
            print(f"📋 Template: {template_config.get('template_name', template_name)}")
            print(f"🎯 Project Type: {template_config.get('project_type', 'unknown')}")
            print(f"📝 Perspectives: {len(template_config.get('perspectives', []))}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating configuration: {e}")
            return False
    
    def create_custom_template(self, project_type: str, perspectives: List[str], output_path: Path) -> bool:
        """Create a custom template with specified perspectives."""
        
        # Define available perspectives
        available_perspectives = {
            "technical": {
                "name": "Technical Architecture",
                "agent_type": "code-architect",
                "focus_areas": ["architecture", "technology_stack", "implementation"]
            },
            "security": {
                "name": "Security & Compliance",
                "agent_type": "code-reviewer", 
                "focus_areas": ["security_controls", "compliance", "threat_modeling"]
            },
            "performance": {
                "name": "Performance & Scalability",
                "agent_type": "general-purpose",
                "focus_areas": ["scalability", "performance", "optimization"]
            },
            "operational": {
                "name": "Operations & DevOps",
                "agent_type": "code-architect",
                "focus_areas": ["deployment", "monitoring", "maintenance"]
            },
            "user_experience": {
                "name": "User Experience",
                "agent_type": "general-purpose",
                "focus_areas": ["usability", "accessibility", "interface_design"]
            },
            "business": {
                "name": "Business & Requirements",
                "agent_type": "general-purpose",
                "focus_areas": ["requirements", "stakeholders", "success_criteria"]
            }
        }
        
        # Validate perspectives
        invalid_perspectives = set(perspectives) - set(available_perspectives.keys())
        if invalid_perspectives:
            print(f"❌ Invalid perspectives: {', '.join(invalid_perspectives)}")
            print(f"Available perspectives: {', '.join(available_perspectives.keys())}")
            return False
        
        # Build custom template
        custom_template = {
            "template_name": f"Custom {project_type.title()} Template",
            "template_version": "1.0",
            "description": f"Custom planning template for {project_type} projects",
            "project_type": project_type,
            "planning_configuration": {
                "max_concurrent_agents": min(len(perspectives), 5),
                "default_timeout": 450,
                "synthesis_strategy": "balanced",
                "conflict_resolution": "priority_based"
            },
            "perspectives": []
        }
        
        # Add selected perspectives
        for i, perspective_id in enumerate(perspectives):
            perspective_config = available_perspectives[perspective_id]
            
            custom_template["perspectives"].append({
                "perspective_id": perspective_id,
                "name": perspective_config["name"],
                "agent_type": perspective_config["agent_type"],
                "focus_areas": perspective_config["focus_areas"],
                "priority": 1 if i < 3 else 2,  # First 3 are high priority
                "timeout": 450,
                "constraints": {}
            })
        
        # Add basic synthesis configuration
        custom_template["synthesis_configuration"] = {
            "synthesis_strategy": "balanced",
            "conflict_resolution_rules": [
                {
                    "rule": "security_priority",
                    "description": "Security requirements take precedence"
                },
                {
                    "rule": "performance_within_constraints", 
                    "description": "Performance optimizations within resource constraints"
                }
            ]
        }
        
        # Add validation criteria
        custom_template["validation_criteria"] = [
            {
                "category": "completeness",
                "criteria": ["all_perspectives_addressed", "implementation_approach_defined"]
            },
            {
                "category": "consistency", 
                "criteria": ["no_conflicting_decisions", "aligned_timelines"]
            }
        ]
        
        try:
            with open(output_path, 'w') as f:
                json.dump(custom_template, f, indent=2)
            
            print(f"✅ Custom template created: {output_path}")
            print(f"🎯 Project Type: {project_type}")
            print(f"📝 Perspectives: {', '.join(perspectives)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating custom template: {e}")
            return False
    
    async def execute_planning_workflow(self, config_path: Path, problem_description: str, requirements: Dict[str, Any] = None) -> bool:
        """Execute a complete planning workflow from configuration."""
        
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return False
        
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create planner
            planner = ParallelPlanner(str(config_path))
            
            # Set planning context
            context = PlanningContext(
                project_type=config.get("project_type", "unknown"),
                problem_description=problem_description,
                requirements=requirements or {},
                constraints={},
                stakeholders=[]
            )
            planner.set_planning_context(context)
            
            print(f"🚀 Starting parallel planning workflow...")
            print(f"📋 Project Type: {context.project_type}")
            print(f"🎯 Problem: {problem_description[:100]}...")
            print(f"👥 Perspectives: {len(planner.perspectives)}")
            
            # Progress callback
            def progress_callback(result, completed, total):
                status_emoji = "✅" if result.status == "success" else "❌"
                print(f"{status_emoji} [{completed}/{total}] {result.task_id} completed in {result.execution_time:.1f}s")
            
            # Execute planning
            results = await planner.execute_parallel_planning(progress_callback)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = config_path.parent / f"planning_results_{timestamp}.json"
            planner.save_results(results_path)
            
            print(f"\n🎉 Planning workflow completed!")
            print(f"📊 Results saved to: {results_path}")
            
            # Print summary
            summary = results.get("execution_summary", {})
            print(f"\n📈 Execution Summary:")
            print(f"   Total perspectives: {summary.get('total_perspectives', 0)}")
            print(f"   Successful: {summary.get('successful_perspectives', 0)}")
            print(f"   Conflicts detected: {summary.get('conflicts_detected', 0)}")
            print(f"   Plan generated: {'Yes' if summary.get('plan_generated') else 'No'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error executing planning workflow: {e}")
            return False
    
    def _apply_customizations(self, template_config: Dict[str, Any], customizations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply customizations to template configuration."""
        
        # Simple deep merge of customizations
        def merge_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
            result = base.copy()
            for key, value in updates.items():
                if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result
        
        return merge_dict(template_config, customizations)
    
    def validate_configuration(self, config_path: Path) -> bool:
        """Validate a planning configuration file."""
        
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return False
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check required fields
            required_fields = ["project_type", "perspectives"]
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                print(f"❌ Missing required fields: {', '.join(missing_fields)}")
                return False
            
            # Validate perspectives
            perspectives = config.get("perspectives", [])
            if not perspectives:
                print("❌ No perspectives defined")
                return False
            
            for i, perspective in enumerate(perspectives):
                required_perspective_fields = ["perspective_id", "name", "agent_type", "focus_areas"]
                missing_perspective_fields = [field for field in required_perspective_fields if field not in perspective]
                
                if missing_perspective_fields:
                    print(f"❌ Perspective {i+1} missing fields: {', '.join(missing_perspective_fields)}")
                    return False
            
            print(f"✅ Configuration validation passed")
            print(f"📋 Project Type: {config['project_type']}")
            print(f"👥 Perspectives: {len(perspectives)}")
            
            for perspective in perspectives:
                print(f"   - {perspective['name']} ({perspective['agent_type']})")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
            return False
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False


async def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Plan Generator Tool for Parallel Planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available templates
  python plan_generator.py list-templates

  # Generate config from template
  python plan_generator.py generate --template trading_system --output my_config.json

  # Create custom template
  python plan_generator.py custom --project-type web_app --perspectives technical security performance --output custom.json

  # Execute planning workflow
  python plan_generator.py plan --config my_config.json --problem "Build a high-performance trading system"

  # Validate configuration
  python plan_generator.py validate --config my_config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List templates command
    list_parser = subparsers.add_parser('list-templates', help='List available project templates')
    
    # List specifications command
    specs_parser = subparsers.add_parser('list-specs', help='List available planning specifications')
    
    # Generate config command
    generate_parser = subparsers.add_parser('generate', help='Generate configuration from template')
    generate_parser.add_argument('--template', required=True, help='Template name')
    generate_parser.add_argument('--output', required=True, help='Output configuration file')
    generate_parser.add_argument('--customize', help='JSON file with customizations')
    
    # Custom template command
    custom_parser = subparsers.add_parser('custom', help='Create custom template')
    custom_parser.add_argument('--project-type', required=True, help='Project type')
    custom_parser.add_argument('--perspectives', nargs='+', required=True, help='Perspectives to include')
    custom_parser.add_argument('--output', required=True, help='Output template file')
    
    # Execute planning command
    plan_parser = subparsers.add_parser('plan', help='Execute planning workflow')
    plan_parser.add_argument('--config', required=True, help='Planning configuration file')
    plan_parser.add_argument('--problem', required=True, help='Problem description')
    plan_parser.add_argument('--requirements', help='JSON file with requirements')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
    validate_parser.add_argument('--config', required=True, help='Configuration file to validate')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    generator = PlanGenerator()
    
    try:
        if args.command == 'list-templates':
            templates = generator.list_available_templates()
            print("📋 Available Project Templates:")
            for template in templates:
                print(f"   - {template}")
            
        elif args.command == 'list-specs':
            specs = generator.list_available_specifications()
            print("📋 Available Planning Specifications:")
            for spec in specs:
                print(f"   - {spec}")
        
        elif args.command == 'generate':
            customizations = {}
            if args.customize:
                with open(args.customize, 'r') as f:
                    customizations = json.load(f)
            
            success = generator.generate_config_from_template(
                args.template, 
                Path(args.output), 
                customizations
            )
            sys.exit(0 if success else 1)
        
        elif args.command == 'custom':
            success = generator.create_custom_template(
                args.project_type,
                args.perspectives,
                Path(args.output)
            )
            sys.exit(0 if success else 1)
        
        elif args.command == 'plan':
            requirements = {}
            if args.requirements:
                with open(args.requirements, 'r') as f:
                    requirements = json.load(f)
            
            success = await generator.execute_planning_workflow(
                Path(args.config),
                args.problem,
                requirements
            )
            sys.exit(0 if success else 1)
        
        elif args.command == 'validate':
            success = generator.validate_configuration(Path(args.config))
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())