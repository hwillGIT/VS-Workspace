"""
System Architect Suite - Usage Examples

This file demonstrates how to use the complete System Architect suite
for analyzing your trading system codebase.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

# Example usage patterns for each component


async def example_basic_usage():
    """Basic usage example - analyzing a project"""
    print("=== BASIC USAGE EXAMPLE ===")
    
    # For demonstration, we'll analyze the current directory
    project_path = str(Path(__file__).parent)
    
    try:
        # Import the master coordinator (in real usage)
        # from trading_system.agents.system_architect.master_coordinator import analyze_project
        
        print(f"Analyzing project at: {project_path}")
        
        # Simple analysis using the convenience function
        # results = await analyze_project(project_path)
        
        # For this example, we'll simulate the results structure
        results = {
            'session_id': f'demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'project_path': project_path,
            'health_report': {
                'overall_score': 75.5,
                'health_status': 'good',
                'key_strengths': [
                    'Good code organization',
                    'Comprehensive documentation',
                    'Well-defined interfaces'
                ],
                'critical_issues': [
                    'High complexity in main processing functions',
                    'Potential security vulnerabilities detected'
                ]
            },
            'insights': [
                {
                    'category': 'complexity',
                    'severity': 'warning',
                    'title': 'High Complexity Functions',
                    'description': 'Several functions exceed complexity threshold',
                    'recommendations': [
                        'Refactor complex functions using Extract Method pattern',
                        'Consider breaking large classes into smaller components'
                    ]
                },
                {
                    'category': 'security',
                    'severity': 'critical',
                    'title': 'Security Vulnerabilities',
                    'description': 'Potential SQL injection and weak cryptography detected',
                    'recommendations': [
                        'Use parameterized queries for database operations',
                        'Replace MD5 hashing with SHA-256 or stronger algorithms'
                    ]
                }
            ],
            'recommendations': [
                'Address critical security vulnerabilities immediately',
                'Refactor high-complexity functions',
                'Implement automated code quality checks'
            ]
        }
        
        print(f"\nAnalysis Results:")
        print(f"Overall Health Score: {results['health_report']['overall_score']}/100")
        print(f"Health Status: {results['health_report']['health_status'].title()}")
        
        print(f"\nKey Insights ({len(results['insights'])} found):")
        for i, insight in enumerate(results['insights'], 1):
            print(f"{i}. [{insight['severity'].upper()}] {insight['title']}")
            print(f"   Description: {insight['description']}")
            print(f"   Recommendations: {len(insight['recommendations'])} suggestions")
        
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"{i}. {rec}")
        
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None


async def example_individual_agents():
    """Example of using individual agents"""
    print("\n=== INDIVIDUAL AGENTS EXAMPLE ===")
    
    project_path = str(Path(__file__).parent)
    
    # Configuration for agents
    config = {
        'code_metrics': {
            'complexity_threshold': 10,
            'coverage_threshold': 80.0
        },
        'dependency_analysis': {
            'include_external_deps': True
        },
        'security_audit': {
            'scan_depth': 'deep'
        }
    }
    
    print("Simulating individual agent usage:")
    
    # 1. Code Metrics Dashboard
    print("\n1. Code Metrics Analysis:")
    metrics_results = {
        'project_metrics': {
            'total_files': 12,
            'total_loc': 3450,
            'overall_complexity': 8.5,
            'overall_coverage': 65.2,
            'quality_gate_status': 'warning'
        },
        'alerts': [
            {'type': 'warning', 'message': 'Test coverage below threshold (65.2% < 80%)'},
            {'type': 'info', 'message': 'Overall complexity within acceptable range'}
        ]
    }
    
    print(f"   Total Files: {metrics_results['project_metrics']['total_files']}")
    print(f"   Lines of Code: {metrics_results['project_metrics']['total_loc']:,}")
    print(f"   Average Complexity: {metrics_results['project_metrics']['overall_complexity']}")
    print(f"   Test Coverage: {metrics_results['project_metrics']['overall_coverage']:.1f}%")
    print(f"   Quality Gate: {metrics_results['project_metrics']['quality_gate_status'].title()}")
    
    # 2. Dependency Analysis
    print("\n2. Dependency Analysis:")
    dependency_results = {
        'dependency_graph': {'nodes': 12, 'edges': 18},
        'circular_dependencies': 1,
        'metrics': {
            'coupling_index': 3.2,
            'stability_index': 0.65
        }
    }
    
    print(f"   Dependencies: {dependency_results['dependency_graph']['nodes']} nodes, {dependency_results['dependency_graph']['edges']} edges")
    print(f"   Circular Dependencies: {dependency_results['circular_dependencies']}")
    print(f"   Coupling Index: {dependency_results['metrics']['coupling_index']}")
    print(f"   Stability Index: {dependency_results['metrics']['stability_index']}")
    
    # 3. Security Audit
    print("\n3. Security Analysis:")
    security_results = {
        'vulnerabilities': 3,
        'critical_issues': 1,
        'security_rating': 'B',
        'recommendations': [
            'Fix SQL injection vulnerability in database module',
            'Replace weak cryptographic functions',
            'Add input validation for user-facing APIs'
        ]
    }
    
    print(f"   Vulnerabilities Found: {security_results['vulnerabilities']}")
    print(f"   Critical Issues: {security_results['critical_issues']}")
    print(f"   Security Rating: {security_results['security_rating']}")
    print(f"   Recommendations: {len(security_results['recommendations'])}")


async def example_migration_planning():
    """Example of migration planning"""
    print("\n=== MIGRATION PLANNING EXAMPLE ===")
    
    # Simulate a Python version upgrade scenario
    source_config = {
        'python_version': '3.8.10',
        'dependencies': {
            'numpy': '1.21.0',
            'pandas': '1.3.0',
            'requests': '2.25.1',
            'flask': '2.0.1'
        }
    }
    
    target_config = {
        'python_version': '3.11.5',
        'dependencies': {
            'numpy': '1.24.0',
            'pandas': '2.0.0',
            'requests': '2.31.0',
            'flask': '2.3.0'
        }
    }
    
    print("Migration Planning: Python 3.8 -> 3.11")
    print(f"Source: Python {source_config['python_version']}")
    print(f"Target: Python {target_config['python_version']}")
    
    # Simulate migration plan results
    migration_plan = {
        'steps': [
            {'name': 'Create System Backup', 'estimated_hours': 2.0, 'complexity': 'medium'},
            {'name': 'Setup Migration Environment', 'estimated_hours': 4.0, 'complexity': 'medium'},
            {'name': 'Upgrade Python Version', 'estimated_hours': 8.0, 'complexity': 'high'},
            {'name': 'Update Dependencies', 'estimated_hours': 12.0, 'complexity': 'high'},
            {'name': 'Run Tests and Validation', 'estimated_hours': 6.0, 'complexity': 'medium'}
        ],
        'timeline': {'total_hours': 32.0, 'total_with_buffer': 41.6},
        'risks': [
            {'severity': 'high', 'description': 'Breaking changes in pandas 2.0'},
            {'severity': 'medium', 'description': 'Potential performance regression'}
        ],
        'compatibility_analysis': [
            {'component': 'numpy', 'status': 'compatible', 'effort': 4.0},
            {'component': 'pandas', 'status': 'major_issues', 'effort': 16.0},
            {'component': 'requests', 'status': 'minor_issues', 'effort': 2.0}
        ]
    }
    
    print(f"\nMigration Plan Summary:")
    print(f"   Total Steps: {len(migration_plan['steps'])}")
    print(f"   Estimated Time: {migration_plan['timeline']['total_hours']:.1f} hours")
    print(f"   With Buffer: {migration_plan['timeline']['total_with_buffer']:.1f} hours")
    print(f"   Risk Count: {len(migration_plan['risks'])}")
    
    print(f"\nKey Steps:")
    for i, step in enumerate(migration_plan['steps'][:3], 1):
        print(f"   {i}. {step['name']} ({step['estimated_hours']:.1f}h, {step['complexity']})")
    
    print(f"\nCompatibility Issues:")
    for comp in migration_plan['compatibility_analysis']:
        status_icon = "!" if comp['status'] == 'major_issues' else "OK" if comp['status'] == 'compatible' else "?"
        print(f"   {status_icon} {comp['component']}: {comp['status']} ({comp['effort']:.1f}h effort)")


async def example_custom_configuration():
    """Example of using custom configuration"""
    print("\n=== CUSTOM CONFIGURATION EXAMPLE ===")
    
    # Custom configuration for different use cases
    configs = {
        'development': {
            'enable_parallel_execution': True,
            'cache_results': True,
            'code_metrics': {
                'complexity_threshold': 15,  # More lenient during development
                'coverage_threshold': 60.0
            },
            'security_audit': {
                'scan_depth': 'standard'  # Faster scans
            }
        },
        'production': {
            'enable_parallel_execution': True,
            'cache_results': False,  # Always fresh analysis
            'code_metrics': {
                'complexity_threshold': 8,   # Stricter for production
                'coverage_threshold': 90.0
            },
            'security_audit': {
                'scan_depth': 'deep',     # Thorough security analysis
                'include_low_confidence': False
            }
        },
        'ci_cd': {
            'enable_parallel_execution': True,
            'cache_results': True,
            'max_concurrent_agents': 2,  # Limited resources in CI
            'code_metrics': {
                'complexity_threshold': 10,
                'coverage_threshold': 80.0
            }
        }
    }
    
    print("Available Configurations:")
    for env_name, config in configs.items():
        print(f"\n{env_name.upper()} Configuration:")
        print(f"   Parallel Execution: {config.get('enable_parallel_execution', False)}")
        print(f"   Result Caching: {config.get('cache_results', False)}")
        
        if 'code_metrics' in config:
            metrics = config['code_metrics']
            print(f"   Complexity Threshold: {metrics.get('complexity_threshold', 10)}")
            print(f"   Coverage Threshold: {metrics.get('coverage_threshold', 80)}%")
        
        if 'security_audit' in config:
            security = config['security_audit']
            print(f"   Security Scan Depth: {security.get('scan_depth', 'standard')}")


async def example_export_and_reporting():
    """Example of exporting results and generating reports"""
    print("\n=== EXPORT AND REPORTING EXAMPLE ===")
    
    # Simulate analysis results
    analysis_results = {
        'session_id': 'analysis_20240127_143022',
        'project_path': '/trading_system',
        'timestamp': datetime.now().isoformat(),
        'health_report': {
            'overall_score': 82.3,
            'health_status': 'good'
        },
        'insights': [
            {'category': 'complexity', 'severity': 'warning', 'title': 'High complexity detected'},
            {'category': 'security', 'severity': 'info', 'title': 'Minor security considerations'}
        ],
        'metadata': {
            'total_execution_time': 45.2,
            'agents_used': ['code_metrics', 'dependency_analysis', 'security_audit']
        }
    }
    
    print("Available Export Formats:")
    
    # 1. JSON Export
    print("\n1. JSON Export:")
    json_filename = f"architecture_analysis_{analysis_results['session_id']}.json"
    print(f"   Filename: {json_filename}")
    print(f"   Contains: Complete analysis results with all data")
    print(f"   Use case: Integration with other tools, detailed analysis")
    
    # 2. HTML Report
    print("\n2. HTML Report:")
    html_filename = f"architecture_report_{analysis_results['session_id']}.html"
    print(f"   Filename: {html_filename}")
    print(f"   Contains: Executive summary, key insights, recommendations")
    print(f"   Use case: Sharing with stakeholders, presentations")
    
    # 3. CSV Export (for metrics)
    print("\n3. CSV Export (Metrics):")
    csv_filename = f"metrics_data_{analysis_results['session_id']}.csv"
    print(f"   Filename: {csv_filename}")
    print(f"   Contains: File-level metrics, complexity scores, coverage data")
    print(f"   Use case: Trend analysis, data visualization")
    
    print(f"\nReport Summary:")
    print(f"   Session ID: {analysis_results['session_id']}")
    print(f"   Overall Score: {analysis_results['health_report']['overall_score']}/100")
    print(f"   Health Status: {analysis_results['health_report']['health_status'].title()}")
    print(f"   Insights Generated: {len(analysis_results['insights'])}")
    print(f"   Execution Time: {analysis_results['metadata']['total_execution_time']}s")


async def example_integration_workflow():
    """Example of integrating the System Architect suite into a development workflow"""
    print("\n=== INTEGRATION WORKFLOW EXAMPLE ===")
    
    workflow_steps = [
        {
            'step': 'Pre-commit Hook',
            'description': 'Run quick analysis on changed files',
            'config': 'development',
            'scope': 'quick',
            'time': '30-60 seconds'
        },
        {
            'step': 'Pull Request Analysis',
            'description': 'Comprehensive analysis of feature branch',
            'config': 'development',
            'scope': 'standard',
            'time': '2-5 minutes'
        },
        {
            'step': 'Nightly Build Analysis',
            'description': 'Deep analysis of entire codebase',
            'config': 'production',
            'scope': 'comprehensive',
            'time': '10-30 minutes'
        },
        {
            'step': 'Release Preparation',
            'description': 'Full architecture audit before release',
            'config': 'production',
            'scope': 'deep',
            'time': '30-60 minutes'
        }
    ]
    
    print("Recommended Integration Points:")
    
    for i, step in enumerate(workflow_steps, 1):
        print(f"\n{i}. {step['step']}:")
        print(f"   Description: {step['description']}")
        print(f"   Configuration: {step['config']}")
        print(f"   Scope: {step['scope']}")
        print(f"   Expected Time: {step['time']}")
    
    print("\nIntegration Benefits:")
    benefits = [
        'Early detection of architecture issues',
        'Consistent code quality standards',
        'Automated technical debt tracking',
        'Security vulnerability prevention',
        'Migration planning automation',
        'Historical trend analysis'
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"   {i}. {benefit}")


async def main():
    """Run all usage examples"""
    print("SYSTEM ARCHITECT SUITE - USAGE EXAMPLES")
    print("=" * 60)
    print("This demonstrates how to use the complete System Architect suite")
    print("for comprehensive code analysis and architecture management.\n")
    
    try:
        # Run all examples
        await example_basic_usage()
        await example_individual_agents()
        await example_migration_planning()
        await example_custom_configuration()
        await example_export_and_reporting()
        await example_integration_workflow()
        
        print("\n" + "=" * 60)
        print("USAGE EXAMPLES COMPLETED")
        print("=" * 60)
        
        print("\nNEXT STEPS:")
        print("1. Choose the appropriate configuration for your environment")
        print("2. Start with individual agents to understand their capabilities")
        print("3. Use the master coordinator for comprehensive analysis")
        print("4. Integrate into your development workflow gradually")
        print("5. Export results for sharing and historical tracking")
        
        print("\nFOR PRODUCTION USE:")
        print("1. Update import paths to match your project structure")
        print("2. Customize configuration for your specific needs")
        print("3. Set up proper error handling and logging")
        print("4. Consider performance implications for large codebases")
        print("5. Establish quality gates and automation triggers")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Please ensure all components are properly installed and configured.")


if __name__ == "__main__":
    asyncio.run(main())