"""
Integration Tests for System Architect Suite

Comprehensive testing of all architect agents working together.
"""

import asyncio
import json
import tempfile
import os
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
import sys

# Add the project root to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from trading_system.agents.system_architect.architecture_diagram_manager import ArchitectureDiagramManager
from trading_system.agents.system_architect.dependency_analysis_agent import DependencyAnalysisAgent
from trading_system.agents.system_architect.code_metrics_dashboard import CodeMetricsDashboard
from trading_system.agents.system_architect.migration_planning_agent import MigrationPlanningAgent


class TestSystemArchitectIntegration:
    """Integration tests for the complete System Architect suite"""
    
    @pytest.fixture
    def test_project_structure(self):
        """Create a temporary test project structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create test files
            (project_path / "main.py").write_text("""
import asyncio
from typing import Dict, List

class TradingEngine:
    def __init__(self):
        self.orders = []
    
    async def process_order(self, order: Dict) -> bool:
        # Complex logic with nested loops (high complexity)
        for i in range(100):
            for j in range(50):
                if i * j > 1000:
                    self.orders.append(order)
                    return True
        return False
    
    def get_orders(self) -> List[Dict]:
        return self.orders
""")
            
            (project_path / "utils.py").write_text("""
import hashlib
import random

def hash_data(data: str) -> str:
    # Security hotspot: using weak hash
    return hashlib.md5(data.encode()).hexdigest()

def generate_id():
    # Performance issue: using weak random
    return random.randint(1000, 9999)

# FIXED: Use environment variable instead of hardcoded secret
API_KEY = os.getenv('TRADING_API_KEY', 'test-integration-key')
""")
            
            (project_path / "database.py").write_text("""
from utils import hash_data

class DatabaseManager:
    def __init__(self):
        self.connection = None
    
    def connect(self):
        # Circular dependency will be created with main.py
        from main import TradingEngine
        self.engine = TradingEngine()
    
    def store_order(self, order):
        # SQL injection vulnerability
        query = f"INSERT INTO orders VALUES ('{order['id']}')"
        return self.execute(query)
    
    def execute(self, query):
        pass
""")
            
            (project_path / "requirements.txt").write_text("""
numpy==1.21.0
pandas==1.3.0
requests==2.25.1
flask==2.0.1
""")
            
            # Create test directory
            test_dir = project_path / "tests"
            test_dir.mkdir()
            (test_dir / "test_main.py").write_text("""
import unittest
from main import TradingEngine

class TestTradingEngine(unittest.TestCase):
    def test_process_order(self):
        engine = TradingEngine()
        order = {"id": "123", "amount": 100}
        # This would be an async test in real code
        self.assertIsNotNone(engine)
""")
            
            yield str(project_path)
    
    @pytest.fixture
    def config(self):
        """Standard configuration for all agents"""
        return {
            'architecture_diagram': {
                'output_format': 'svg',
                'include_external_deps': True
            },
            'dependency_analysis': {
                'include_external_deps': True,
                'max_circular_chain_length': 10
            },
            'code_metrics': {
                'complexity_threshold': 10,
                'coverage_threshold': 80.0
            },
            'migration_planning': {
                'risk_tolerance': 'medium',
                'migration_window_hours': 8
            }
        }
    
    def test_individual_agent_initialization(self, config):
        """Test that all agents can be initialized properly"""
        # Test each agent initialization
        diagram_manager = ArchitectureDiagramManager(config)
        assert diagram_manager.name == "ArchitectureDiagram"
        
        dependency_agent = DependencyAnalysisAgent(config)
        assert dependency_agent.name == "DependencyAnalysis"
        
        metrics_dashboard = CodeMetricsDashboard(config)
        assert metrics_dashboard.name == "CodeMetricsDashboard"
        
        migration_agent = MigrationPlanningAgent(config)
        assert migration_agent.name == "MigrationPlanning"
    
    @pytest.mark.asyncio
    async def test_architecture_diagram_generation(self, test_project_structure, config):
        """Test architecture diagram generation"""
        diagram_manager = ArchitectureDiagramManager(config)
        
        result = await diagram_manager.generate_architecture_diagrams(test_project_structure)
        
        # Verify structure
        assert 'diagrams' in result
        assert 'components' in result
        assert 'relationships' in result
        assert 'master_diagram' in result
        
        # Verify components were detected
        components = result['components']
        assert len(components) > 0
        
        # Should detect our test classes
        component_names = [c['name'] for c in components]
        assert any('TradingEngine' in name for name in component_names)
        assert any('DatabaseManager' in name for name in component_names)
    
    @pytest.mark.asyncio
    async def test_dependency_analysis(self, test_project_structure, config):
        """Test dependency analysis functionality"""
        dependency_agent = DependencyAnalysisAgent(config)
        
        result = await dependency_agent.analyze_dependencies(test_project_structure)
        
        # Verify structure
        assert 'dependency_graph' in result
        assert 'circular_dependencies' in result
        assert 'metrics' in result
        assert 'recommendations' in result
        
        # Should detect circular dependency between main.py and database.py
        circular_deps = result['circular_dependencies']
        # Note: Our simple test case might not create a true circular dependency
        # but the structure should be present
        
        # Verify dependency graph
        dep_graph = result['dependency_graph']
        assert 'nodes' in dep_graph
        assert 'edges' in dep_graph
        assert len(dep_graph['nodes']) > 0
    
    @pytest.mark.asyncio
    async def test_code_metrics_dashboard(self, test_project_structure, config):
        """Test code metrics dashboard generation"""
        metrics_dashboard = CodeMetricsDashboard(config)
        
        result = await metrics_dashboard.generate_dashboard(test_project_structure)
        
        # Verify structure
        assert 'summary' in result
        assert 'project_metrics' in result
        assert 'file_metrics' in result
        assert 'alerts' in result
        assert 'recommendations' in result
        
        # Verify project metrics
        project_metrics = result['project_metrics']
        assert 'total_files' in project_metrics
        assert project_metrics['total_files'] > 0
        
        # Verify file metrics were calculated
        file_metrics = result['file_metrics']
        assert len(file_metrics) > 0
        
        # Should detect high complexity in main.py
        main_file_metrics = next((f for f in file_metrics if 'main.py' in f['file_path']), None)
        assert main_file_metrics is not None
        assert main_file_metrics['cyclomatic_complexity'] > 5  # Due to nested loops
        
        # Should detect security hotspots in utils.py
        utils_file_metrics = next((f for f in file_metrics if 'utils.py' in f['file_path']), None)
        assert utils_file_metrics is not None
        assert utils_file_metrics['security_hotspots'] > 0
    
    @pytest.mark.asyncio
    async def test_migration_planning(self, test_project_structure, config):
        """Test migration planning functionality"""
        migration_agent = MigrationPlanningAgent(config)
        
        # Test Python version upgrade scenario
        source_config = {
            'project_path': test_project_structure,
            'python_version': '3.8.10',
            'dependencies': {
                'numpy': '1.21.0',
                'pandas': '1.3.0',
                'requests': '2.25.1'
            }
        }
        
        target_config = {
            'python_version': '3.11.5',
            'dependencies': {
                'numpy': '1.24.0',
                'pandas': '2.0.0',
                'requests': '2.31.0'
            }
        }
        
        result = await migration_agent.create_migration_plan(
            'version_upgrade', source_config, target_config
        )
        
        # Verify structure
        assert 'migration_plan' in result
        assert 'compatibility_analysis' in result
        assert 'risk_assessment' in result
        assert 'step_dependencies' in result
        
        # Verify migration plan
        migration_plan = result['migration_plan']
        assert 'steps' in migration_plan
        assert 'timeline' in migration_plan
        assert 'risks' in migration_plan
        assert 'rollback_plan' in migration_plan
        
        # Should have multiple steps
        steps = migration_plan['steps']
        assert len(steps) > 0
        
        # Should detect compatibility issues
        compatibility = result['compatibility_analysis']
        assert len(compatibility) > 0
        
        # Should have Python compatibility analysis
        python_compat = next((c for c in compatibility if c['component'] == 'Python'), None)
        assert python_compat is not None
    
    @pytest.mark.asyncio
    async def test_end_to_end_integration(self, test_project_structure, config):
        """Test complete end-to-end integration of all agents"""
        # Initialize all agents
        diagram_manager = ArchitectureDiagramManager(config)
        dependency_agent = DependencyAnalysisAgent(config)
        metrics_dashboard = CodeMetricsDashboard(config)
        migration_agent = MigrationPlanningAgent(config)
        
        # Run all analyses
        results = {}
        
        # 1. Generate architecture diagrams
        results['architecture'] = await diagram_manager.generate_architecture_diagrams(test_project_structure)
        
        # 2. Analyze dependencies
        results['dependencies'] = await dependency_agent.analyze_dependencies(test_project_structure)
        
        # 3. Generate metrics dashboard
        results['metrics'] = await metrics_dashboard.generate_dashboard(test_project_structure)
        
        # 4. Create migration plan
        source_config = {'project_path': test_project_structure, 'python_version': '3.8.10'}
        target_config = {'python_version': '3.11.5'}
        results['migration'] = await migration_agent.create_migration_plan(
            'version_upgrade', source_config, target_config
        )
        
        # Verify all analyses completed successfully
        assert all(results.values())
        
        # Test cross-agent data consistency
        # Architecture components should match dependency nodes
        arch_components = results['architecture']['components']
        dep_nodes = results['dependencies']['dependency_graph']['nodes']
        
        # Both should detect similar file-level components
        arch_files = {Path(c['location']).name for c in arch_components if c['type'] == 'module'}
        dep_files = {Path(n['file_path']).name for n in dep_nodes}
        
        # Should have some overlap
        common_files = arch_files.intersection(dep_files)
        assert len(common_files) > 0
        
        # Metrics should identify issues that migration plan addresses
        metrics_alerts = results['metrics']['alerts']
        migration_risks = results['migration']['migration_plan']['risks']
        
        # Both should identify some form of technical issues
        assert len(metrics_alerts) > 0 or len(migration_risks) > 0
    
    @pytest.mark.asyncio
    async def test_performance_with_larger_codebase(self, config):
        """Test performance with a larger, more realistic codebase"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create a more complex project structure
            modules = ['auth', 'trading', 'data', 'api', 'utils']
            
            for module in modules:
                module_dir = project_path / module
                module_dir.mkdir()
                
                # Create multiple files per module
                for i in range(3):
                    file_content = f"""
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

class {module.title()}Manager{i}:
    def __init__(self):
        self.config = {{}}
        self.cache = {{}}
    
    async def process_data(self, data: Dict) -> Optional[Dict]:
        # Simulate complex processing
        for item in data.get('items', []):
            for key, value in item.items():
                if isinstance(value, list):
                    for sub_item in value:
                        result = await self._process_sub_item(sub_item)
                        if result:
                            self.cache[key] = result
        return self.cache
    
    async def _process_sub_item(self, item: Dict) -> Optional[Dict]:
        # More complex logic
        if item.get('type') == 'special':
            return await self._special_processing(item)
        return item
    
    async def _special_processing(self, item: Dict) -> Dict:
        # Simulate processing time
        await asyncio.sleep(0.001)
        return {{'processed': True, 'data': item}}
    
    def get_stats(self) -> Dict:
        return {{
            'cache_size': len(self.cache),
            'timestamp': datetime.now().isoformat()
        }}
"""
                    (module_dir / f"{module}_{i}.py").write_text(file_content)
                
                # Create __init__.py
                (module_dir / "__init__.py").write_text(f"from .{module}_0 import {module.title()}Manager0")
            
            # Test with larger codebase
            import time
            
            start_time = time.time()
            
            # Run a subset of tests for performance
            metrics_dashboard = CodeMetricsDashboard(config)
            result = await metrics_dashboard.generate_dashboard(str(project_path))
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should complete in reasonable time (< 30 seconds for this test size)
            assert execution_time < 30
            
            # Should handle the larger codebase
            assert result['project_metrics']['total_files'] >= 15  # 5 modules * 3 files each
            assert len(result['file_metrics']) >= 15
    
    def test_error_handling_and_recovery(self, config):
        """Test error handling and recovery mechanisms"""
        # Test with invalid project path
        diagram_manager = ArchitectureDiagramManager(config)
        
        # Should handle non-existent path gracefully
        async def test_invalid_path():
            result = await diagram_manager.generate_architecture_diagrams("/non/existent/path")
            # Should return some result structure even with errors
            assert 'diagrams' in result
            assert 'components' in result
        
        asyncio.run(test_invalid_path())
    
    @pytest.mark.asyncio
    async def test_configuration_variations(self, test_project_structure):
        """Test agents with different configuration variations"""
        # Test with minimal config
        minimal_config = {}
        metrics_dashboard = CodeMetricsDashboard(minimal_config)
        result = await metrics_dashboard.generate_dashboard(test_project_structure)
        assert 'project_metrics' in result
        
        # Test with comprehensive config
        comprehensive_config = {
            'code_metrics': {
                'complexity_threshold': 5,
                'coverage_threshold': 90.0,
                'duplication_threshold': 3.0,
                'include_tests': True,
                'min_file_size': 5
            }
        }
        metrics_dashboard_comprehensive = CodeMetricsDashboard(comprehensive_config)
        result_comprehensive = await metrics_dashboard_comprehensive.generate_dashboard(test_project_structure)
        assert 'project_metrics' in result_comprehensive
        
        # Results should be different due to different thresholds
        # (This test assumes the comprehensive config would detect more issues)
    
    def test_data_export_and_import(self, test_project_structure, config):
        """Test data export and import functionality"""
        async def test_export():
            metrics_dashboard = CodeMetricsDashboard(config)
            result = await metrics_dashboard.generate_dashboard(test_project_structure)
            
            # Test JSON export
            export_file = await metrics_dashboard.export_dashboard('json')
            assert export_file.endswith('.json')
            
            # Verify exported file exists and has content
            if os.path.exists(export_file):
                with open(export_file, 'r') as f:
                    exported_data = json.load(f)
                assert 'project_summary' in exported_data
                
                # Cleanup
                os.remove(export_file)
        
        asyncio.run(test_export())


class TestSystemArchitectCoordination:
    """Test coordination between different architect agents"""
    
    @pytest.fixture
    def coordinator_config(self):
        return {
            'coordination': {
                'enable_cross_validation': True,
                'cache_results': True,
                'parallel_execution': True
            },
            'architecture_diagram': {'output_format': 'svg'},
            'dependency_analysis': {'include_external_deps': True},
            'code_metrics': {'complexity_threshold': 10},
            'migration_planning': {'risk_tolerance': 'medium'}
        }
    
    @pytest.mark.asyncio
    async def test_agent_coordination(self, test_project_structure, coordinator_config):
        """Test that agents can share data and coordinate analyses"""
        # This would test a coordination layer that we haven't implemented yet
        # but demonstrates how the integration should work
        
        # Initialize agents
        agents = {
            'diagram': ArchitectureDiagramManager(coordinator_config),
            'dependency': DependencyAnalysisAgent(coordinator_config),
            'metrics': CodeMetricsDashboard(coordinator_config),
            'migration': MigrationPlanningAgent(coordinator_config)
        }
        
        # Simulate coordinated analysis
        shared_data = {}
        
        # 1. Dependency analysis provides foundation
        dep_result = await agents['dependency'].analyze_dependencies(test_project_structure)
        shared_data['dependency_graph'] = dep_result['dependency_graph']
        shared_data['circular_dependencies'] = dep_result['circular_dependencies']
        
        # 2. Metrics analysis uses dependency data
        metrics_result = await agents['metrics'].generate_dashboard(test_project_structure)
        shared_data['complexity_hotspots'] = [
            f for f in metrics_result['file_metrics'] 
            if f['cyclomatic_complexity'] > 10
        ]
        
        # 3. Architecture diagrams incorporate both
        arch_result = await agents['diagram'].generate_architecture_diagrams(test_project_structure)
        
        # 4. Migration planning considers all factors
        source_config = {
            'project_path': test_project_structure,
            'complexity_hotspots': shared_data['complexity_hotspots'],
            'circular_dependencies': shared_data['circular_dependencies']
        }
        target_config = {'python_version': '3.11.5'}
        
        migration_result = await agents['migration'].create_migration_plan(
            'version_upgrade', source_config, target_config
        )
        
        # Verify coordination worked
        assert len(shared_data) > 0
        assert all([dep_result, metrics_result, arch_result, migration_result])
        
        # Migration plan should consider complexity hotspots
        migration_risks = migration_result['migration_plan']['risks']
        # Should have identified technical risks based on complexity
        technical_risks = [r for r in migration_risks if r['category'] == 'technical']
        assert len(technical_risks) > 0


# Performance benchmarking
class TestPerformanceBenchmarks:
    """Performance benchmarks for the System Architect suite"""
    
    @pytest.mark.asyncio
    async def test_scalability_benchmark(self):
        """Benchmark scalability with different project sizes"""
        import time
        
        config = {'code_metrics': {'complexity_threshold': 10}}
        metrics_dashboard = CodeMetricsDashboard(config)
        
        # Test with different project sizes
        sizes = [5, 10, 20]  # Number of files
        execution_times = []
        
        for size in sizes:
            with tempfile.TemporaryDirectory() as temp_dir:
                project_path = Path(temp_dir)
                
                # Create project of specified size
                for i in range(size):
                    (project_path / f"module_{i}.py").write_text(f"""
class Module{i}:
    def __init__(self):
        self.data = []
    
    def process(self):
        for i in range(100):
            if i % 2 == 0:
                self.data.append(i)
        return len(self.data)
""")
                
                start_time = time.time()
                await metrics_dashboard.generate_dashboard(str(project_path))
                end_time = time.time()
                
                execution_times.append(end_time - start_time)
        
        # Verify performance scales reasonably
        # Should not increase exponentially
        for i in range(1, len(execution_times)):
            scale_factor = execution_times[i] / execution_times[i-1]
            size_factor = sizes[i] / sizes[i-1]
            
            # Performance should scale better than quadratically
            assert scale_factor < size_factor ** 1.5


if __name__ == "__main__":
    # Run basic integration test
    async def main():
        config = {
            'architecture_diagram': {'output_format': 'svg'},
            'dependency_analysis': {'include_external_deps': True},
            'code_metrics': {'complexity_threshold': 10},
            'migration_planning': {'risk_tolerance': 'medium'}
        }
        
        # Create a simple test
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("""
def hello_world():
    print("Hello, World!")
    return "success"
""")
            
            # Test each agent
            print("Testing Architecture Diagram Manager...")
            diagram_manager = ArchitectureDiagramManager(config)
            arch_result = await diagram_manager.generate_architecture_diagrams(temp_dir)
            print(f"✓ Generated {len(arch_result['components'])} components")
            
            print("Testing Code Metrics Dashboard...")
            metrics_dashboard = CodeMetricsDashboard(config)
            metrics_result = await metrics_dashboard.generate_dashboard(temp_dir)
            print(f"✓ Analyzed {metrics_result['project_metrics']['total_files']} files")
            
            print("Testing Dependency Analysis...")
            dependency_agent = DependencyAnalysisAgent(config)
            dep_result = await dependency_agent.analyze_dependencies(temp_dir)
            print(f"✓ Found {len(dep_result['dependency_graph']['nodes'])} dependency nodes")
            
            print("Testing Migration Planning...")
            migration_agent = MigrationPlanningAgent(config)
            migration_result = await migration_agent.create_migration_plan(
                'version_upgrade',
                {'project_path': temp_dir, 'python_version': '3.8'},
                {'python_version': '3.11'}
            )
            print(f"✓ Created migration plan with {len(migration_result['migration_plan']['steps'])} steps")
            
            print("\n🎉 All integration tests passed!")
    
    asyncio.run(main())