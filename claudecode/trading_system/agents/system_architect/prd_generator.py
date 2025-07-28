"""
Product Requirements Document (PRD) Generator

This agent automatically generates comprehensive PRDs for trading system
components based on code analysis and documentation extraction.
"""

import ast
import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import yaml

from ...core.base.agent import BaseAgent


@dataclass
class ComponentAnalysis:
    """Analysis results for a component"""
    name: str
    type: str  # 'service', 'module', 'class', 'function'
    purpose: str
    dependencies: List[str]
    interfaces: List[Dict[str, Any]]
    data_models: List[Dict[str, Any]]
    business_logic: List[str]
    technical_requirements: List[str]
    performance_requirements: List[str]
    security_requirements: List[str]
    file_paths: List[str]


@dataclass
class PRDDocument:
    """Complete PRD document structure"""
    title: str
    version: str
    author: str
    date: str
    overview: str
    objectives: List[str]
    success_metrics: List[str]
    user_stories: List[Dict[str, Any]]
    functional_requirements: List[Dict[str, Any]]
    technical_requirements: List[Dict[str, Any]]
    architecture: Dict[str, Any]
    data_models: List[Dict[str, Any]]
    api_specifications: List[Dict[str, Any]]
    security_requirements: List[str]
    performance_requirements: List[str]
    integration_points: List[Dict[str, Any]]
    testing_strategy: Dict[str, Any]
    deployment_requirements: List[str]
    monitoring_requirements: List[str]
    risks_and_mitigations: List[Dict[str, Any]]
    timeline: Dict[str, Any]
    acceptance_criteria: List[str]


class PRDGenerator(BaseAgent):
    """
    Product Requirements Document Generator
    
    Analyzes code components and generates comprehensive PRDs including:
    - Business objectives and success metrics
    - User stories and use cases
    - Functional and technical requirements
    - Architecture specifications
    - API documentation
    - Security and performance requirements
    - Testing and deployment strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("PRDGenerator", config.get('prd_generator', {}))
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.template_dir = Path(config.get('template_dir', 'templates'))
        self.output_dir = Path(config.get('output_dir', 'docs/prds'))
        self.include_code_examples = config.get('include_code_examples', True)
        self.prd_format = config.get('prd_format', 'markdown')  # markdown, json, yaml
        
        # Analysis patterns
        self.business_patterns = self._load_business_patterns()
        self.technical_patterns = self._load_technical_patterns()
        
    async def generate_prd(self, component_path: str) -> Dict[str, Any]:
        """
        Generate a comprehensive PRD for a component
        
        Args:
            component_path: Path to the component to analyze
            
        Returns:
            Generated PRD document and metadata
        """
        self.logger.info(f"Generating PRD for component: {component_path}")
        
        # Analyze component
        analysis = await self._analyze_component(component_path)
        
        # Generate PRD document
        prd = await self._create_prd_document(analysis)
        
        # Save PRD
        output_path = await self._save_prd(prd, analysis.name)
        
        return {
            'prd_document': self._prd_to_dict(prd),
            'component_analysis': self._analysis_to_dict(analysis),
            'output_path': str(output_path),
            'format': self.prd_format
        }
    
    async def _analyze_component(self, component_path: str) -> ComponentAnalysis:
        """Analyze component to extract requirements information"""
        path = Path(component_path)
        
        if path.is_file():
            return await self._analyze_single_file(path)
        elif path.is_dir():
            return await self._analyze_directory(path)
        else:
            raise ValueError(f"Invalid component path: {component_path}")
    
    async def _analyze_single_file(self, file_path: Path) -> ComponentAnalysis:
        """Analyze a single file component"""
        self.logger.info(f"Analyzing file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.logger.error(f"Could not parse {file_path}: {e}")
            tree = None
        
        # Extract component information
        name = file_path.stem
        component_type = self._determine_component_type(content, tree)
        purpose = self._extract_purpose(content, tree)
        dependencies = self._extract_dependencies(content, tree)
        interfaces = self._extract_interfaces(content, tree)
        data_models = self._extract_data_models(content, tree)
        business_logic = self._extract_business_logic(content, tree)
        technical_reqs = self._extract_technical_requirements(content)
        performance_reqs = self._extract_performance_requirements(content)
        security_reqs = self._extract_security_requirements(content)
        
        return ComponentAnalysis(
            name=name,
            type=component_type,
            purpose=purpose,
            dependencies=dependencies,
            interfaces=interfaces,
            data_models=data_models,
            business_logic=business_logic,
            technical_requirements=technical_reqs,
            performance_requirements=performance_reqs,
            security_requirements=security_reqs,
            file_paths=[str(file_path)]
        )
    
    async def _analyze_directory(self, dir_path: Path) -> ComponentAnalysis:
        """Analyze a directory component (service/module)"""
        self.logger.info(f"Analyzing directory: {dir_path}")
        
        # Find Python files
        py_files = list(dir_path.rglob('*.py'))
        
        if not py_files:
            raise ValueError(f"No Python files found in {dir_path}")
        
        # Aggregate analysis from all files
        name = dir_path.name
        component_type = 'service' if self._is_service_directory(dir_path) else 'module'
        
        all_content = ""
        all_trees = []
        file_paths = []
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_content += content + "\n"
                    file_paths.append(str(py_file))
                
                tree = ast.parse(content)
                all_trees.append(tree)
            except Exception as e:
                self.logger.warning(f"Could not process {py_file}: {e}")
        
        # Extract aggregated information
        purpose = self._extract_purpose(all_content, all_trees[0] if all_trees else None)
        dependencies = self._extract_dependencies(all_content, None)
        interfaces = []
        data_models = []
        business_logic = []
        
        for tree in all_trees:
            interfaces.extend(self._extract_interfaces("", tree))
            data_models.extend(self._extract_data_models("", tree))
            business_logic.extend(self._extract_business_logic("", tree))
        
        # Remove duplicates
        interfaces = self._deduplicate_interfaces(interfaces)
        data_models = self._deduplicate_data_models(data_models)
        business_logic = list(set(business_logic))
        
        technical_reqs = self._extract_technical_requirements(all_content)
        performance_reqs = self._extract_performance_requirements(all_content)
        security_reqs = self._extract_security_requirements(all_content)
        
        return ComponentAnalysis(
            name=name,
            type=component_type,
            purpose=purpose,
            dependencies=dependencies,
            interfaces=interfaces,
            data_models=data_models,
            business_logic=business_logic,
            technical_requirements=technical_reqs,
            performance_requirements=performance_reqs,
            security_requirements=security_reqs,
            file_paths=file_paths
        )
    
    def _determine_component_type(self, content: str, tree: Optional[ast.AST]) -> str:
        """Determine the type of component"""
        if 'class ' in content and 'def ' in content:
            return 'class'
        elif 'def ' in content and 'class ' not in content:
            return 'function'
        elif any(pattern in content for pattern in ['FastAPI', 'Flask', 'django', 'tornado']):
            return 'service'
        else:
            return 'module'
    
    def _is_service_directory(self, dir_path: Path) -> bool:
        """Check if directory represents a service"""
        service_indicators = [
            'main.py', 'app.py', 'server.py', 'service.py',
            'requirements.txt', 'Dockerfile', 'docker-compose.yml'
        ]
        
        return any((dir_path / indicator).exists() for indicator in service_indicators)
    
    def _extract_purpose(self, content: str, tree: Optional[ast.AST]) -> str:
        """Extract component purpose from docstrings and comments"""
        # Try to extract from module docstring
        if tree and isinstance(tree, ast.Module) and tree.body:
            first_node = tree.body[0]
            if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Str):
                return first_node.value.s.strip()
        
        # Try to extract from file header comments
        lines = content.split('\n')
        purpose_lines = []
        
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if line.startswith('"""') or line.startswith("'''"):
                # Start of docstring
                purpose_lines.append(line[3:])
            elif line.startswith('#'):
                # Comment
                purpose_lines.append(line[1:].strip())
            elif purpose_lines:
                # End of initial comments/docstring
                break
        
        if purpose_lines:
            return ' '.join(purpose_lines).strip()
        
        # Fallback: analyze function/class names
        return self._infer_purpose_from_names(content)
    
    def _infer_purpose_from_names(self, content: str) -> str:
        """Infer purpose from function and class names"""
        # Extract class and function names
        class_names = re.findall(r'class\s+(\w+)', content)
        func_names = re.findall(r'def\s+(\w+)', content)
        
        # Analyze naming patterns
        purpose_keywords = []
        
        for name in class_names + func_names:
            # Convert camelCase to words
            words = re.findall(r'[A-Z][a-z]*|[a-z]+', name)
            purpose_keywords.extend(words)
        
        if purpose_keywords:
            return f"Component for {' '.join(purpose_keywords[:5]).lower()}"
        
        return "Component purpose not clearly defined in code"
    
    def _extract_dependencies(self, content: str, tree: Optional[ast.AST]) -> List[str]:
        """Extract component dependencies"""
        dependencies = []
        
        # Extract imports
        import_patterns = [
            r'import\s+(\w+)',
            r'from\s+(\w+)\s+import',
            r'from\s+([\w.]+)\s+import'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            dependencies.extend(matches)
        
        # Filter out standard library modules
        standard_lib = {
            'os', 'sys', 'json', 'time', 'datetime', 'logging', 're', 'pathlib',
            'typing', 'asyncio', 'functools', 'itertools', 'collections'
        }
        
        external_deps = [dep for dep in dependencies if dep not in standard_lib]
        
        return list(set(external_deps))
    
    def _extract_interfaces(self, content: str, tree: Optional[ast.AST]) -> List[Dict[str, Any]]:
        """Extract API interfaces and endpoints"""
        interfaces = []
        
        if not tree:
            return interfaces
        
        # Find FastAPI/Flask routes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for route decorators
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        func_name = getattr(decorator.func, 'attr', '')
                        if func_name in ['get', 'post', 'put', 'delete', 'patch']:
                            # Extract route information
                            route_path = ''
                            if decorator.args:
                                if isinstance(decorator.args[0], ast.Str):
                                    route_path = decorator.args[0].s
                            
                            interfaces.append({
                                'type': 'REST',
                                'method': func_name.upper(),
                                'path': route_path,
                                'function': node.name,
                                'description': self._extract_function_docstring(node),
                                'parameters': self._extract_function_parameters(node),
                                'returns': self._extract_return_type(node)
                            })
        
        # Find class methods that look like interfaces
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and not method.name.startswith('_'):
                        interfaces.append({
                            'type': 'method',
                            'class': node.name,
                            'method': method.name,
                            'description': self._extract_function_docstring(method),
                            'parameters': self._extract_function_parameters(method),
                            'returns': self._extract_return_type(method)
                        })
        
        return interfaces
    
    def _extract_data_models(self, content: str, tree: Optional[ast.AST]) -> List[Dict[str, Any]]:
        """Extract data models and schemas"""
        models = []
        
        if not tree:
            return models
        
        # Find Pydantic models, dataclasses, and other data structures
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it's a data model
                is_pydantic = any(
                    isinstance(base, ast.Name) and base.id == 'BaseModel'
                    for base in node.bases
                )
                
                has_dataclass = any(
                    isinstance(decorator, ast.Name) and decorator.id == 'dataclass'
                    for decorator in node.decorator_list
                )
                
                if is_pydantic or has_dataclass or self._looks_like_data_model(node):
                    models.append({
                        'name': node.name,
                        'type': 'pydantic' if is_pydantic else 'dataclass' if has_dataclass else 'class',
                        'fields': self._extract_class_attributes(node),
                        'description': self._extract_class_docstring(node),
                        'methods': [method.name for method in node.body if isinstance(method, ast.FunctionDef)]
                    })
        
        return models
    
    def _extract_business_logic(self, content: str, tree: Optional[ast.AST]) -> List[str]:
        """Extract business logic descriptions"""
        business_logic = []
        
        # Look for business-related function names and comments
        business_keywords = [
            'calculate', 'process', 'validate', 'execute', 'trade', 'order',
            'portfolio', 'strategy', 'risk', 'price', 'market', 'account'
        ]
        
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name.lower()
                    if any(keyword in func_name for keyword in business_keywords):
                        description = self._extract_function_docstring(node)
                        if description:
                            business_logic.append(f"{node.name}: {description}")
                        else:
                            business_logic.append(f"{node.name}: {self._infer_function_purpose(node.name)}")
        
        # Extract from comments
        comment_logic = re.findall(r'#.*(?:business|logic|rule|requirement).*', content, re.IGNORECASE)
        business_logic.extend([comment.strip('#').strip() for comment in comment_logic])
        
        return business_logic
    
    def _extract_technical_requirements(self, content: str) -> List[str]:
        """Extract technical requirements from code"""
        requirements = []
        
        # Database requirements
        if any(db in content for db in ['postgresql', 'mysql', 'mongodb', 'redis']):
            requirements.append("Database connectivity required")
        
        # Message queue requirements
        if any(mq in content for mq in ['rabbitmq', 'kafka', 'celery']):
            requirements.append("Message queue integration required")
        
        # External API requirements
        if any(api in content for api in ['requests', 'aiohttp', 'httpx']):
            requirements.append("External API integration required")
        
        # Authentication requirements
        if any(auth in content for auth in ['jwt', 'oauth', 'auth', 'token']):
            requirements.append("Authentication and authorization required")
        
        # Caching requirements
        if any(cache in content for cache in ['cache', 'redis', 'memcached']):
            requirements.append("Caching mechanism required")
        
        return requirements
    
    def _extract_performance_requirements(self, content: str) -> List[str]:
        """Extract performance requirements"""
        requirements = []
        
        # Async requirements
        if 'async def' in content or 'await ' in content:
            requirements.append("Asynchronous processing capability")
        
        # Concurrency requirements
        if any(conc in content for conc in ['threading', 'multiprocessing', 'concurrent']):
            requirements.append("Concurrent processing support")
        
        # Large data handling
        if any(data in content for data in ['pandas', 'numpy', 'batch', 'bulk']):
            requirements.append("Large dataset processing capability")
        
        return requirements
    
    def _extract_security_requirements(self, content: str) -> List[str]:
        """Extract security requirements"""
        requirements = []
        
        # Encryption requirements
        if any(crypto in content for crypto in ['encrypt', 'decrypt', 'hash', 'bcrypt']):
            requirements.append("Data encryption and hashing")
        
        # Input validation
        if any(val in content for val in ['validate', 'sanitize', 'escape']):
            requirements.append("Input validation and sanitization")
        
        # Secure communication
        if any(sec in content for sec in ['https', 'ssl', 'tls']):
            requirements.append("Secure communication protocols")
        
        return requirements
    
    async def _create_prd_document(self, analysis: ComponentAnalysis) -> PRDDocument:
        """Create PRD document from component analysis"""
        
        # Generate document sections
        title = f"Product Requirements Document - {analysis.name}"
        overview = self._generate_overview(analysis)
        objectives = self._generate_objectives(analysis)
        success_metrics = self._generate_success_metrics(analysis)
        user_stories = self._generate_user_stories(analysis)
        functional_requirements = self._generate_functional_requirements(analysis)
        technical_requirements = self._generate_technical_requirements_section(analysis)
        architecture = self._generate_architecture_section(analysis)
        api_specs = self._generate_api_specifications(analysis)
        testing_strategy = self._generate_testing_strategy(analysis)
        deployment_requirements = self._generate_deployment_requirements(analysis)
        monitoring_requirements = self._generate_monitoring_requirements(analysis)
        risks_and_mitigations = self._generate_risks_and_mitigations(analysis)
        timeline = self._generate_timeline(analysis)
        acceptance_criteria = self._generate_acceptance_criteria(analysis)
        
        return PRDDocument(
            title=title,
            version="1.0",
            author="System Architect Agent",
            date=datetime.now().strftime("%Y-%m-%d"),
            overview=overview,
            objectives=objectives,
            success_metrics=success_metrics,
            user_stories=user_stories,
            functional_requirements=functional_requirements,
            technical_requirements=technical_requirements,
            architecture=architecture,
            data_models=analysis.data_models,
            api_specifications=api_specs,
            security_requirements=analysis.security_requirements,
            performance_requirements=analysis.performance_requirements,
            integration_points=self._generate_integration_points(analysis),
            testing_strategy=testing_strategy,
            deployment_requirements=deployment_requirements,
            monitoring_requirements=monitoring_requirements,
            risks_and_mitigations=risks_and_mitigations,
            timeline=timeline,
            acceptance_criteria=acceptance_criteria
        )
    
    def _generate_overview(self, analysis: ComponentAnalysis) -> str:
        """Generate overview section"""
        return f"""
This document outlines the product requirements for the {analysis.name} {analysis.type}.

**Purpose**: {analysis.purpose}

**Component Type**: {analysis.type.title()}

**Key Dependencies**: {', '.join(analysis.dependencies[:5])}

The {analysis.name} component is designed to provide specific functionality within the trading system architecture, ensuring robust performance, security, and maintainability.
""".strip()
    
    def _generate_objectives(self, analysis: ComponentAnalysis) -> List[str]:
        """Generate business objectives"""
        objectives = []
        
        if analysis.type == 'service':
            objectives.extend([
                "Provide reliable and scalable service functionality",
                "Ensure high availability and fault tolerance",
                "Maintain consistent API performance"
            ])
        elif analysis.type == 'module':
            objectives.extend([
                "Deliver reusable and maintainable functionality",
                "Provide clear and intuitive interfaces",
                "Ensure modularity and loose coupling"
            ])
        
        # Add business-specific objectives based on business logic
        for logic in analysis.business_logic[:3]:
            objectives.append(f"Enable {logic.split(':')[0].lower()} functionality")
        
        return objectives
    
    def _generate_success_metrics(self, analysis: ComponentAnalysis) -> List[str]:
        """Generate success metrics"""
        metrics = [
            "Response time < 100ms for 95% of requests",
            "99.9% uptime and availability",
            "Zero critical security vulnerabilities",
            "Code coverage > 80%",
            "Documentation coverage > 90%"
        ]
        
        if analysis.type == 'service':
            metrics.extend([
                "API error rate < 0.1%",
                "Successful deployment in production",
                "Monitoring and alerting fully configured"
            ])
        
        return metrics
    
    def _generate_user_stories(self, analysis: ComponentAnalysis) -> List[Dict[str, Any]]:
        """Generate user stories"""
        user_stories = []
        
        # Generate stories based on interfaces
        for interface in analysis.interfaces[:5]:
            if interface['type'] == 'REST':
                user_stories.append({
                    'id': f"US-{len(user_stories) + 1}",
                    'title': f"Access {interface['function']} endpoint",
                    'description': f"As a system user, I want to {interface['method']} {interface['path']} so that I can {interface['description'] or 'perform the required operation'}",
                    'acceptance_criteria': [
                        f"Endpoint responds with appropriate HTTP status codes",
                        f"Response format matches API specification",
                        f"Request validation works correctly"
                    ]
                })
        
        # Generate stories based on business logic
        for logic in analysis.business_logic[:3]:
            user_stories.append({
                'id': f"US-{len(user_stories) + 1}",
                'title': f"Use {logic.split(':')[0]} functionality",
                'description': f"As a system user, I want to utilize {logic.split(':')[0]} so that I can achieve the required business outcomes",
                'acceptance_criteria': [
                    "Functionality works as expected",
                    "Error handling is robust",
                    "Performance meets requirements"
                ]
            })
        
        return user_stories
    
    def _generate_functional_requirements(self, analysis: ComponentAnalysis) -> List[Dict[str, Any]]:
        """Generate functional requirements"""
        requirements = []
        
        # Generate requirements from interfaces
        for i, interface in enumerate(analysis.interfaces):
            requirements.append({
                'id': f"FR-{i + 1}",
                'title': f"{interface['type']} Interface",
                'description': f"The system shall provide {interface.get('method', 'access to')} {interface.get('path', interface.get('method', 'functionality'))}",
                'priority': 'High',
                'category': 'Interface'
            })
        
        # Generate requirements from business logic
        for i, logic in enumerate(analysis.business_logic):
            requirements.append({
                'id': f"FR-{len(requirements) + 1}",
                'title': logic.split(':')[0],
                'description': f"The system shall {logic}",
                'priority': 'High',
                'category': 'Business Logic'
            })
        
        return requirements
    
    def _generate_technical_requirements_section(self, analysis: ComponentAnalysis) -> List[Dict[str, Any]]:
        """Generate technical requirements section"""
        requirements = []
        
        for i, req in enumerate(analysis.technical_requirements):
            requirements.append({
                'id': f"TR-{i + 1}",
                'description': req,
                'priority': 'High',
                'category': 'Technical'
            })
        
        # Add common technical requirements
        common_reqs = [
            "Logging and monitoring capabilities",
            "Error handling and recovery mechanisms",
            "Configuration management",
            "Health check endpoints"
        ]
        
        for req in common_reqs:
            requirements.append({
                'id': f"TR-{len(requirements) + 1}",
                'description': req,
                'priority': 'Medium',
                'category': 'Technical'
            })
        
        return requirements
    
    def _generate_architecture_section(self, analysis: ComponentAnalysis) -> Dict[str, Any]:
        """Generate architecture section"""
        return {
            'component_type': analysis.type,
            'dependencies': analysis.dependencies,
            'interfaces': len(analysis.interfaces),
            'data_models': len(analysis.data_models),
            'design_patterns': self._identify_design_patterns(analysis),
            'scalability_considerations': self._generate_scalability_considerations(analysis),
            'security_architecture': self._generate_security_architecture(analysis)
        }
    
    def _generate_api_specifications(self, analysis: ComponentAnalysis) -> List[Dict[str, Any]]:
        """Generate API specifications"""
        api_specs = []
        
        for interface in analysis.interfaces:
            if interface['type'] == 'REST':
                api_specs.append({
                    'endpoint': interface['path'],
                    'method': interface['method'],
                    'description': interface['description'],
                    'parameters': interface['parameters'],
                    'responses': {
                        '200': 'Success',
                        '400': 'Bad Request',
                        '500': 'Internal Server Error'
                    },
                    'example_request': self._generate_example_request(interface),
                    'example_response': self._generate_example_response(interface)
                })
        
        return api_specs
    
    def _generate_testing_strategy(self, analysis: ComponentAnalysis) -> Dict[str, Any]:
        """Generate testing strategy"""
        return {
            'unit_tests': {
                'description': "Test individual functions and methods",
                'coverage_target': "90%",
                'test_cases': len(analysis.interfaces) + len(analysis.business_logic)
            },
            'integration_tests': {
                'description': "Test component interactions",
                'focus_areas': analysis.dependencies[:5]
            },
            'performance_tests': {
                'description': "Validate performance requirements",
                'metrics': analysis.performance_requirements
            },
            'security_tests': {
                'description': "Verify security requirements",
                'areas': analysis.security_requirements
            }
        }
    
    # Helper methods for extracting information from AST
    def _extract_function_docstring(self, node: ast.FunctionDef) -> str:
        """Extract docstring from function"""
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            return node.body[0].value.s.strip()
        return ""
    
    def _extract_class_docstring(self, node: ast.ClassDef) -> str:
        """Extract docstring from class"""
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            return node.body[0].value.s.strip()
        return ""
    
    def _extract_function_parameters(self, node: ast.FunctionDef) -> List[str]:
        """Extract function parameters"""
        return [arg.arg for arg in node.args.args if arg.arg != 'self']
    
    def _extract_return_type(self, node: ast.FunctionDef) -> str:
        """Extract return type annotation"""
        if node.returns:
            return ast.dump(node.returns)
        return "Any"
    
    def _extract_class_attributes(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract class attributes"""
        attributes = []
        
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                attributes.append({
                    'name': child.target.id,
                    'type': ast.dump(child.annotation),
                    'required': True
                })
        
        return attributes
    
    def _looks_like_data_model(self, node: ast.ClassDef) -> bool:
        """Check if class looks like a data model"""
        # Heuristic: class with mostly annotated attributes
        annotations = sum(1 for child in node.body if isinstance(child, ast.AnnAssign))
        methods = sum(1 for child in node.body if isinstance(child, ast.FunctionDef))
        
        return annotations > 0 and annotations >= methods
    
    def _infer_function_purpose(self, func_name: str) -> str:
        """Infer function purpose from name"""
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', func_name)
        return f"Function to {' '.join(words).lower()}"
    
    def _deduplicate_interfaces(self, interfaces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate interfaces"""
        seen = set()
        unique_interfaces = []
        
        for interface in interfaces:
            key = (interface.get('method', ''), interface.get('path', ''), interface.get('function', ''))
            if key not in seen:
                seen.add(key)
                unique_interfaces.append(interface)
        
        return unique_interfaces
    
    def _deduplicate_data_models(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate data models"""
        seen = set()
        unique_models = []
        
        for model in models:
            if model['name'] not in seen:
                seen.add(model['name'])
                unique_models.append(model)
        
        return unique_models
    
    def _identify_design_patterns(self, analysis: ComponentAnalysis) -> List[str]:
        """Identify design patterns used"""
        patterns = []
        
        # Simple heuristics based on naming and structure
        if any('factory' in dep.lower() for dep in analysis.dependencies):
            patterns.append("Factory Pattern")
        
        if len(analysis.interfaces) > 3:
            patterns.append("Facade Pattern")
        
        if any('strategy' in logic.lower() for logic in analysis.business_logic):
            patterns.append("Strategy Pattern")
        
        return patterns
    
    def _generate_scalability_considerations(self, analysis: ComponentAnalysis) -> List[str]:
        """Generate scalability considerations"""
        considerations = []
        
        if analysis.type == 'service':
            considerations.extend([
                "Horizontal scaling with load balancers",
                "Database connection pooling",
                "Caching strategy implementation"
            ])
        
        if any('async' in req for req in analysis.performance_requirements):
            considerations.append("Asynchronous processing for improved throughput")
        
        return considerations
    
    def _generate_security_architecture(self, analysis: ComponentAnalysis) -> List[str]:
        """Generate security architecture considerations"""
        security_items = []
        
        if analysis.security_requirements:
            security_items.extend(analysis.security_requirements)
        
        # Add common security requirements
        security_items.extend([
            "Input validation and sanitization",
            "Output encoding",
            "Secure configuration management",
            "Audit logging"
        ])
        
        return list(set(security_items))
    
    def _generate_example_request(self, interface: Dict[str, Any]) -> Dict[str, Any]:
        """Generate example request for API"""
        return {
            'method': interface['method'],
            'url': interface['path'],
            'headers': {'Content-Type': 'application/json'},
            'body': {} if interface['method'] in ['POST', 'PUT'] else None
        }
    
    def _generate_example_response(self, interface: Dict[str, Any]) -> Dict[str, Any]:
        """Generate example response for API"""
        return {
            'status_code': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': {'status': 'success', 'data': {}}
        }
    
    def _generate_deployment_requirements(self, analysis: ComponentAnalysis) -> List[str]:
        """Generate deployment requirements"""
        return [
            "Containerized deployment with Docker",
            "Environment-specific configuration",
            "Health check endpoints",
            "Graceful shutdown handling",
            "Resource limits and requests",
            "Service discovery integration"
        ]
    
    def _generate_monitoring_requirements(self, analysis: ComponentAnalysis) -> List[str]:
        """Generate monitoring requirements"""
        return [
            "Application metrics collection",
            "Error rate monitoring",
            "Performance metrics tracking",
            "Business metrics dashboards",
            "Alerting for critical issues",
            "Log aggregation and analysis"
        ]
    
    def _generate_risks_and_mitigations(self, analysis: ComponentAnalysis) -> List[Dict[str, Any]]:
        """Generate risks and mitigations"""
        return [
            {
                'risk': 'Dependency failure',
                'probability': 'Medium',
                'impact': 'High',
                'mitigation': 'Implement circuit breakers and fallback mechanisms'
            },
            {
                'risk': 'Performance degradation',
                'probability': 'Medium',
                'impact': 'Medium',
                'mitigation': 'Load testing and performance monitoring'
            },
            {
                'risk': 'Security vulnerabilities',
                'probability': 'Low',
                'impact': 'High',
                'mitigation': 'Regular security audits and dependency updates'
            }
        ]
    
    def _generate_timeline(self, analysis: ComponentAnalysis) -> Dict[str, Any]:
        """Generate project timeline"""
        phases = {
            'Analysis & Design': '1 week',
            'Development': '2-3 weeks',
            'Testing': '1 week',
            'Deployment': '3 days',
            'Monitoring Setup': '2 days'
        }
        
        return {
            'phases': phases,
            'total_duration': '4-5 weeks',
            'critical_path': ['Development', 'Testing']
        }
    
    def _generate_acceptance_criteria(self, analysis: ComponentAnalysis) -> List[str]:
        """Generate acceptance criteria"""
        criteria = [
            "All functional requirements implemented and tested",
            "Performance benchmarks met",
            "Security requirements validated",
            "Documentation completed",
            "Code review passed",
            "Deployment successful in staging environment"
        ]
        
        if analysis.interfaces:
            criteria.append("All API endpoints working as specified")
        
        return criteria
    
    def _generate_integration_points(self, analysis: ComponentAnalysis) -> List[Dict[str, Any]]:
        """Generate integration points"""
        integration_points = []
        
        for dep in analysis.dependencies[:5]:
            integration_points.append({
                'component': dep,
                'type': 'dependency',
                'communication': 'API calls',
                'data_format': 'JSON'
            })
        
        return integration_points
    
    def _load_business_patterns(self) -> Dict[str, Any]:
        """Load business logic patterns"""
        return {
            'trading_patterns': ['execute', 'order', 'trade', 'portfolio'],
            'risk_patterns': ['risk', 'validation', 'limit', 'threshold'],
            'data_patterns': ['process', 'transform', 'aggregate', 'calculate']
        }
    
    def _load_technical_patterns(self) -> Dict[str, Any]:
        """Load technical patterns"""
        return {
            'service_patterns': ['FastAPI', 'Flask', 'endpoint', 'route'],
            'data_patterns': ['database', 'model', 'schema', 'repository'],
            'integration_patterns': ['client', 'adapter', 'gateway', 'proxy']
        }
    
    async def _save_prd(self, prd: PRDDocument, component_name: str) -> Path:
        """Save PRD document to file"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{component_name}_prd"
        
        if self.prd_format == 'markdown':
            output_path = self.output_dir / f"{filename}.md"
            content = self._format_prd_markdown(prd)
        elif self.prd_format == 'json':
            output_path = self.output_dir / f"{filename}.json"
            content = json.dumps(self._prd_to_dict(prd), indent=2)
        elif self.prd_format == 'yaml':
            output_path = self.output_dir / f"{filename}.yaml"
            content = yaml.dump(self._prd_to_dict(prd), default_flow_style=False)
        else:
            raise ValueError(f"Unsupported PRD format: {self.prd_format}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"PRD saved to {output_path}")
        return output_path
    
    def _format_prd_markdown(self, prd: PRDDocument) -> str:
        """Format PRD as Markdown"""
        md_content = f"""# {prd.title}

**Version:** {prd.version}  
**Author:** {prd.author}  
**Date:** {prd.date}

## Overview

{prd.overview}

## Objectives

{chr(10).join(f"- {obj}" for obj in prd.objectives)}

## Success Metrics

{chr(10).join(f"- {metric}" for metric in prd.success_metrics)}

## User Stories

{chr(10).join(f"### {story['title']} (ID: {story['id']}){chr(10)}{story['description']}{chr(10)}" for story in prd.user_stories)}

## Functional Requirements

{chr(10).join(f"### {req['title']} (ID: {req['id']}){chr(10)}{req['description']} (Priority: {req['priority']}){chr(10)}" for req in prd.functional_requirements)}

## Technical Requirements

{chr(10).join(f"- **{req['id']}**: {req['description']} (Priority: {req['priority']})" for req in prd.technical_requirements)}

## Architecture

- **Component Type**: {prd.architecture['component_type']}
- **Dependencies**: {', '.join(prd.architecture['dependencies'])}
- **Interfaces**: {prd.architecture['interfaces']}
- **Data Models**: {prd.architecture['data_models']}

## Data Models

{chr(10).join(f"### {model['name']}{chr(10)}- **Type**: {model['type']}{chr(10)}- **Description**: {model['description']}{chr(10)}" for model in prd.data_models)}

## API Specifications

{chr(10).join(f"### {api['method']} {api['endpoint']}{chr(10)}{api['description']}{chr(10)}" for api in prd.api_specifications)}

## Security Requirements

{chr(10).join(f"- {req}" for req in prd.security_requirements)}

## Performance Requirements

{chr(10).join(f"- {req}" for req in prd.performance_requirements)}

## Testing Strategy

- **Unit Tests**: {prd.testing_strategy['unit_tests']['description']} (Target: {prd.testing_strategy['unit_tests']['coverage_target']})
- **Integration Tests**: {prd.testing_strategy['integration_tests']['description']}
- **Performance Tests**: {prd.testing_strategy['performance_tests']['description']}
- **Security Tests**: {prd.testing_strategy['security_tests']['description']}

## Deployment Requirements

{chr(10).join(f"- {req}" for req in prd.deployment_requirements)}

## Monitoring Requirements

{chr(10).join(f"- {req}" for req in prd.monitoring_requirements)}

## Risks and Mitigations

{chr(10).join(f"### {risk['risk']}{chr(10)}- **Probability**: {risk['probability']}{chr(10)}- **Impact**: {risk['impact']}{chr(10)}- **Mitigation**: {risk['mitigation']}{chr(10)}" for risk in prd.risks_and_mitigations)}

## Timeline

**Total Duration**: {prd.timeline['total_duration']}

{chr(10).join(f"- **{phase}**: {duration}" for phase, duration in prd.timeline['phases'].items())}

## Acceptance Criteria

{chr(10).join(f"- {criteria}" for criteria in prd.acceptance_criteria)}

---

*This PRD was automatically generated by the System Architect Agent*
"""
        return md_content
    
    def _prd_to_dict(self, prd: PRDDocument) -> Dict[str, Any]:
        """Convert PRD to dictionary"""
        return {
            'title': prd.title,
            'version': prd.version,
            'author': prd.author,
            'date': prd.date,
            'overview': prd.overview,
            'objectives': prd.objectives,
            'success_metrics': prd.success_metrics,
            'user_stories': prd.user_stories,
            'functional_requirements': prd.functional_requirements,
            'technical_requirements': prd.technical_requirements,
            'architecture': prd.architecture,
            'data_models': prd.data_models,
            'api_specifications': prd.api_specifications,
            'security_requirements': prd.security_requirements,
            'performance_requirements': prd.performance_requirements,
            'integration_points': prd.integration_points,
            'testing_strategy': prd.testing_strategy,
            'deployment_requirements': prd.deployment_requirements,
            'monitoring_requirements': prd.monitoring_requirements,
            'risks_and_mitigations': prd.risks_and_mitigations,
            'timeline': prd.timeline,
            'acceptance_criteria': prd.acceptance_criteria
        }
    
    def _analysis_to_dict(self, analysis: ComponentAnalysis) -> Dict[str, Any]:
        """Convert analysis to dictionary"""
        return {
            'name': analysis.name,
            'type': analysis.type,
            'purpose': analysis.purpose,
            'dependencies': analysis.dependencies,
            'interfaces': analysis.interfaces,
            'data_models': analysis.data_models,
            'business_logic': analysis.business_logic,
            'technical_requirements': analysis.technical_requirements,
            'performance_requirements': analysis.performance_requirements,
            'security_requirements': analysis.security_requirements,
            'file_paths': analysis.file_paths
        }