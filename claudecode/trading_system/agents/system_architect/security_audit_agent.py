"""
Security Audit Agent

This agent performs comprehensive security vulnerability scanning and provides
automated remediation recommendations for trading system code.
"""

import ast
import re
import hashlib
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from ...core.base.agent import BaseAgent


@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability"""
    category: str  # e.g., 'injection', 'crypto', 'auth', 'data_exposure'
    severity: str  # 'low', 'medium', 'high', 'critical'
    file_path: str
    line_number: int
    function_or_class: str
    description: str
    cwe_id: Optional[str]  # Common Weakness Enumeration ID
    owasp_category: Optional[str]
    proof_of_concept: Optional[str]
    remediation: str
    automated_fix_available: bool
    confidence: float  # 0.0 to 1.0


@dataclass
class SecurityReport:
    """Comprehensive security audit report"""
    overall_score: float
    vulnerabilities: List[SecurityVulnerability]
    risk_score: float
    compliance_status: Dict[str, bool]
    recommendations: List[str]
    scan_metadata: Dict[str, Any]


class SecurityAuditAgent(BaseAgent):
    """
    Security Audit Agent
    
    Performs comprehensive security analysis including:
    - Code injection vulnerabilities
    - Cryptographic weaknesses
    - Authentication/Authorization flaws
    - Data exposure risks
    - Input validation issues
    - Configuration security
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("SecurityAudit", config.get('security_audit', {}))
        self.logger = logging.getLogger(__name__)
        
        # Security patterns and rules
        self.injection_patterns = self._load_injection_patterns()
        self.crypto_patterns = self._load_crypto_patterns()
        self.auth_patterns = self._load_auth_patterns()
        self.data_exposure_patterns = self._load_data_exposure_patterns()
        
        # Configuration
        self.scan_depth = config.get('scan_depth', 'deep')
        self.include_low_confidence = config.get('include_low_confidence', False)
        self.compliance_standards = config.get('compliance_standards', ['OWASP', 'PCI-DSS'])
        
    async def audit_security(self, target_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive security audit
        
        Args:
            target_path: Path to audit
            
        Returns:
            Security audit results
        """
        self.logger.info(f"Starting security audit of {target_path}")
        
        vulnerabilities = []
        path = Path(target_path)
        
        # Scan files
        if path.is_file() and path.suffix == '.py':
            vulnerabilities.extend(await self._scan_file(path))
        elif path.is_dir():
            for py_file in path.rglob('*.py'):
                if not self._should_skip_file(py_file):
                    vulnerabilities.extend(await self._scan_file(py_file))
        
        # Filter by confidence if needed
        if not self.include_low_confidence:
            vulnerabilities = [v for v in vulnerabilities if v.confidence >= 0.7]
        
        # Generate report
        report = self._generate_security_report(vulnerabilities)
        
        return {
            'overall_score': report.overall_score,
            'risk_score': report.risk_score,
            'vulnerabilities': [self._vulnerability_to_dict(v) for v in vulnerabilities],
            'compliance_status': report.compliance_status,
            'recommendations': report.recommendations,
            'critical_issues': [self._vulnerability_to_dict(v) for v in vulnerabilities if v.severity == 'critical'],
            'refactoring_priorities': self._prioritize_security_fixes(vulnerabilities),
            'scan_metadata': report.scan_metadata
        }
    
    async def _scan_file(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan a single file for security vulnerabilities"""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Static analysis
            vulnerabilities.extend(self._scan_injection_vulnerabilities(file_path, content))
            vulnerabilities.extend(self._scan_crypto_vulnerabilities(file_path, content))
            vulnerabilities.extend(self._scan_auth_vulnerabilities(file_path, content))
            vulnerabilities.extend(self._scan_data_exposure(file_path, content))
            vulnerabilities.extend(self._scan_input_validation(file_path, content))
            vulnerabilities.extend(self._scan_configuration_issues(file_path, content))
            
            # AST-based analysis
            try:
                tree = ast.parse(content)
                vulnerabilities.extend(self._ast_security_analysis(tree, file_path, content))
            except SyntaxError:
                self.logger.warning(f"Could not parse {file_path} for AST analysis")
                
        except Exception as e:
            self.logger.error(f"Error scanning file {file_path}: {e}")
        
        return vulnerabilities
    
    def _scan_injection_vulnerabilities(self, file_path: Path, content: str) -> List[SecurityVulnerability]:
        """Scan for injection vulnerabilities"""
        vulnerabilities = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            # SQL Injection patterns
            for pattern in self.injection_patterns['sql']:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(SecurityVulnerability(
                        category='injection',
                        severity='high',
                        file_path=str(file_path),
                        line_number=i,
                        function_or_class=self._extract_function_name(lines, i),
                        description=f"Potential SQL injection vulnerability detected",
                        cwe_id='CWE-89',
                        owasp_category='A03:2021 – Injection',
                        proof_of_concept=f"Line contains SQL injection pattern: {line.strip()}",
                        remediation="Use parameterized queries or ORM methods instead of string concatenation",
                        automated_fix_available=True,
                        confidence=0.8
                    ))
            
            # Command Injection patterns
            for pattern in self.injection_patterns['command']:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(SecurityVulnerability(
                        category='injection',
                        severity='critical',
                        file_path=str(file_path),
                        line_number=i,
                        function_or_class=self._extract_function_name(lines, i),
                        description=f"Potential command injection vulnerability detected",
                        cwe_id='CWE-78',
                        owasp_category='A03:2021 – Injection',
                        proof_of_concept=f"Line contains command injection pattern: {line.strip()}",
                        remediation="Use subprocess with shell=False and validate all inputs",
                        automated_fix_available=True,
                        confidence=0.9
                    ))
            
            # LDAP Injection patterns
            for pattern in self.injection_patterns['ldap']:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(SecurityVulnerability(
                        category='injection',
                        severity='medium',
                        file_path=str(file_path),
                        line_number=i,
                        function_or_class=self._extract_function_name(lines, i),
                        description=f"Potential LDAP injection vulnerability detected",
                        cwe_id='CWE-90',
                        owasp_category='A03:2021 – Injection',
                        proof_of_concept=f"Line contains LDAP injection pattern: {line.strip()}",
                        remediation="Escape LDAP special characters and use parameterized queries",
                        automated_fix_available=False,
                        confidence=0.7
                    ))
        
        return vulnerabilities
    
    def _scan_crypto_vulnerabilities(self, file_path: Path, content: str) -> List[SecurityVulnerability]:
        """Scan for cryptographic vulnerabilities"""
        vulnerabilities = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Weak encryption algorithms
            for weak_algo in self.crypto_patterns['weak_algorithms']:
                if weak_algo in line.lower():
                    vulnerabilities.append(SecurityVulnerability(
                        category='crypto',
                        severity='high',
                        file_path=str(file_path),
                        line_number=i,
                        function_or_class=self._extract_function_name(lines, i),
                        description=f"Weak cryptographic algorithm detected: {weak_algo}",
                        cwe_id='CWE-327',
                        owasp_category='A02:2021 – Cryptographic Failures',
                        proof_of_concept=f"Line uses weak algorithm: {line.strip()}",
                        remediation=f"Replace {weak_algo} with AES-256-GCM or other strong algorithms",
                        automated_fix_available=True,
                        confidence=0.9
                    ))
            
            # Hardcoded secrets
            if re.search(r'(password|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']', line, re.IGNORECASE):
                vulnerabilities.append(SecurityVulnerability(
                    category='crypto',
                    severity='critical',
                    file_path=str(file_path),
                    line_number=i,
                    function_or_class=self._extract_function_name(lines, i),
                    description="Hardcoded secret detected",
                    cwe_id='CWE-798',
                    owasp_category='A07:2021 – Identification and Authentication Failures',
                    proof_of_concept=f"Line contains hardcoded secret: {line.strip()[:50]}...",
                    remediation="Use environment variables or secure key management systems",
                    automated_fix_available=True,
                    confidence=0.85
                ))
            
            # Weak random number generation
            if re.search(r'random\.(random|randint|choice)', line):
                vulnerabilities.append(SecurityVulnerability(
                    category='crypto',
                    severity='medium',
                    file_path=str(file_path),
                    line_number=i,
                    function_or_class=self._extract_function_name(lines, i),
                    description="Use of weak random number generator",
                    cwe_id='CWE-338',
                    owasp_category='A02:2021 – Cryptographic Failures',
                    proof_of_concept=f"Line uses weak RNG: {line.strip()}",
                    remediation="Use secrets.SystemRandom() or os.urandom() for cryptographic purposes",
                    automated_fix_available=True,
                    confidence=0.8
                ))
        
        return vulnerabilities
    
    def _scan_auth_vulnerabilities(self, file_path: Path, content: str) -> List[SecurityVulnerability]:
        """Scan for authentication and authorization vulnerabilities"""
        vulnerabilities = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Missing authentication
            if re.search(r'@app\.route.*methods.*POST', line) and i + 5 < len(lines):
                # Check next few lines for authentication
                auth_found = False
                for j in range(i, min(i + 5, len(lines))):
                    if any(auth_term in lines[j].lower() for auth_term in ['authenticate', 'login', 'token', 'session']):
                        auth_found = True
                        break
                
                if not auth_found:
                    vulnerabilities.append(SecurityVulnerability(
                        category='auth',
                        severity='high',
                        file_path=str(file_path),
                        line_number=i,
                        function_or_class=self._extract_function_name(lines, i),
                        description="HTTP endpoint may lack authentication",
                        cwe_id='CWE-306',
                        owasp_category='A07:2021 – Identification and Authentication Failures',
                        proof_of_concept=f"POST endpoint without visible authentication: {line.strip()}",
                        remediation="Add authentication middleware or decorators",
                        automated_fix_available=False,
                        confidence=0.6
                    ))
            
            # Weak session management
            if 'session[' in line and 'secure=false' in line.lower():
                vulnerabilities.append(SecurityVulnerability(
                    category='auth',
                    severity='medium',
                    file_path=str(file_path),
                    line_number=i,
                    function_or_class=self._extract_function_name(lines, i),
                    description="Insecure session configuration",
                    cwe_id='CWE-614',
                    owasp_category='A07:2021 – Identification and Authentication Failures',
                    proof_of_concept=f"Session not marked as secure: {line.strip()}",
                    remediation="Set secure=True and httponly=True for session cookies",
                    automated_fix_available=True,
                    confidence=0.9
                ))
        
        return vulnerabilities
    
    def _scan_data_exposure(self, file_path: Path, content: str) -> List[SecurityVulnerability]:
        """Scan for data exposure vulnerabilities"""
        vulnerabilities = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Debug mode in production
            if re.search(r'debug\s*=\s*true', line, re.IGNORECASE):
                vulnerabilities.append(SecurityVulnerability(
                    category='data_exposure',
                    severity='medium',
                    file_path=str(file_path),
                    line_number=i,
                    function_or_class=self._extract_function_name(lines, i),
                    description="Debug mode enabled",
                    cwe_id='CWE-489',
                    owasp_category='A05:2021 – Security Misconfiguration',
                    proof_of_concept=f"Debug mode enabled: {line.strip()}",
                    remediation="Disable debug mode in production environments",
                    automated_fix_available=True,
                    confidence=0.8
                ))
            
            # Sensitive data in logs
            if re.search(r'log.*\.(password|secret|key|token|credit)', line, re.IGNORECASE):
                vulnerabilities.append(SecurityVulnerability(
                    category='data_exposure',
                    severity='high',
                    file_path=str(file_path),
                    line_number=i,
                    function_or_class=self._extract_function_name(lines, i),
                    description="Sensitive data may be logged",
                    cwe_id='CWE-532',
                    owasp_category='A09:2021 – Security Logging and Monitoring Failures',
                    proof_of_concept=f"Sensitive data in logs: {line.strip()}",
                    remediation="Sanitize sensitive data before logging",
                    automated_fix_available=True,
                    confidence=0.7
                ))
        
        return vulnerabilities
    
    def _scan_input_validation(self, file_path: Path, content: str) -> List[SecurityVulnerability]:
        """Scan for input validation vulnerabilities"""
        vulnerabilities = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Missing input validation on user inputs
            if re.search(r'request\.(args|form|json)\[', line):
                # Check if validation is present nearby
                validation_found = False
                for j in range(max(0, i-3), min(i+3, len(lines))):
                    if any(val_term in lines[j].lower() for val_term in ['validate', 'sanitize', 'escape', 'isinstance']):
                        validation_found = True
                        break
                
                if not validation_found:
                    vulnerabilities.append(SecurityVulnerability(
                        category='input_validation',
                        severity='medium',
                        file_path=str(file_path),
                        line_number=i,
                        function_or_class=self._extract_function_name(lines, i),
                        description="User input lacks validation",
                        cwe_id='CWE-20',
                        owasp_category='A03:2021 – Injection',
                        proof_of_concept=f"Unvalidated user input: {line.strip()}",
                        remediation="Add input validation and sanitization",
                        automated_fix_available=False,
                        confidence=0.6
                    ))
        
        return vulnerabilities
    
    def _scan_configuration_issues(self, file_path: Path, content: str) -> List[SecurityVulnerability]:
        """Scan for security configuration issues"""
        vulnerabilities = []
        
        # Check for insecure SSL/TLS settings
        if 'ssl_context' in content and 'verify_mode=ssl.CERT_NONE' in content:
            vulnerabilities.append(SecurityVulnerability(
                category='configuration',
                severity='high',
                file_path=str(file_path),
                line_number=1,
                function_or_class='module',
                description="SSL certificate verification disabled",
                cwe_id='CWE-295',
                owasp_category='A07:2021 – Identification and Authentication Failures',
                proof_of_concept="SSL verification disabled",
                remediation="Enable SSL certificate verification",
                automated_fix_available=True,
                confidence=0.9
            ))
        
        return vulnerabilities
    
    def _ast_security_analysis(self, tree: ast.AST, file_path: Path, content: str) -> List[SecurityVulnerability]:
        """Perform AST-based security analysis"""
        vulnerabilities = []
        
        for node in ast.walk(tree):
            # Check for eval/exec usage
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ['eval', 'exec', 'compile']:
                    vulnerabilities.append(SecurityVulnerability(
                        category='injection',
                        severity='critical',
                        file_path=str(file_path),
                        line_number=node.lineno,
                        function_or_class='',
                        description=f"Dangerous function '{node.func.id}' detected",
                        cwe_id='CWE-94',
                        owasp_category='A03:2021 – Injection',
                        proof_of_concept=f"Use of {node.func.id} function",
                        remediation=f"Avoid using {node.func.id} or implement strict input validation",
                        automated_fix_available=False,
                        confidence=0.95
                    ))
            
            # Check for pickle usage with untrusted data
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (hasattr(node.func.value, 'id') and node.func.value.id == 'pickle' and 
                    node.func.attr in ['load', 'loads']):
                    vulnerabilities.append(SecurityVulnerability(
                        category='injection',
                        severity='high',
                        file_path=str(file_path),
                        line_number=node.lineno,
                        function_or_class='',
                        description="Unsafe pickle deserialization detected",
                        cwe_id='CWE-502',
                        owasp_category='A08:2021 – Software and Data Integrity Failures',
                        proof_of_concept="pickle.load/loads can execute arbitrary code",
                        remediation="Use json or implement safe deserialization",
                        automated_fix_available=True,
                        confidence=0.8
                    ))
        
        return vulnerabilities
    
    def _load_injection_patterns(self) -> Dict[str, List[str]]:
        """Load injection vulnerability patterns"""
        return {
            'sql': [
                r'.*\+.*["\'].*SELECT.*FROM.*["\']',
                r'.*\+.*["\'].*INSERT.*INTO.*["\']',
                r'.*\+.*["\'].*UPDATE.*SET.*["\']',
                r'.*\+.*["\'].*DELETE.*FROM.*["\']',
                r'cursor\.execute\(["\'][^"\']*["\'].*\+',
                r'\.format\(.*\).*SELECT.*FROM',
            ],
            'command': [
                r'os\.system\(.*\+',
                r'subprocess\.(call|run|Popen).*shell=True',
                r'exec\(.*input\(',
                r'eval\(.*input\(',
            ],
            'ldap': [
                r'ldap.*search.*\+',
                r'ldap.*filter.*\+',
            ]
        }
    
    def _load_crypto_patterns(self) -> Dict[str, List[str]]:
        """Load cryptographic vulnerability patterns"""
        return {
            'weak_algorithms': [
                'md5', 'sha1', 'des', 'rc4', 'md4',
                'blowfish', '3des', 'tripledes'
            ],
            'weak_modes': [
                'ecb', 'cbc_without_iv'
            ]
        }
    
    def _load_auth_patterns(self) -> Dict[str, List[str]]:
        """Load authentication vulnerability patterns"""
        return {
            'weak_auth': [
                'authenticate.*==.*password',
                'login.*==.*hardcoded',
            ]
        }
    
    def _load_data_exposure_patterns(self) -> Dict[str, List[str]]:
        """Load data exposure vulnerability patterns"""
        return {
            'sensitive_data': [
                'password', 'secret', 'key', 'token',
                'credit_card', 'ssn', 'social_security'
            ]
        }
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during scan"""
        skip_patterns = [
            '__pycache__',
            '.git',
            'test_',
            '_test.py',
            'tests/',
            'venv/',
            '.venv/',
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def _extract_function_name(self, lines: List[str], line_num: int) -> str:
        """Extract function name from context"""
        for i in range(line_num - 1, max(0, line_num - 20), -1):
            line = lines[i].strip()
            if line.startswith('def ') or line.startswith('class '):
                return line.split('(')[0].replace('def ', '').replace('class ', '')
        return 'unknown'
    
    def _generate_security_report(self, vulnerabilities: List[SecurityVulnerability]) -> SecurityReport:
        """Generate comprehensive security report"""
        # Calculate scores
        severity_weights = {'low': 1, 'medium': 3, 'high': 7, 'critical': 10}
        total_score = sum(severity_weights.get(v.severity, 0) for v in vulnerabilities)
        max_possible_score = len(vulnerabilities) * 10 if vulnerabilities else 10
        
        overall_score = max(0, 10 - (total_score / max_possible_score * 10))
        risk_score = min(10, total_score / 10)
        
        # Check compliance
        compliance_status = {
            'OWASP': self._check_owasp_compliance(vulnerabilities),
            'PCI-DSS': self._check_pci_compliance(vulnerabilities),
        }
        
        # Generate recommendations
        recommendations = self._generate_security_recommendations(vulnerabilities)
        
        scan_metadata = {
            'scan_date': datetime.utcnow().isoformat(),
            'total_vulnerabilities': len(vulnerabilities),
            'by_severity': {
                'critical': len([v for v in vulnerabilities if v.severity == 'critical']),
                'high': len([v for v in vulnerabilities if v.severity == 'high']),
                'medium': len([v for v in vulnerabilities if v.severity == 'medium']),
                'low': len([v for v in vulnerabilities if v.severity == 'low']),
            }
        }
        
        return SecurityReport(
            overall_score=overall_score,
            vulnerabilities=vulnerabilities,
            risk_score=risk_score,
            compliance_status=compliance_status,
            recommendations=recommendations,
            scan_metadata=scan_metadata
        )
    
    def _check_owasp_compliance(self, vulnerabilities: List[SecurityVulnerability]) -> bool:
        """Check OWASP Top 10 compliance"""
        critical_high = [v for v in vulnerabilities if v.severity in ['critical', 'high']]
        return len(critical_high) == 0
    
    def _check_pci_compliance(self, vulnerabilities: List[SecurityVulnerability]) -> bool:
        """Check PCI-DSS compliance"""
        crypto_vulns = [v for v in vulnerabilities if v.category == 'crypto']
        return len(crypto_vulns) == 0
    
    def _generate_security_recommendations(self, vulnerabilities: List[SecurityVulnerability]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Group by category
        by_category = {}
        for vuln in vulnerabilities:
            if vuln.category not in by_category:
                by_category[vuln.category] = []
            by_category[vuln.category].append(vuln)
        
        for category, vulns in by_category.items():
            count = len(vulns)
            if category == 'injection':
                recommendations.append(f"Fix {count} injection vulnerabilities by using parameterized queries")
            elif category == 'crypto':
                recommendations.append(f"Address {count} cryptographic issues by updating algorithms and key management")
            elif category == 'auth':
                recommendations.append(f"Strengthen authentication for {count} identified weaknesses")
            elif category == 'data_exposure':
                recommendations.append(f"Prevent data exposure in {count} locations")
        
        return recommendations
    
    def _prioritize_security_fixes(self, vulnerabilities: List[SecurityVulnerability]) -> List[Dict[str, Any]]:
        """Prioritize security fixes"""
        priorities = []
        
        for vuln in vulnerabilities:
            impact = {'critical': 10, 'high': 8, 'medium': 5, 'low': 2}[vuln.severity]
            effort = 3 if vuln.automated_fix_available else 7
            
            priorities.append({
                'type': 'security_fix',
                'target': f"{vuln.file_path}:{vuln.line_number}",
                'description': vuln.description,
                'category': vuln.category,
                'severity': vuln.severity,
                'impact': impact,
                'effort': effort,
                'automated_fix': vuln.automated_fix_available,
                'cwe_id': vuln.cwe_id,
                'remediation': vuln.remediation
            })
        
        return sorted(priorities, key=lambda x: (x['impact'], -x['effort']), reverse=True)
    
    def _vulnerability_to_dict(self, vuln: SecurityVulnerability) -> Dict[str, Any]:
        """Convert vulnerability to dictionary"""
        return {
            'category': vuln.category,
            'severity': vuln.severity,
            'file_path': vuln.file_path,
            'line_number': vuln.line_number,
            'function_or_class': vuln.function_or_class,
            'description': vuln.description,
            'cwe_id': vuln.cwe_id,
            'owasp_category': vuln.owasp_category,
            'proof_of_concept': vuln.proof_of_concept,
            'remediation': vuln.remediation,
            'automated_fix_available': vuln.automated_fix_available,
            'confidence': vuln.confidence
        }
    
    async def fix_security_issue(self, target: str, issue_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to fix a security issue automatically
        
        Args:
            target: File path and line number (file:line)
            issue_info: Information about the security issue
            
        Returns:
            Result of the fix attempt
        """
        self.logger.info(f"Attempting to fix security issue: {issue_info.get('description')}")
        
        if not issue_info.get('automated_fix'):
            return {
                'success': False,
                'message': "Automated fix not available for this issue",
                'manual_steps': issue_info.get('remediation', 'Manual review required')
            }
        
        category = issue_info.get('category')
        file_path, line_num = target.split(':')
        
        try:
            if category == 'crypto' and 'hardcoded' in issue_info.get('description', ''):
                return await self._fix_hardcoded_secret(file_path, int(line_num))
            elif category == 'crypto' and 'weak' in issue_info.get('description', ''):
                return await self._fix_weak_crypto(file_path, int(line_num))
            else:
                return {
                    'success': False,
                    'message': f"Automated fix for {category} not yet implemented",
                    'manual_steps': issue_info.get('remediation', 'Manual review required')
                }
        except Exception as e:
            return {
                'success': False,
                'message': f"Error applying fix: {str(e)}",
                'manual_steps': issue_info.get('remediation', 'Manual review required')
            }
    
    async def _fix_hardcoded_secret(self, file_path: str, line_num: int) -> Dict[str, Any]:
        """Fix hardcoded secret by suggesting environment variable"""
        return {
            'success': False,
            'message': "Hardcoded secret fix requires manual intervention",
            'manual_steps': "Replace hardcoded secret with os.environ.get('SECRET_NAME') and add to environment"
        }
    
    async def _fix_weak_crypto(self, file_path: str, line_num: int) -> Dict[str, Any]:
        """Fix weak cryptography by suggesting stronger alternatives"""
        return {
            'success': False,
            'message': "Weak crypto fix requires manual review",
            'manual_steps': "Replace with AES-256-GCM or other NIST-approved algorithms"
        }