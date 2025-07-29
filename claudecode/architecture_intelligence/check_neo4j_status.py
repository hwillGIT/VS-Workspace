#!/usr/bin/env python3
"""
Check Neo4j Desktop Status
"""

import socket
import subprocess
import sys

def check_port(host='localhost', port=7687):
    """Check if Neo4j port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def check_neo4j_browser():
    """Check if Neo4j Browser is accessible"""
    try:
        import urllib.request
        response = urllib.request.urlopen('http://localhost:7474', timeout=5)
        return response.getcode() == 200
    except:
        return False

def main():
    print("Neo4j Desktop Status Check")
    print("=" * 30)
    
    # Check Bolt port (7687)
    bolt_open = check_port('localhost', 7687)
    print(f"Bolt Port (7687): {'OPEN' if bolt_open else 'CLOSED'}")
    
    # Check HTTP port (7474)
    http_open = check_port('localhost', 7474)
    print(f"HTTP Port (7474): {'OPEN' if http_open else 'CLOSED'}")
    
    # Check Neo4j Browser
    browser_accessible = check_neo4j_browser()
    print(f"Neo4j Browser: {'ACCESSIBLE' if browser_accessible else 'NOT ACCESSIBLE'}")
    
    print("\nDiagnosis:")
    if not bolt_open and not http_open:
        print("Neo4j is NOT running")
        print("\nAction Required:")
        print("1. Open Neo4j Desktop")
        print("2. Click on your database")
        print("3. Click the green START button")
        print("4. Wait for it to show 'Active'")
    elif http_open and not bolt_open:
        print("Neo4j HTTP is running but Bolt is not")
        print("This is unusual - check Neo4j Desktop logs")
    else:
        print("Neo4j appears to be running")
        print("Check your password in Neo4j Desktop matches 'architecture123'")
    
    if browser_accessible:
        print("\nYou can access Neo4j Browser at: http://localhost:7474")

if __name__ == "__main__":
    main()