import requests
import json
import time
from typing import List, Dict
from pathlib import Path


class SonarQubeDataFetcher:
    """
    Fetch issues and metrics from SonarQube API for multiple projects.
    """
    
    def __init__(self, sonarqube_url: str, token: str = None):
        """
        Initialize SonarQube API client.
        
        Args:
            sonarqube_url: Base URL of your SonarQube instance (e.g., 'http://localhost:9000')
            token: SonarQube authentication token (optional if anonymous access is enabled)
        """
        self.base_url = sonarqube_url.rstrip('/')
        self.token = token
        self.session = requests.Session()
        
        if token:
            # Use token authentication
            self.session.auth = (token, '')
        
    def get_all_projects(self) -> List[Dict]:
        """Retrieve all projects from SonarQube."""
        url = f"{self.base_url}/api/components/search"
        params = {
            'qualifiers': 'TRK',  # TRK = projects
            'ps': 500  # page size
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            projects = data.get('components', [])
            print(f"Found {len(projects)} projects in SonarQube")
            
            for project in projects:
                print(f"  - {project['key']}: {project['name']}")
            
            return projects
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching projects: {e}")
            return []
    
    def get_project_issues(self, project_key: str, page_size: int = 500) -> List[Dict]:
        """
        Retrieve all issues for a specific project.
        Handles SonarQube's 10,000 issue limit by splitting queries by severity and type.
        
        Args:
            project_key: SonarQube project key
            page_size: Number of issues per page (max 500)
            
        Returns:
            List of issue dictionaries
        """
        all_issues = []
        
        # SonarQube has a 10k limit, so we split by severities and types
        severities = ['BLOCKER', 'CRITICAL', 'MAJOR', 'MINOR', 'INFO']
        types = ['BUG', 'VULNERABILITY', 'CODE_SMELL', 'SECURITY_HOTSPOT']
        
        # Try to get all issues first
        print(f"  Attempting to fetch all issues...")
        issues = self._fetch_issues_with_filter(project_key, {}, page_size)
        
        if len(issues) < 10000:
            # We got all issues without hitting the limit
            return issues
        
        # Hit the 10k limit - split by severity and type
        print(f"  Hit 10k limit. Splitting by severity and type...")
        all_issues = []
        issue_keys_seen = set()
        
        for severity in severities:
            for issue_type in types:
                print(f"    Fetching {severity} {issue_type} issues...")
                filters = {
                    'severities': severity,
                    'types': issue_type
                }
                
                issues = self._fetch_issues_with_filter(project_key, filters, page_size)
                
                # Deduplicate based on issue key
                new_issues = [i for i in issues if i['key'] not in issue_keys_seen]
                for issue in new_issues:
                    issue_keys_seen.add(issue['key'])
                
                all_issues.extend(new_issues)
                print(f"      Retrieved {len(new_issues)} new issues (Total unique: {len(all_issues)})")
        
        return all_issues
    
    def _fetch_issues_with_filter(self, project_key: str, filters: Dict, page_size: int = 500) -> List[Dict]:
        """
        Fetch issues with specific filters.
        
        Args:
            project_key: SonarQube project key
            filters: Dictionary of filter parameters (severities, types, etc.)
            page_size: Number of issues per page
            
        Returns:
            List of issues
        """
        url = f"{self.base_url}/api/issues/search"
        all_issues = []
        page = 1
        max_pages = 20  # 10k / 500 = 20 pages max
        
        while page <= max_pages:
            params = {
                'componentKeys': project_key,
                'ps': page_size,
                'p': page,
                'resolved': 'false',
            }
            params.update(filters)
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                issues = data.get('issues', [])
                all_issues.extend(issues)
                
                total = data.get('total', 0)
                
                # Check if there are more pages
                if len(all_issues) >= total or len(issues) == 0:
                    break
                
                page += 1
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                if '400' in str(e):
                    # Hit the 10k limit
                    print(f"      Hit 10k limit at page {page}")
                else:
                    print(f"      Error at page {page}: {e}")
                break
        
        return all_issues
    
    def get_project_metrics(self, project_key: str) -> Dict:
        """
        Get project-level metrics.
        
        Args:
            project_key: SonarQube project key
            
        Returns:
            Dictionary of metrics
        """
        url = f"{self.base_url}/api/measures/component"
        
        # Common metrics to fetch
        metric_keys = [
            'bugs',
            'vulnerabilities',
            'code_smells',
            'security_hotspots',
            'sqale_index',  # Technical debt in minutes
            'sqale_debt_ratio',
            'reliability_rating',
            'security_rating',
            'sqale_rating',  # Maintainability rating
            'coverage',
            'duplicated_lines_density',
            'ncloc',  # Lines of code
            'complexity',
        ]
        
        params = {
            'component': project_key,
            'metricKeys': ','.join(metric_keys)
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            component = data.get('component', {})
            measures = component.get('measures', [])
            
            # Convert to dictionary
            metrics = {}
            for measure in measures:
                metrics[measure['metric']] = measure.get('value')
            
            return metrics
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching metrics for {project_key}: {e}")
            return {}
    
    def fetch_all_data(self, project_keys: List[str] = None, output_file: str = 'sonarqube_data.json'):
        """
        Fetch all issues and metrics for specified projects and save to JSON.
        
        Args:
            project_keys: List of project keys to fetch (if None, fetches all projects)
            output_file: Output JSON file path
        """
        # If no project keys specified, get all projects
        if project_keys is None:
            projects = self.get_all_projects()
            project_keys = [p['key'] for p in projects]
        
        all_data = {
            'projects': {},
            'issues': [],
            'metadata': {
                'sonarqube_url': self.base_url,
                'total_projects': len(project_keys),
                'fetch_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        for project_key in project_keys:
            print(f"\nProcessing project: {project_key}")
            
            # Get issues
            issues = self.get_project_issues(project_key)
            
            # Get metrics
            metrics = self.get_project_metrics(project_key)
            
            # Store project data
            all_data['projects'][project_key] = {
                'metrics': metrics,
                'issue_count': len(issues)
            }
            
            # Add project key to each issue for easy filtering
            for issue in issues:
                issue['project_key'] = project_key
            
            all_data['issues'].extend(issues)
            
            print(f"  Retrieved {len(issues)} issues and {len(metrics)} metrics")
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Data saved to {output_file}")
        print(f"Total projects: {len(project_keys)}")
        print(f"Total issues: {len(all_data['issues'])}")
        print(f"{'='*60}")
        
        return all_data
    
    def fetch_data_for_satd_projects(self, satd_file: str, output_file: str = 'sonarqube_data.json'):
        """
        Fetch SonarQube data only for projects that appear in your SATD dataset.
        
        Args:
            satd_file: Path to SATD JSONL file
            output_file: Output JSON file path
        """
        # Extract unique projects from SATD data
        projects = set()
        with open(satd_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Extract project key from repo_name (e.g., "ornladios/ADIOS2")
                repo_name = data.get('repo_name', '')
                if repo_name:
                    projects.add(repo_name)
        
        print(f"Found {len(projects)} unique projects in SATD data:")
        for project in sorted(projects):
            print(f"  - {project}")
        
        # Map GitHub repo names to SonarQube project keys
        # SonarQube might use different naming conventions
        # You may need to manually map these
        print("\nAttempting to fetch data for these projects...")
        print("Note: If project keys don't match, you may need to manually specify them")
        
        return self.fetch_all_data(project_keys=list(projects), output_file=output_file)


# Example usage
if __name__ == "__main__":
    # CONFIGURATION 
    SONARQUBE_URL = "http://localhost:9000"
    SONARQUBE_TOKEN = "" 
    SATD_FILE = "fixed_code_comments_with_priority.jsonl" 
    
    # Initialize fetcher
    fetcher = SonarQubeDataFetcher(SONARQUBE_URL, SONARQUBE_TOKEN)
    
    # Option 1: Fetch all projects in SonarQube
    print("Option 1: Fetching all SonarQube projects...")
    fetcher.fetch_all_data(output_file='sonarqube_all_projects.json')
    
    # Option 2: Fetch only projects from your SATD dataset
    # print("\nOption 2: Fetching only projects in SATD dataset...")
    # fetcher.fetch_data_for_satd_projects(
    #     satd_file=SATD_FILE,
    #     output_file='sonarqube_satd_projects.json'
    # )
    
    # Option 3: Fetch specific projects manually
    # specific_projects = [
    #     'ornladios/ADIOS2',
    #     'your-org/your-project',
    # ]
    # fetcher.fetch_all_data(project_keys=specific_projects, output_file='sonarqube_specific.json')
    
    print("\nDone!")