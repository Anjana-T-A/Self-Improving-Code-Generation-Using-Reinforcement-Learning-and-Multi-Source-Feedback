# import os
# import requests

# SONARQUBE_URL = "http://localhost:9000"
# SONARQUBE_TOKEN = "squ_2101ef51cb5fe6f9a51fd92010e72647f3c0c569"
# SONARQUBE_PROJECT = "sonarqube"

# def analyze_code_with_sonarqube(code_snippet, file_name="generated_code.py"):
#     with open(file_name, 'w') as f:
#         f.write(code_snippet)
#     os.system(f"sonar-scanner -Dsonar.projectKey={SONARQUBE_PROJECT} -Dsonar.sources=. "
#               f"-Dsonar.host.url={SONARQUBE_URL} -Dsonar.login={SONARQUBE_TOKEN} "
#               f"-Dsonar.inclusions={file_name}")
#     url = f"{SONARQUBE_URL}/api/measures/component"
#     params = {
#         "component": SONARQUBE_PROJECT,
#         "metricKeys": "reliability_rating,security_rating,sqale_rating"
#     }
#     response = requests.get(url, auth=(SONARQUBE_TOKEN, ''))
#     if response.status_code != 200:
#         raise Exception("Failed to fetch SonarQube analysis")
#     return response.json()


import os
import requests
import time

SONARQUBE_URL = "http://localhost:9000"
SONARQUBE_TOKEN = ""
SONARQUBE_PROJECT = "sonarqube"
HEADERS = {
    "Authorization": f"Bearer {SONARQUBE_TOKEN}"
}

def analyze_code_with_sonarqube(code_snippet, file_name="generated_code.py"):
    # Write code to file
    with open(file_name, 'w') as f:
        f.write(code_snippet)

    # Trigger sonar-scanner
    os.system(
        f"sonar-scanner -Dsonar.projectKey={SONARQUBE_PROJECT} "
        f"-Dsonar.sources=. "
        f"-Dsonar.inclusions={file_name} "
        f"-Dsonar.host.url={SONARQUBE_URL} "
        f"-Dsonar.token={SONARQUBE_TOKEN}"
    )

    # Wait briefly to allow server to process the report
    time.sleep(3)

    # Get component metrics
    params = {
        "component": SONARQUBE_PROJECT,

        "metricKeys": "reliability_rating,security_rating,sqale_rating"
    }
    response = requests.get(
        f"{SONARQUBE_URL}/api/measures/component",
        headers=HEADERS,
        params=params
    )

    if response.status_code != 200:
        print(f"SonarQube API error: {response.status_code} - {response.text}")
        raise Exception("Failed to fetch SonarQube analysis")

    return response.json()
