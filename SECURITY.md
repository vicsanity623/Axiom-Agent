# Security Policy

The Axiom Agent team and community take the security of our software seriously. We appreciate your efforts to responsibly disclose your findings, and we will make every effort to acknowledge your contributions.

This document outlines the process for reporting security vulnerabilities.

## Supported Versions

As a rapidly evolving project, security updates are only provided for the most recent version of the Axiom Agent available on the `main` branch. We encourage all users to stay up-to-date with the latest version.

| Version | Supported          |
| ------- | ------------------ |
| 0.5.5   | :white_check_mark: |
| < 0.4   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

We are committed to working with the community to verify and respond to any potential vulnerabilities that are reported to us. To ensure the safety of our users, we ask that you follow this process:

### Preferred Method: Private Vulnerability Reporting

The best way to report a vulnerability is through **GitHub's private vulnerability reporting feature**. This ensures that your report is delivered directly to the project maintainers in a secure and confidential manner.

1.  Navigate to the main page of the Axiom Agent repository.
2.  Under the repository name, click on the **Security** tab.
3.  In the left sidebar, click **Report a vulnerability**.
4.  Fill out the form with the details of the vulnerability. Please be as descriptive as possible.

### Alternate Method: Email

If you are unable to use the GitHub private reporting feature, you can send an email to **vicsanity623@gmail.com**. Please use a clear and descriptive subject line, such as: `SECURITY: Potential Vulnerability in Axiom Agent`.

### What to Include in Your Report

A good vulnerability report helps us resolve the issue faster. Please include:
- A clear description of the vulnerability and its potential impact.
- The version of the Axiom Agent you are using.
- Step-by-step instructions to reproduce the issue, including any necessary code snippets, configurations, or specific user inputs.
- Any proof-of-concept exploits, if available.

## The Disclosure Process

Once a vulnerability is reported, we will follow this process:

1.  **Acknowledge:** We will acknowledge receipt of your report within 3 business days.
2.  **Investigate:** We will investigate the report to confirm the vulnerability and determine its severity. We may contact you for more information if needed.
3.  **Remediate:** If the vulnerability is confirmed, we will work on a fix and prepare a patch.
4.  **Disclose:** After a patch is released, we will publicly disclose the vulnerability. We are happy to provide credit to you for the discovery and will coordinate the disclosure timeline with you.

## Scope of Vulnerabilities

We are primarily interested in vulnerabilities in the core Axiom Agent codebase located in the `src/` directory, including but not limited to:

-   Insecure handling of file I/O (e.g., path traversal when loading brain or model files).
-   Vulnerabilities in how we process or handle data from external APIs (e.g., Wikipedia).
-   Prompt injection attacks that lead to unintended, privileged, or dangerous agent behavior (especially concerning the future "Tool Use" phase).

### Out of Scope

-   Vulnerabilities in third-party dependencies (e.g., `llama-cpp-python`, `Flask`). Please report these directly to the respective project maintainers.
-   Security issues related to the underlying LLM model files themselves.
-   General security issues with the GitHub platform.

Thank you for helping to keep the Axiom Agent and its community safe.