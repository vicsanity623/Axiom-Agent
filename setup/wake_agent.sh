#!/bin/bash
# wake_agent.sh
# THIS IS NOT BEING USED AS OF 09/27/2025
# This script sends a single, silent HTTP request to the agent's status endpoint
# to trigger its initialization and start the autonomous learning cycles.

echo "Sending wake-up ping to Axiom Agent..."
curl --silent --output /dev/null http://127.0.0.1:7500/status
echo "Wake-up ping sent."
