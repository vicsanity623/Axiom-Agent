#!/bin/bash
# monitor.sh
#
# This is the primary script for monitoring the Axiom Agent.
# It ensures the agent's service is running and then connects
# to its live output log, waiting for the log to be created if needed.

SERVICE_NAME="com.axiomagent.app"
LOG_FILE="axiom_agent_output.log"

echo "--- Axiom Agent Mission Control ---"

# Step 1: Ensure the agent service is started.
# If it's already running, this command does nothing.
echo "Pinging the agent service to ensure it is running..."
launchctl start $SERVICE_NAME

# Step 2: Wait for the log file to exist before trying to tail it.
# This prevents a "file not found" error on a fresh start.
echo "Waiting for live log stream at '$LOG_FILE'..."
while ! [ -f "$LOG_FILE" ]; do
    sleep 1
done

# Step 3: Connect to the live log stream.
# The -f flag means "follow," printing new lines as they appear.
echo "Connecting to live log stream. Press CTRL+C to disconnect."
echo "----------------------------------------------------"
tail -f "$LOG_FILE"
