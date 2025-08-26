#!/bin/bash

# Enhanced Configuration
REPO_DIR="/home/tamnnm/gitsync"
COMMIT_MESSAGE="Automated commit $(date +'%Y-%m-%d %H:%M:%S')"
COMMIT_LOG="$REPO_DIR/commit_log.txt"
MAX_RETRIES=3
LOCK_WAIT_SECONDS=5
SUBMODULE_PATH="startup/tmux-config"

# Functions
cleanup() {
    if [ -f "$REPO_DIR/.git/index.lock" ]; then
        echo "Cleaning up lock file..."
        rm -f "$REPO_DIR/.git/index.lock"
    fi
}

check_lock() {
    for ((i=1; i<=MAX_RETRIES; i++)); do
        if [ -f .git/index.lock ]; then
            if [ $i -eq $MAX_RETRIES ]; then
                echo "Git lock file persists after $MAX_RETRIES attempts. Aborting."
                exit 1
            fi
            echo "Lock file detected (attempt $i/$MAX_RETRIES), waiting..."
            sleep $LOCK_WAIT_SECONDS
        else
            break
        fi
    done
}

handle_submodules() {
    # Check if there are submodule changes
    if git submodule status | grep -q '^+'; then
        echo "Found modified submodules"
        git submodule update --init --recursive
        git add "$SUBMODULE_PATH"
        git commit -m "Update submodule $SUBMODULE_PATH"
    fi
}

# Main execution
cd "$REPO_DIR" || { echo "Failed to change directory"; exit 1; }

check_lock

# Run copy script (ensure it has proper error handling)
if ! sh copy_python.sh; then
    echo "Error in copy_python.sh"
fi

# Handle submodules first
handle_submodules

# Main Git operations
git add . || { echo "git add failed"; cleanup; exit 1; }

if git diff-index --quiet HEAD --; then
    echo "No changes to commit."
else
    if git commit -m "$COMMIT_MESSAGE" && git push; then
        echo "Changes committed and pushed: $COMMIT_MESSAGE"
        echo "$(date +'%Y-%m-%d %H:%M:%S') - $COMMIT_MESSAGE" >> "$COMMIT_LOG"
    else
        echo "Commit/push failed"
        cleanup
        exit 1
    fi
fi
