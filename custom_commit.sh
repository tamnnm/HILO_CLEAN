#!/bin/bash
<<<<<<< HEAD
commit_message_file="/home/tamnnm/git_code/commit_message.txt"
commit_log="/home/tamnnm/git_code/commit_log.txt"
cd /home/tamnnm/git_code

if [ -f .git/index.lock ]; then
    echo "Lock file exists. Exiting script."
    exit 1
fi

# Clear the commit message file
> "$commit_message_file"

# Open the commit message file for editing
vi "$commit_message_file"

# Read the commit message from the file
commit_message=$(cat "$commit_message_file")

# Check if the commit message is empty
if [ -z "$commit_message" ]; then
    echo "Commit message is empty. Aborting commit."
    exit 1
fi

# Stage all changes
git add .

# Commit the changes with the provided commit message
git commit -m "$commit_message"

# Push the changes to the remote repository
git push -u

if git diff-index --quiet HEAD --; then
    echo "No changes to commit."
else
    end_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Changes have been committed and pushed with the message: $commit_message"
    echo "$commit_message" >> "$commit_log"
    echo "Log time: $end_time" >> "$commit_log"
fi

cd -


=======

# Configuration
REPO_DIR="/home/tamnnm/gitsync"
COMMIT_MESSAGE_FILE="$REPO_DIR/commit_message.txt"
COMMIT_LOG="$REPO_DIR/commit_log.txt"
EDITOR="${EDITOR:-vi}"  # Use system editor or fallback to vi

# Change to repo directory or exit
cd "$REPO_DIR" || {
    echo "Error: Failed to change to directory $REPO_DIR"
    exit 1
}

# Check for git lock file
if [ -f .git/index.lock ]; then
    echo "Error: Git lock file exists. Another git operation may be in progress."
    echo "If you're sure no other git process is running, you can remove .git/index.lock"
    exit 1
fi

# Run copy script
if ! bash copy_python.sh; then
    echo "Error: Failed to execute copy_python.sh"
    exit 1
fi

# Check if there are changes to commit
if git diff --quiet --cached && git diff --quiet; then
    echo "No changes detected to commit."
    exit 0
fi

# Prepare commit message file
echo "# Please enter the commit message for your changes." > "$COMMIT_MESSAGE_FILE"
echo "# Lines starting with '#' will be ignored." >> "$COMMIT_MESSAGE_FILE"
echo -e "\n# Changes to be committed:" >> "$COMMIT_MESSAGE_FILE"
git diff --cached --name-status >> "$COMMIT_MESSAGE_FILE"

# Open editor for commit message
$EDITOR "$COMMIT_MESSAGE_FILE"

# Read and validate commit message
commit_message=$(grep -v '^#' "$COMMIT_MESSAGE_FILE" | sed '/^$/d')
if [ -z "$commit_message" ]; then
    echo "Error: Empty commit message. Aborting."
    exit 1
fi

# Perform git operations
if ! git add .; then
    echo "Error: Failed to stage changes"
    exit 1
fi

if ! git commit -m "$commit_message"; then
    echo "Error: Commit failed"
    exit 1
fi

if ! git push; then
    echo "Error: Push failed"
    exit 1
fi

# Log successful commit
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "[$timestamp] $commit_message" >> "$COMMIT_LOG"
echo "Successfully committed and pushed changes with message:"
echo "$commit_message"
>>>>>>> c80f4457 (First commit)
