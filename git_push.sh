#!/bin/bash

# Checking if commit message is supplied
if [ -z "$1" ]
then
    echo "No commit message provided, exiting."
    exit 1
fi

# Add all changes to the staging area
git add .

# Commit the changes
git commit -m "$1"

# Push the changes to the 'main' branch
git push origin main
