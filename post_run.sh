#!/bin/bash

# Define hosts from config.yaml in order (rank 0, 1, 2, 3)
hosts=("james" "mike" "s17" "s18")

# Rsync model_output directory from rank 0 (james)
echo "Syncing model_output/ from james (rank 0)..."
rsync -avz --delete james:/Users/Shared/mlx-train/model_output/ ./model_output/

# # SCP g_s.dot files from all hosts
# for i in "${!hosts[@]}"; do
#   host="${hosts[$i]}"
#   echo "Downloading g_s.dot from $host (rank $i)..."
#   scp "$host:/Users/Shared/mlx-train/g_s.dot" "./$i.dot"
# done
#
# # Convert all dot files to PNG
# for i in "${!hosts[@]}"; do
#   if [ -f "$i.dot" ]; then
#     echo "Converting $i.dot to PNG..."
#     dot -Tpng "$i.dot" -o "$i.png"
#   fi
# done
#
# # Open all PNG files
# for i in "${!hosts[@]}"; do
#   if [ -f "$i.png" ]; then
#     echo "Opening $i.png..."
#     open "$i.png"
#   fi
# done
