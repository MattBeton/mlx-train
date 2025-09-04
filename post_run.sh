#!/bin/bash

# 1. SCP forwards.dot from james to local machine
echo "Copying forwards.dot from james..."
scp james:/Users/Shared/mlx-train/forwards.dot ./james.dot

# 2. Export dot to PNG
echo "Converting james.dot to PNG..."
dot -Tpng james.dot -o james.png

# 3. Open the dot file
echo "Opening james.dot..."
open james.png

echo "Post-run processing complete!"

