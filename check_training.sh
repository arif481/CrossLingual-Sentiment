#!/bin/bash
cd /home/stressuwu/Desktop/CrossLingual-Sentiment/crosslingual-sentiment
echo "=== Training Progress ==="
cat training.log | grep -oP "\\d+%.*\\d+s/it" | tail -1
echo ""
echo "=== Process Status ==="
ps aux | grep "src.train" | grep -v grep | head -1 | awk "{print \"CPU:\", \$3\"%\", \"MEM:\", \$4\"%\"}"

