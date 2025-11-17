#!/bin/bash
# Search git dangling blobs for Tucker-CAM files

git fsck --lost-found 2>&1 | grep "dangling blob" | head -100 | while read type blob; do
    if git cat-file -p $blob 2>/dev/null | head -20 | grep -q "TuckerFastCAMDAG\|from_pandas_dynamic_cam\|class.*Tucker"; then
        echo "===== FOUND TUCKER/CAM CODE IN $blob ====="
        filename=$(git cat-file -p $blob | head -50 | grep -E "^#.*\.py$|^\"\"\".*Tucker|^class " | head -1)
        echo "Possible file: $filename"
        echo "First 50 lines:"
        git cat-file -p $blob | head -50
        echo ""
        echo "===== END $blob ====="
        echo ""
    fi
done
