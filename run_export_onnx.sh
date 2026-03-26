#!/bin/bash
source venv/bin/activate

# models/ 폴더에서 .pth 파일 목록
PTH_FILES=(models/*.pth)

if [ ${#PTH_FILES[@]} -eq 0 ] || [ ! -e "${PTH_FILES[0]}" ]; then
    echo "No .pth files found in models/"
    exit 1
fi

echo ""
echo "Available models:"
echo ""
for i in "${!PTH_FILES[@]}"; do
    SIZE=$(du -h "${PTH_FILES[$i]}" | cut -f1)
    echo "  [$((i+1))] $(basename "${PTH_FILES[$i]}") ($SIZE)"
done

echo ""
read -p "Select model (1-${#PTH_FILES[@]}): " CHOICE

if ! [[ "$CHOICE" =~ ^[0-9]+$ ]] || [ "$CHOICE" -lt 1 ] || [ "$CHOICE" -gt ${#PTH_FILES[@]} ]; then
    echo "Invalid choice."
    exit 1
fi

SELECTED="${PTH_FILES[$((CHOICE-1))]}"
BASENAME=$(basename "$SELECTED" .pth)
OUTPUT="models/${BASENAME}.onnx"

echo ""
echo "Converting: $SELECTED -> $OUTPUT"
echo ""

python src/export_onnx.py --checkpoint "$SELECTED" --output "$OUTPUT"
