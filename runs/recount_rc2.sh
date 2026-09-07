cd /work/aupai || exit 1
for d in code_dedup08 cot_fable; do
  echo "=== $d ==="
  [ -d "data/corpus/$d" ] && python3 runs/count_dir.py "data/corpus/$d" 32 2>&1 | tail -3 || echo "ABSENT"
done
echo "=== RC2 DONE ==="
