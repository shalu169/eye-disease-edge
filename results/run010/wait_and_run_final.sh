cd /Users/piuaggawal/Documents/my_workspace/eye-disease-edge-ai
until grep -q '"updated_after_all_seeds": true' results/run008_FINAL_CHECKPOINT.json 2>/dev/null; do sleep 300; done
echo "[$(date '+%F %T')] final flag seen" > results/run010/log_final_driver.txt
bash results/run010_run_final.sh >> results/run010/log_final_driver.txt 2>&1
echo "EXIT $?" >> results/run010/log_final_driver.txt
