submission_path=standalone_eval/sample_dev_submission.jsonl
save_path=standalone_eval/sample_dev_submission_metrics_new.json
gt_path=data/vlep_dev_release.jsonl

python standalone_eval/eval.py \
--submission_path=${submission_path} \
--gt_path=${gt_path} \
--save_path=${save_path}
